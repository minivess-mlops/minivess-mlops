"""Topology-aware evaluation metrics for vascular segmentation.

Provides graph-level and boundary metrics that complement standard Dice:
- NSD (Surface Dice) — boundary quality within tolerance
- HD95 — worst-case boundary error (95th percentile)
- ccDice — connected-component Dice (fragmentation-aware)
- Betti error — topological invariant counting
- Junction F1 — bifurcation detection accuracy

All functions accept numpy arrays (binary masks) and return float scalars.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from monai.metrics import HausdorffDistanceMetric, SurfaceDiceMetric
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NSD (Surface Dice) — Issue #134
# ---------------------------------------------------------------------------


def compute_nsd(
    pred: np.ndarray,
    target: np.ndarray,
    tau: float = 1.0,
) -> float:
    """Compute Normalized Surface Distance (Surface Dice).

    Measures the fraction of the predicted boundary within tolerance `tau`
    of the ground truth boundary, and vice versa.

    Args:
        pred: Binary prediction mask (D, H, W).
        target: Binary ground truth mask (D, H, W).
        tau: Tolerance in voxels. Default 1.0 (~2 * median_voxel_spacing for MiniVess).

    Returns:
        NSD score in [0, 1]. Higher is better.
    """
    if pred.sum() == 0 or target.sum() == 0:
        if pred.sum() == 0 and target.sum() == 0:
            return 1.0
        return 0.0

    # Convert to MONAI format: (B, C, D, H, W) one-hot
    pred_tensor = _to_onehot_tensor(pred)
    target_tensor = _to_onehot_tensor(target)

    metric = SurfaceDiceMetric(
        class_thresholds=[tau],
        include_background=False,
        reduction="mean",
    )
    result = metric(pred_tensor, target_tensor)
    value = result.item()

    if np.isnan(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


# ---------------------------------------------------------------------------
# HD95 — Issue #135
# ---------------------------------------------------------------------------


def compute_hd95(
    pred: np.ndarray,
    target: np.ndarray,
) -> float:
    """Compute 95th-percentile Hausdorff Distance.

    Measures worst-case boundary error while being robust to single-voxel
    outliers.

    Args:
        pred: Binary prediction mask (D, H, W).
        target: Binary ground truth mask (D, H, W).

    Returns:
        HD95 in voxels. Lower is better. Returns inf for empty pred or target.
    """
    if pred.sum() == 0 or target.sum() == 0:
        if pred.sum() == 0 and target.sum() == 0:
            return 0.0
        return float("inf")

    pred_tensor = _to_onehot_tensor(pred)
    target_tensor = _to_onehot_tensor(target)

    metric = HausdorffDistanceMetric(
        percentile=95,
        include_background=False,
        reduction="mean",
    )
    result = metric(pred_tensor, target_tensor)
    value = result.item()

    if np.isnan(value):
        return float("inf")
    return float(value)


# ---------------------------------------------------------------------------
# ccDice (Connected-Component Dice) — Issue #112
# ---------------------------------------------------------------------------


def compute_ccdice(
    pred: np.ndarray,
    target: np.ndarray,
) -> float:
    """Compute connected-component Dice (ccDice).

    Decomposes pred and target into connected components, matches them
    via IoU using the Hungarian algorithm, and computes per-component Dice
    averaged over matched pairs. Penalises fragmentation that standard
    Dice ignores.

    Args:
        pred: Binary prediction mask (D, H, W).
        target: Binary ground truth mask (D, H, W).

    Returns:
        ccDice score in [0, 1]. Higher is better.
    """
    if pred.sum() == 0 and target.sum() == 0:
        return 1.0
    if pred.sum() == 0 or target.sum() == 0:
        return 0.0

    # Extract connected components
    pred_labels, n_pred = ndimage.label(pred)
    target_labels, n_target = ndimage.label(target)

    if n_pred == 0 or n_target == 0:
        return 0.0

    # Build IoU cost matrix for Hungarian matching
    iou_matrix = np.zeros((n_target, n_pred))
    for i in range(n_target):
        gt_mask = target_labels == (i + 1)
        for j in range(n_pred):
            pred_mask = pred_labels == (j + 1)
            intersection = np.sum(gt_mask & pred_mask)
            union = np.sum(gt_mask | pred_mask)
            if union > 0:
                iou_matrix[i, j] = intersection / union

    # Hungarian matching (maximize IoU → minimize negative IoU)
    gt_indices, pred_indices = linear_sum_assignment(-iou_matrix)

    # Compute per-component Dice for matched pairs
    total_dice = 0.0
    n_matched = 0
    for gi, pi in zip(gt_indices, pred_indices, strict=True):
        if iou_matrix[gi, pi] > 0:
            gt_mask = target_labels == (gi + 1)
            pred_mask = pred_labels == (pi + 1)
            intersection = np.sum(gt_mask & pred_mask)
            dice = 2 * intersection / (np.sum(gt_mask) + np.sum(pred_mask))
            total_dice += dice
            n_matched += 1

    # Average over all GT components (unmatched GT components contribute 0)
    return float(total_dice / n_target)


# ---------------------------------------------------------------------------
# Betti Error — Issue #113
# ---------------------------------------------------------------------------


def compute_betti_error(
    pred: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    """Compute Betti number errors.

    Beta_0 (connected components): |cc_count(pred) - cc_count(gt)|
    Beta_1 (loops): requires gudhi (optional), returns NaN if unavailable.

    Args:
        pred: Binary prediction mask (D, H, W).
        target: Binary ground truth mask (D, H, W).

    Returns:
        Dict with 'beta0_error' and 'beta1_error'.
    """
    _, n_pred = ndimage.label(pred)
    _, n_target = ndimage.label(target)

    beta0_error = abs(n_pred - n_target)

    # Beta_1 requires persistent homology (gudhi)
    beta1_error = _compute_beta1_error(pred, target)

    return {
        "beta0_error": float(beta0_error),
        "beta1_error": beta1_error,
    }


def _compute_beta1_error(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute beta_1 error using gudhi if available."""
    try:
        import gudhi  # noqa: F401

        pred_beta1 = _compute_beta1_gudhi(pred)
        target_beta1 = _compute_beta1_gudhi(target)
        return float(abs(pred_beta1 - target_beta1))
    except ImportError:
        logger.debug("gudhi not installed; beta_1 error unavailable (returning NaN)")
        return float("nan")


def _compute_beta1_gudhi(mask: np.ndarray) -> int:
    """Compute beta_1 (number of loops) using gudhi cubical complex."""
    import gudhi

    # Invert mask for sublevel filtration (foreground = 0, background = 1)
    filtration = 1.0 - mask.astype(np.float64)
    cc = gudhi.CubicalComplex(
        dimensions=list(mask.shape),
        top_dimensional_cells=filtration.ravel().tolist(),
    )
    cc.persistence()
    # Count persistent features in dimension 1 (loops)
    betti = cc.persistent_betti_numbers(0.5, 1.5)
    return betti[1] if len(betti) > 1 else 0


def compute_persistence_distance(
    pred: np.ndarray,
    target: np.ndarray,
) -> float:
    """Compute Wasserstein-1 distance between persistence diagrams.

    Requires gudhi. Returns NaN if gudhi is not installed.

    Args:
        pred: Binary prediction mask (D, H, W).
        target: Binary ground truth mask (D, H, W).

    Returns:
        Wasserstein-1 distance (float), or NaN if gudhi unavailable.
    """
    try:
        import gudhi
        from gudhi.wasserstein import wasserstein_distance

        pred_diag = _compute_persistence_diagram(pred, gudhi)
        target_diag = _compute_persistence_diagram(target, gudhi)

        if len(pred_diag) == 0 and len(target_diag) == 0:
            return 0.0
        if len(pred_diag) == 0 or len(target_diag) == 0:
            return float("inf")

        return float(wasserstein_distance(pred_diag, target_diag, order=1))
    except ImportError:
        logger.debug("gudhi not installed; persistence distance unavailable")
        return float("nan")


def _compute_persistence_diagram(mask: np.ndarray, gudhi_module: object) -> np.ndarray:
    """Compute persistence diagram using gudhi cubical complex."""
    import gudhi

    filtration = 1.0 - mask.astype(np.float64)
    cc = gudhi.CubicalComplex(
        dimensions=list(mask.shape),
        top_dimensional_cells=filtration.ravel().tolist(),
    )
    cc.persistence()
    # Get all finite persistence pairs
    pairs = cc.persistence_intervals_in_dimension(0)
    finite_pairs = pairs[np.isfinite(pairs).all(axis=1)]
    return finite_pairs


# ---------------------------------------------------------------------------
# Junction F1 — Issue #117
# ---------------------------------------------------------------------------


def compute_junction_f1(
    pred: np.ndarray,
    target: np.ndarray,
    tolerance: int = 3,
) -> dict[str, float]:
    """Compute Junction F1 score for bifurcation detection.

    Extracts junction points from predicted and ground truth centreline
    graphs, matches within distance tolerance, computes P/R/F1.

    Requires centreline_extraction module.

    Args:
        pred: Binary prediction mask (D, H, W).
        target: Binary ground truth mask (D, H, W).
        tolerance: Maximum distance (voxels) for junction matching.

    Returns:
        Dict with 'precision', 'recall', 'f1'.
    """
    from minivess.pipeline.centreline_extraction import extract_centreline

    pred_graph = extract_centreline(pred)
    target_graph = extract_centreline(target)

    pred_junctions = pred_graph.junction_coords
    target_junctions = target_graph.junction_coords

    # Handle edge cases
    if len(target_junctions) == 0 and len(pred_junctions) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if len(target_junctions) == 0:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}
    if len(pred_junctions) == 0:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0}

    # Build distance matrix
    n_target = len(target_junctions)
    n_pred = len(pred_junctions)
    dist_matrix = np.zeros((n_target, n_pred))
    for i, tj in enumerate(target_junctions):
        for j, pj in enumerate(pred_junctions):
            dist_matrix[i, j] = np.sqrt(np.sum((np.array(tj) - np.array(pj)) ** 2))

    # Hungarian matching
    gt_idx, pred_idx = linear_sum_assignment(dist_matrix)

    # Count matches within tolerance
    tp = sum(
        1
        for gi, pi in zip(gt_idx, pred_idx, strict=True)
        if dist_matrix[gi, pi] <= tolerance
    )

    precision = tp / n_pred if n_pred > 0 else 0.0
    recall = tp / n_target if n_target > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_onehot_tensor(mask: np.ndarray) -> torch.Tensor:
    """Convert binary numpy mask to MONAI-format one-hot tensor (B, C, D, H, W)."""
    mask_bin = (mask > 0).astype(np.float32)
    # Create 2-channel one-hot: [background, foreground]
    bg = 1.0 - mask_bin
    fg = mask_bin
    onehot = np.stack([bg, fg], axis=0)  # (C, D, H, W)
    return torch.from_numpy(onehot[np.newaxis])  # (1, C, D, H, W)
