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

import networkx as nx
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
# APLS (Average Path Length Similarity) — Issue #124
# ---------------------------------------------------------------------------


def compute_apls(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> float:
    """Compute Average Path Length Similarity between pred and GT vessel graphs.

    Extracts vessel graphs from both masks, matches endpoint nodes,
    and compares shortest path lengths between matched node pairs.

    Uses networkx.shortest_path_length for path computation and
    scipy.optimize.linear_sum_assignment for node matching.

    Parameters
    ----------
    pred_mask:
        Binary 3D prediction mask.
    gt_mask:
        Binary 3D ground truth mask.

    Returns
    -------
    APLS score in [0, 1]. 1.0 = perfect path length agreement.
    """
    from minivess.pipeline.vessel_graph import extract_vessel_graph

    pred_bin = (pred_mask > 0).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)

    # Handle empty masks
    if pred_bin.sum() == 0 and gt_bin.sum() == 0:
        return 1.0
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return 0.0

    # Extract vessel graphs
    pred_graph = extract_vessel_graph(pred_bin, min_branch_length=0)
    gt_graph = extract_vessel_graph(gt_bin, min_branch_length=0)

    if pred_graph.number_of_nodes() < 2 or gt_graph.number_of_nodes() < 2:
        # Need at least 2 nodes for path comparison
        if pred_graph.number_of_nodes() == gt_graph.number_of_nodes():
            return 1.0
        return 0.0

    # Get endpoint nodes
    gt_endpoints = [
        n for n, d in gt_graph.nodes(data=True) if d.get("node_type") == "endpoint"
    ]
    pred_endpoints = [
        n for n, d in pred_graph.nodes(data=True) if d.get("node_type") == "endpoint"
    ]

    if len(gt_endpoints) < 2 or len(pred_endpoints) < 2:
        return 0.0

    # Match endpoints by spatial proximity (Hungarian algorithm)
    gt_coords = np.array(
        [
            [gt_graph.nodes[n]["z"], gt_graph.nodes[n]["y"], gt_graph.nodes[n]["x"]]
            for n in gt_endpoints
        ]
    )
    pred_coords = np.array(
        [
            [
                pred_graph.nodes[n]["z"],
                pred_graph.nodes[n]["y"],
                pred_graph.nodes[n]["x"],
            ]
            for n in pred_endpoints
        ]
    )

    # Cost matrix: Euclidean distance between endpoints
    from scipy.spatial.distance import cdist

    cost_matrix = cdist(gt_coords, pred_coords, metric="euclidean")
    gt_idx, pred_idx = linear_sum_assignment(cost_matrix)

    # Penalize poor endpoint matches (far apart = not the same structure)
    max_match_dist = max(pred_mask.shape) * 0.3  # 30% of volume diagonal
    match_penalties = []
    for gi, pi in zip(gt_idx, pred_idx, strict=True):
        dist = cost_matrix[gi, pi]
        if dist > max_match_dist:
            match_penalties.append(0.0)
        else:
            match_penalties.append(1.0 - dist / max_match_dist)

    if not match_penalties or np.mean(match_penalties) < 0.1:
        return 0.0

    # Compute path length similarity for matched endpoint pairs
    similarities = []
    matched_gt = [gt_endpoints[i] for i in gt_idx]
    matched_pred = [pred_endpoints[i] for i in pred_idx]

    # For each pair of matched endpoints, compare shortest paths
    for i in range(len(matched_gt)):
        for j in range(i + 1, len(matched_gt)):
            gt_src, gt_dst = matched_gt[i], matched_gt[j]
            pred_src, pred_dst = matched_pred[i], matched_pred[j]

            try:
                gt_path_len = nx.shortest_path_length(
                    gt_graph, gt_src, gt_dst, weight="length"
                )
            except nx.NetworkXNoPath:
                continue

            try:
                pred_path_len = nx.shortest_path_length(
                    pred_graph, pred_src, pred_dst, weight="length"
                )
            except nx.NetworkXNoPath:
                similarities.append(0.0)
                continue

            if gt_path_len == 0 and pred_path_len == 0:
                similarities.append(1.0)
            elif gt_path_len == 0:
                similarities.append(0.0)
            else:
                ratio = min(pred_path_len, gt_path_len) / max(
                    pred_path_len, gt_path_len
                )
                similarities.append(ratio)

    if not similarities:
        return 0.0

    return float(np.mean(similarities))


# ---------------------------------------------------------------------------
# Skeleton Recall Metric — Issue #124
# ---------------------------------------------------------------------------


def compute_skeleton_recall(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    *,
    tolerance: int = 2,
) -> float:
    """Compute skeleton recall: fraction of GT skeleton covered by prediction.

    Extracts true morphological skeleton of GT via skimage.morphology.skeletonize,
    then measures what fraction of GT skeleton voxels are within `tolerance`
    voxels of the prediction foreground.

    Parameters
    ----------
    pred_mask:
        Binary 3D prediction mask.
    gt_mask:
        Binary 3D ground truth mask.
    tolerance:
        Distance tolerance in voxels. A GT skeleton voxel is "covered"
        if it's within this distance of any prediction foreground voxel.

    Returns
    -------
    Recall in [0, 1]. 1.0 = all GT skeleton voxels covered.
    """
    from skimage.morphology import skeletonize

    pred_bin = (pred_mask > 0).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)

    if gt_bin.sum() == 0:
        return 1.0

    # Compute true GT skeleton via skimage
    gt_skeleton = skeletonize(gt_bin).astype(np.uint8)

    if gt_skeleton.sum() == 0:
        # Thin structure: fall back to the mask itself
        gt_skeleton = gt_bin

    n_gt_skel = int(gt_skeleton.sum())

    if pred_bin.sum() == 0:
        return 0.0

    # Compute distance transform of prediction foreground
    pred_dt = ndimage.distance_transform_edt(1 - pred_bin)

    # Count GT skeleton voxels within tolerance of prediction
    gt_skel_coords = np.argwhere(gt_skeleton > 0)
    covered = 0
    for coord in gt_skel_coords:
        if pred_dt[coord[0], coord[1], coord[2]] <= tolerance:
            covered += 1

    return float(covered) / n_gt_skel


# ---------------------------------------------------------------------------
# Branch Detection Rate (BDR) — Issue #124
# ---------------------------------------------------------------------------


def compute_bdr(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    *,
    length_tolerance: float = 0.5,
    endpoint_tolerance: float = 5.0,
) -> float:
    """Compute Branch Detection Rate: fraction of GT branches matched in prediction.

    Extracts vessel graphs from both masks via skan, then matches branches
    (edges) based on endpoint proximity using Hungarian algorithm.

    A GT branch is "detected" if a matching prediction branch exists with:
    - Both endpoints within `endpoint_tolerance` voxels
    - Similar length (within `length_tolerance` relative difference)

    Parameters
    ----------
    pred_mask:
        Binary 3D prediction mask.
    gt_mask:
        Binary 3D ground truth mask.
    length_tolerance:
        Maximum relative length difference for branch matching.
    endpoint_tolerance:
        Maximum Euclidean distance for endpoint matching.

    Returns
    -------
    BDR in [0, 1]. 1.0 = all GT branches detected.
    """
    from minivess.pipeline.vessel_graph import extract_vessel_graph

    pred_bin = (pred_mask > 0).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)

    if gt_bin.sum() == 0 and pred_bin.sum() == 0:
        return 1.0
    if gt_bin.sum() == 0:
        return 1.0
    if pred_bin.sum() == 0:
        return 0.0

    gt_graph = extract_vessel_graph(gt_bin, min_branch_length=0)
    pred_graph = extract_vessel_graph(pred_bin, min_branch_length=0)

    gt_edges = list(gt_graph.edges(data=True))
    pred_edges = list(pred_graph.edges(data=True))

    if len(gt_edges) == 0:
        return 1.0
    if len(pred_edges) == 0:
        return 0.0

    # Match branches by endpoint proximity
    n_gt = len(gt_edges)
    n_pred = len(pred_edges)

    def _edge_endpoints(
        graph: nx.Graph, src: int, dst: int
    ) -> tuple[np.ndarray, np.ndarray]:
        s = np.array(
            [graph.nodes[src]["z"], graph.nodes[src]["y"], graph.nodes[src]["x"]],
            dtype=float,
        )
        d = np.array(
            [graph.nodes[dst]["z"], graph.nodes[dst]["y"], graph.nodes[dst]["x"]],
            dtype=float,
        )
        return s, d

    # Build cost matrix
    cost = np.full((n_gt, n_pred), 1e6)
    for i, (gs, gd, gdata) in enumerate(gt_edges):
        gs_coord, gd_coord = _edge_endpoints(gt_graph, gs, gd)
        for j, (ps, pd, pdata) in enumerate(pred_edges):
            ps_coord, pd_coord = _edge_endpoints(pred_graph, ps, pd)

            # Try both orientations
            dist_fwd = np.linalg.norm(gs_coord - ps_coord) + np.linalg.norm(
                gd_coord - pd_coord
            )
            dist_rev = np.linalg.norm(gs_coord - pd_coord) + np.linalg.norm(
                gd_coord - ps_coord
            )
            min_dist = min(float(dist_fwd), float(dist_rev))

            if min_dist <= 2 * endpoint_tolerance:
                # Check length similarity
                gl = gdata.get("length", 1.0)
                pl = pdata.get("length", 1.0)
                len_ratio = abs(gl - pl) / gl if gl > 0 else 0.0

                if len_ratio <= length_tolerance:
                    cost[i, j] = min_dist

    gt_idx, pred_idx = linear_sum_assignment(cost)
    detected = sum(
        1 for gi, pi in zip(gt_idx, pred_idx, strict=True) if cost[gi, pi] < 1e5
    )

    return float(detected) / n_gt


# ---------------------------------------------------------------------------
# Murray's Law Compliance — Issue #131
# ---------------------------------------------------------------------------


def compute_murray_compliance(
    mask: np.ndarray,
) -> dict[str, float]:
    """Compute Murray's law compliance at vessel bifurcations.

    EXPERIMENTAL: Uses centreline graph extraction to find bifurcation nodes,
    then measures how well vessel radii follow Murray's cubic branching law:
        r_parent^3 = sum(r_daughter_i^3)

    Murray's law predicts optimal vascular branching for minimizing the
    metabolic cost of blood transport (Murray, 1926).

    Args:
        mask: Binary 3D mask (D, H, W).

    Returns:
        Dict with:
          'mean_ratio': Mean Murray ratio across bifurcations (1.0 = perfect).
          'mean_deviation': Mean absolute deviation from ideal ratio of 1.0.
          'compliance_score': Normalized score in [0, 1] (1.0 = perfect Murray's law).
          'n_bifurcations': Number of bifurcations analysed.
    """
    from minivess.pipeline.centreline_extraction import extract_centreline

    logger.warning(
        "Murray's law compliance is EXPERIMENTAL — uses centreline graph radii "
        "which are estimated from distance transform, not true vessel diameters. "
        "See docs/planning/novel-loss-debugging-plan.xml."
    )

    mask_bin = (mask > 0).astype(np.uint8)
    if mask_bin.sum() == 0:
        return {
            "mean_ratio": float("nan"),
            "mean_deviation": float("nan"),
            "compliance_score": 0.0,
            "n_bifurcations": 0,
        }

    graph = extract_centreline(mask_bin)

    # Find junction nodes (bifurcations) and their connected edges
    junction_indices = {
        i for i, n in enumerate(graph.nodes) if n.node_type == "junction"
    }

    if not junction_indices:
        return {
            "mean_ratio": float("nan"),
            "mean_deviation": float("nan"),
            "compliance_score": 0.0,
            "n_bifurcations": 0,
        }

    # Build adjacency: node_idx → list of (neighbor_idx, edge)
    adjacency: dict[int, list[tuple[int, float]]] = {}
    for edge in graph.edges:
        adjacency.setdefault(edge.source_idx, []).append(
            (edge.target_idx, edge.mean_radius)
        )
        adjacency.setdefault(edge.target_idx, []).append(
            (edge.source_idx, edge.mean_radius)
        )

    ratios = []
    for jidx in junction_indices:
        neighbors = adjacency.get(jidx, [])
        if len(neighbors) < 3:
            # Not a true bifurcation (need parent + 2 daughters)
            continue

        # Sort neighbors by radius (largest = parent, rest = daughters)
        neighbor_radii = sorted([r for _, r in neighbors], reverse=True)
        parent_r = neighbor_radii[0]
        daughter_radii = neighbor_radii[1:]

        if parent_r <= 0:
            continue

        parent_cubed = parent_r**3
        daughters_cubed_sum = sum(r**3 for r in daughter_radii)

        if daughters_cubed_sum > 0:
            ratio = parent_cubed / daughters_cubed_sum
            ratios.append(ratio)

    if not ratios:
        return {
            "mean_ratio": float("nan"),
            "mean_deviation": float("nan"),
            "compliance_score": 0.0,
            "n_bifurcations": 0,
        }

    mean_ratio = float(np.mean(ratios))
    mean_deviation = float(np.mean([abs(r - 1.0) for r in ratios]))
    compliance_score = float(1.0 / (1.0 + mean_deviation))

    return {
        "mean_ratio": mean_ratio,
        "mean_deviation": mean_deviation,
        "compliance_score": compliance_score,
        "n_bifurcations": len(ratios),
    }


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
