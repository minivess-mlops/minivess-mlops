"""Evaluation-time calibration metrics for binary segmentation.

Computes Brier score, O:E ratio, IPA, and calibration slope at the voxel level.
These are logged to MLflow during the analysis flow, not used as training losses.

Tier 1 (fast, O(N)) metrics are suitable for training validation loops:
  ECE, MCE, RMSCE, Brier, NLL, Overconfidence Error, Debiased ECE.

Tier 2 (comprehensive) metrics are for the analysis flow only:
  ACE, BA-ECE, Brier map, NLL map.

References
----------
- Van Calster et al. (2016). "A calibration hierarchy for risk models." Stat Med.
- Steyerberg et al. (2010). "Assessing the performance of prediction models." Epidemiology.
- Guo et al. (2017). "On Calibration of Modern Neural Networks." ICML.
- Kumar et al. (2019). "Verified Uncertainty Calibration." NeurIPS.
- Nixon et al. (2019). "Measuring Calibration in Deep Learning." CVPR Workshops.
- Zeevi et al. (2025). "Boundary-Aware Calibration for Semantic Segmentation."
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import distance_transform_edt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from minivess.ensemble.calibration import expected_calibration_error
from minivess.pipeline.biostatistics_types import CalibrationMetricsResult

logger = logging.getLogger(__name__)


def compute_calibration_metrics(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    *,
    max_voxels: int = 100_000,
    seed: int = 42,
) -> CalibrationMetricsResult:
    """Compute calibration metrics for binary segmentation predictions.

    Parameters
    ----------
    y_true:
        Ground truth binary labels (0 or 1), flat array.
    p_pred:
        Predicted probabilities for the positive class, flat array.
    max_voxels:
        Maximum voxels to use (subsamples if exceeded for efficiency).
    seed:
        Random seed for subsampling.

    Returns
    -------
    CalibrationMetricsResult with brier_score, oe_ratio, ipa, calibration_slope.
    """
    rng = np.random.default_rng(seed)

    # Subsample if needed
    if len(y_true) > max_voxels:
        idx = rng.choice(len(y_true), size=max_voxels, replace=False)
        y_true = y_true[idx]
        p_pred = p_pred[idx]

    # Brier score: mean squared error of probabilities
    brier = float(brier_score_loss(y_true, p_pred))

    # O:E ratio: observed / expected event rate
    observed = float(y_true.sum())
    expected = float(p_pred.sum())
    oe_ratio = observed / expected if expected > 0 else 0.0

    # IPA (Index of Prediction Accuracy): 1 - Brier / Brier_null
    prevalence = float(y_true.mean())
    brier_null = prevalence * (1 - prevalence)
    ipa = 1.0 - (brier / brier_null) if brier_null > 0 else 0.0

    # Calibration slope via logistic regression of outcome on log-odds
    log_odds = np.log(
        np.clip(p_pred, 1e-7, 1 - 1e-7) / (1 - np.clip(p_pred, 1e-7, 1 - 1e-7))
    )
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(log_odds.reshape(-1, 1), y_true)
    calibration_slope = float(lr.coef_[0, 0])

    return CalibrationMetricsResult(
        brier_score=brier,
        oe_ratio=oe_ratio,
        ipa=ipa,
        calibration_slope=calibration_slope,
    )


# ---------------------------------------------------------------------------
# Helper: equal-width binning statistics
# ---------------------------------------------------------------------------


def _equal_width_bin_stats(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-bin statistics using equal-width binning.

    Returns
    -------
    bin_weights : Weight (fraction of total) per bin.
    bin_accuracies : Mean label per bin.
    bin_confidences : Mean probability per bin.
    bin_counts : Number of samples per bin.
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(probs)

    bin_weights = np.zeros(n_bins)
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
        count = int(mask.sum())
        if count == 0:
            continue
        bin_counts[i] = count
        bin_weights[i] = count / total
        bin_accuracies[i] = labels[mask].mean()
        bin_confidences[i] = probs[mask].mean()

    return bin_weights, bin_accuracies, bin_confidences, bin_counts


# ---------------------------------------------------------------------------
# Tier 1: Fast metrics — O(N), suitable for training validation loop
# ---------------------------------------------------------------------------


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error (Guo et al. 2017). Equal-width binning.

    Delegates to ``expected_calibration_error`` from ``minivess.ensemble.calibration``.

    Parameters
    ----------
    probs : Predicted probabilities (flat array).
    labels : Ground truth binary labels (flat array, same length as probs).
    n_bins : Number of equal-width bins.

    Returns
    -------
    ECE value in [0, 1].
    """
    ece, _mce = expected_calibration_error(
        confidences=probs.astype(np.float64),
        accuracies=labels.astype(np.float64),
        n_bins=n_bins,
    )
    return float(ece)


def compute_mce(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    """Maximum Calibration Error.

    The worst-case gap across all bins.  Uses ``expected_calibration_error`` internally.

    Parameters
    ----------
    probs : Predicted probabilities (flat array).
    labels : Ground truth binary labels (flat array).
    n_bins : Number of equal-width bins.

    Returns
    -------
    MCE value in [0, 1].
    """
    _ece, mce = expected_calibration_error(
        confidences=probs.astype(np.float64),
        accuracies=labels.astype(np.float64),
        n_bins=n_bins,
    )
    return float(mce)


def compute_rmsce(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    """Root Mean Squared Calibration Error.

    RMSCE = sqrt( sum_b w_b * gap_b^2 )

    Parameters
    ----------
    probs : Predicted probabilities (flat array).
    labels : Ground truth binary labels (flat array).
    n_bins : Number of equal-width bins.

    Returns
    -------
    RMSCE value in [0, 1].
    """
    bin_weights, bin_acc, bin_conf, _counts = _equal_width_bin_stats(
        probs, labels, n_bins=n_bins
    )
    gaps = np.abs(bin_acc - bin_conf)
    rmsce = float(np.sqrt(np.sum(bin_weights * gaps**2)))
    return rmsce


def compute_brier_score(
    probs: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Brier score: mean((p - y)^2).  Lower is better.

    Parameters
    ----------
    probs : Predicted probabilities (flat array).
    labels : Ground truth binary labels (flat array).

    Returns
    -------
    Brier score in [0, 1].
    """
    return float(np.mean((probs - labels) ** 2))


def compute_nll(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    eps: float = 1e-7,
) -> float:
    """Negative log-likelihood (binary cross-entropy).  Lower is better.

    NLL = -mean( y * log(p) + (1 - y) * log(1 - p) )

    Parameters
    ----------
    probs : Predicted probabilities (flat array).
    labels : Ground truth binary labels (flat array).
    eps : Small constant to avoid log(0).

    Returns
    -------
    NLL value >= 0.
    """
    p = np.clip(probs, eps, 1.0 - eps)
    nll = -np.mean(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p))
    return float(nll)


def compute_overconfidence_error(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    """Overconfidence Error (OE): only penalizes bins where confidence > accuracy.

    OE = sum_b w_b * max(0, conf_b - acc_b)

    Parameters
    ----------
    probs : Predicted probabilities (flat array).
    labels : Ground truth binary labels (flat array).
    n_bins : Number of equal-width bins.

    Returns
    -------
    OE value in [0, 1].
    """
    bin_weights, bin_acc, bin_conf, _counts = _equal_width_bin_stats(
        probs, labels, n_bins=n_bins
    )
    overconf_gaps = np.maximum(0.0, bin_conf - bin_acc)
    return float(np.sum(bin_weights * overconf_gaps))


def compute_debiased_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    """Debiased ECE (Kumar et al. 2019).  Corrects for finite-sample bias.

    D-ECE = ECE - bias, where bias approx sum_b w_b / (2 * n_b).
    Clamped at 0.

    Parameters
    ----------
    probs : Predicted probabilities (flat array).
    labels : Ground truth binary labels (flat array).
    n_bins : Number of equal-width bins.

    Returns
    -------
    Debiased ECE value >= 0.
    """
    bin_weights, bin_acc, bin_conf, bin_counts = _equal_width_bin_stats(
        probs, labels, n_bins=n_bins
    )
    gaps = np.abs(bin_acc - bin_conf)
    ece = float(np.sum(bin_weights * gaps))

    # Finite-sample bias correction: sum_b w_b / (2 * n_b)
    bias = 0.0
    for i in range(n_bins):
        if bin_counts[i] > 0:
            bias += bin_weights[i] / (2.0 * bin_counts[i])

    return float(max(0.0, ece - bias))


# ---------------------------------------------------------------------------
# Tier 2: Comprehensive metrics — analysis flow only
# ---------------------------------------------------------------------------


def compute_ace(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 20,
) -> float:
    """Adaptive Calibration Error (Nixon et al. 2019).  Equal-count bins.

    Unlike ECE (equal-width), ACE sorts samples by confidence and splits them
    into bins of approximately equal count, giving each bin equal weight.

    Parameters
    ----------
    probs : Predicted probabilities (flat array).
    labels : Ground truth binary labels (flat array).
    n_bins : Number of equal-count bins.

    Returns
    -------
    ACE value in [0, 1].
    """
    n = len(probs)
    if n == 0:
        raise ValueError("probs and labels must not be empty")

    # Sort by confidence
    order = np.argsort(probs)
    sorted_probs = probs[order]
    sorted_labels = labels[order]

    # Split into approximately equal-count bins
    bin_edges = np.array_split(np.arange(n), n_bins)

    ace = 0.0
    for indices in bin_edges:
        if len(indices) == 0:
            continue
        bin_conf = sorted_probs[indices].mean()
        bin_acc = sorted_labels[indices].mean()
        ace += abs(bin_conf - bin_acc)

    # Each bin has equal weight = 1/n_bins
    n_nonempty = sum(1 for indices in bin_edges if len(indices) > 0)
    if n_nonempty == 0:
        return 0.0
    return float(ace / n_nonempty)


def compute_ba_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    n_bands: int = 10,
    boundary_width: int = 3,
) -> float:
    """Boundary-Aware ECE (Zeevi et al. 2025).

    Computes ECE weighted by distance to the segmentation boundary.
    Voxels near boundaries get higher weight than interior voxels.

    For 3D volumes, labels should be 3D (D, H, W).  Uses
    ``scipy.ndimage.distance_transform_edt`` for boundary distance.
    For flat/1D arrays, skips spatial weighting and returns standard ECE.

    Parameters
    ----------
    probs : Predicted probabilities (flat or 3D volumetric).
    labels : Ground truth binary labels (same shape as probs).
    n_bands : Number of distance bands for boundary weighting.
    boundary_width : Width in voxels defining the "boundary zone" (weight = 1).

    Returns
    -------
    BA-ECE value in [0, 1].
    """
    # Flat / 1D input: fall back to standard ECE
    if probs.ndim <= 1:
        return compute_ece(probs.ravel(), labels.ravel())

    # Spatial input (2D or 3D): compute boundary distance weighting
    labels_binary = (labels > 0.5).astype(np.float64)

    # Distance from foreground boundary (inside) + from background boundary (outside)
    dist_inside = distance_transform_edt(labels_binary)
    dist_outside = distance_transform_edt(1.0 - labels_binary)
    dist_to_boundary = np.minimum(dist_inside, dist_outside)

    # Convert distance to weight: boundary voxels get weight 1.0,
    # weight decays linearly to a floor of 0.1 beyond boundary_width
    weight_floor = 0.1
    weights = np.where(
        dist_to_boundary <= boundary_width,
        1.0,
        np.maximum(
            weight_floor,
            1.0 - (dist_to_boundary - boundary_width) / (n_bands * boundary_width),
        ),
    )

    # Flatten everything for binned computation
    flat_probs = probs.ravel()
    flat_labels = labels.ravel()
    flat_weights = weights.ravel()

    # Normalize weights so they sum to 1
    weight_sum = flat_weights.sum()
    if weight_sum == 0:
        return 0.0
    flat_weights = flat_weights / weight_sum

    # Weighted ECE: equal-width binning with per-voxel weights
    n_bins = 15
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ba_ece = 0.0

    for i in range(n_bins):
        mask = (flat_probs > bin_boundaries[i]) & (flat_probs <= bin_boundaries[i + 1])
        if not mask.any():
            continue

        w_bin = flat_weights[mask]
        bin_weight = w_bin.sum()

        # Weighted mean accuracy and confidence within bin
        bin_acc = np.average(flat_labels[mask], weights=w_bin)
        bin_conf = np.average(flat_probs[mask], weights=w_bin)

        ba_ece += bin_weight * abs(bin_acc - bin_conf)

    return float(ba_ece)


def compute_brier_map(
    probs: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Per-voxel Brier score map: (p - y)^2.  Returns same shape as input.

    Parameters
    ----------
    probs : Predicted probabilities (any shape).
    labels : Ground truth binary labels (same shape as probs).

    Returns
    -------
    Array of per-voxel Brier scores, same shape as input.
    """
    result: np.ndarray = (probs - labels) ** 2
    return result


def compute_nll_map(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    eps: float = 1e-7,
) -> np.ndarray:
    """Per-voxel NLL map: -y*log(p) - (1-y)*log(1-p).  Returns same shape as input.

    Parameters
    ----------
    probs : Predicted probabilities (any shape).
    labels : Ground truth binary labels (same shape as probs).
    eps : Small constant to avoid log(0).

    Returns
    -------
    Array of per-voxel NLL values, same shape as input.
    """
    p = np.clip(probs, eps, 1.0 - eps)
    result: np.ndarray = -(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p))
    return result


# ---------------------------------------------------------------------------
# Convenience: compute all metrics at once
# ---------------------------------------------------------------------------


def compute_all_calibration_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    tier: str = "comprehensive",
    n_bins: int = 15,
) -> dict[str, float]:
    """Compute ALL calibration metrics at once.

    Parameters
    ----------
    probs : Predicted probabilities (flat or volumetric).
    labels : Ground truth binary labels (same shape as probs).
    tier : ``"fast"`` for training-loop metrics (Tier 1 only),
           ``"comprehensive"`` for all scalar metrics (Tier 1 + Tier 2).
    n_bins : Number of bins for binned metrics.

    Returns
    -------
    Dict mapping metric names to float values.
    Maps do NOT appear in the dict — use ``compute_brier_map`` / ``compute_nll_map``
    directly if spatial maps are needed.
    """
    if tier not in ("fast", "comprehensive"):
        raise ValueError(f"tier must be 'fast' or 'comprehensive', got {tier!r}")

    flat_probs = probs.ravel()
    flat_labels = labels.ravel()

    # Tier 1: fast metrics
    result: dict[str, float] = {
        "ece": compute_ece(flat_probs, flat_labels, n_bins=n_bins),
        "mce": compute_mce(flat_probs, flat_labels, n_bins=n_bins),
        "rmsce": compute_rmsce(flat_probs, flat_labels, n_bins=n_bins),
        "brier": compute_brier_score(flat_probs, flat_labels),
        "nll": compute_nll(flat_probs, flat_labels),
        "overconfidence_error": compute_overconfidence_error(
            flat_probs, flat_labels, n_bins=n_bins
        ),
        "debiased_ece": compute_debiased_ece(flat_probs, flat_labels, n_bins=n_bins),
    }

    if tier == "comprehensive":
        result["ace"] = compute_ace(flat_probs, flat_labels)
        result["ba_ece"] = compute_ba_ece(probs, labels)

    return result
