from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Summary of drift detection results."""

    is_drifted: bool
    dataset_drift_score: float
    feature_scores: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


def detect_prediction_drift(
    reference: NDArray[np.float64],
    current: NDArray[np.float64],
    *,
    threshold: float = 0.05,
    method: str = "ks",
) -> DriftReport:
    """Detect drift in model predictions between reference and current data.

    Uses Kolmogorov-Smirnov test or Population Stability Index.

    Args:
        reference: Prediction scores from reference period, shape (N,).
        current: Prediction scores from current period, shape (M,).
        threshold: Significance level for drift detection.
        method: Detection method ("ks" for Kolmogorov-Smirnov, "psi" for PSI).

    Returns:
        DriftReport with detection results.
    """
    if method == "ks":
        from scipy import stats

        statistic, p_value = stats.ks_2samp(reference, current)
        is_drifted = bool(p_value < threshold)
        return DriftReport(
            is_drifted=is_drifted,
            dataset_drift_score=float(statistic),
            feature_scores={"prediction": float(statistic)},
            details={"p_value": float(p_value), "method": "ks"},
        )
    if method == "psi":
        psi = _compute_psi(reference, current)
        is_drifted = psi > 0.2  # PSI > 0.2 = significant drift
        return DriftReport(
            is_drifted=is_drifted,
            dataset_drift_score=float(psi),
            feature_scores={"prediction": float(psi)},
            details={"method": "psi"},
        )
    msg = f"Unknown drift method: {method}"
    raise ValueError(msg)


def _compute_psi(
    reference: NDArray[np.float64],
    current: NDArray[np.float64],
    *,
    n_bins: int = 10,
) -> float:
    """Compute Population Stability Index."""
    eps = 1e-6
    bins = np.linspace(
        min(float(reference.min()), float(current.min())),
        max(float(reference.max()), float(current.max())),
        n_bins + 1,
    )
    ref_hist, _ = np.histogram(reference, bins=bins)
    cur_hist, _ = np.histogram(current, bins=bins)

    ref_pct = ref_hist / max(len(reference), 1) + eps
    cur_pct = cur_hist / max(len(current), 1) + eps

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi
