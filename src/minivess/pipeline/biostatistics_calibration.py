"""Calibration metrics for the biostatistics flow.

Computes Brier score, O/E ratio, and IPA (Index of Prediction Accuracy)
to assess how well predicted probabilities reflect true frequencies.

These metrics are computed from the auxiliary calibration output of the
post-training flow (aux_calibration factor in the factorial design).

Pure functions — no Prefect, no Docker dependency.

References
----------
- Van Calster et al. (2019). "Calibration: the Achilles heel of
  predictive analytics." *BMC Medicine*.
- Steyerberg et al. (2010). "Assessing the performance of prediction
  models." *Epidemiology*.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationSummary:
    """Summary of calibration metrics for one set of predictions."""

    brier_score: float
    oe_ratio: float
    ipa: float  # Index of Prediction Accuracy (Brier skill score)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_brier_score(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute Brier score: mean squared error between predictions and labels.

    Lower is better. Perfect = 0.0, worst = 1.0.
    """
    return float(np.mean((predictions - labels) ** 2))


def compute_oe_ratio(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute Observed/Expected ratio.

    O/E = sum(labels) / sum(predictions).
    Perfect calibration → O/E = 1.0.
    O/E > 1 = underprediction, O/E < 1 = overprediction.
    """
    expected = float(np.sum(predictions))
    if expected == 0:
        return float("inf")
    observed = float(np.sum(labels))
    return observed / expected


def compute_ipa(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute Index of Prediction Accuracy (Brier skill score).

    IPA = 1 - Brier / Brier_ref
    where Brier_ref = prevalence * (1 - prevalence).
    IPA > 0 means predictions are better than the prevalence baseline.
    IPA = 1 is perfect, IPA = 0 is no skill.
    """
    brier = compute_brier_score(predictions, labels)
    prevalence = float(np.mean(labels))
    brier_ref = prevalence * (1 - prevalence)
    if brier_ref == 0:
        return 0.0
    return 1.0 - brier / brier_ref


def compute_calibration_summary(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> CalibrationSummary:
    """Compute all calibration metrics at once."""
    return CalibrationSummary(
        brier_score=compute_brier_score(predictions, labels),
        oe_ratio=compute_oe_ratio(predictions, labels),
        ipa=compute_ipa(predictions, labels),
    )
