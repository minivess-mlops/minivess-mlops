from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Calibration metrics and transformed probabilities."""

    ece: float
    mce: float
    calibrated_probs: NDArray[np.float64] | None = None


def expected_calibration_error(
    confidences: NDArray[np.float64],
    accuracies: NDArray[np.float64],
    *,
    n_bins: int = 15,
) -> tuple[float, float]:
    """Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Args:
        confidences: Predicted confidence scores (max probability per sample).
        accuracies: Binary correctness indicator per sample.
        n_bins: Number of bins for calibration histogram.

    Returns:
        Tuple of (ECE, MCE).
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    total = len(confidences)

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (
            confidences <= bin_boundaries[i + 1]
        )
        bin_size = mask.sum()
        if bin_size == 0:
            continue

        bin_accuracy = accuracies[mask].mean()
        bin_confidence = confidences[mask].mean()
        gap = abs(bin_accuracy - bin_confidence)

        ece += (bin_size / total) * gap
        mce = max(mce, gap)

    return float(ece), float(mce)


def temperature_scale(
    logits: NDArray[np.float64],
    temperature: float,
) -> NDArray[np.float64]:
    """Apply temperature scaling to logits.

    Args:
        logits: Raw model logits, shape (N, C).
        temperature: Temperature parameter (>1 softens, <1 sharpens).

    Returns:
        Calibrated probabilities after softmax(logits/T).
    """
    scaled = logits / temperature
    # Stable softmax
    shifted = scaled - scaled.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)
