"""Conformal prediction for mask-level coverage guarantees.

Implements split conformal prediction for 3D segmentation,
providing distribution-free coverage guarantees on prediction sets.
Based on MAPIE / ConSeMa principles (Mossina & Friedrich, 2025).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class ConformalResult:
    """Result of conformal prediction."""

    prediction_sets: NDArray[np.bool_]  # (B, C, D, H, W) boolean inclusion mask
    quantile: float  # Calibrated quantile threshold
    alpha: float  # Significance level


class ConformalPredictor:
    """Split conformal prediction for 3D segmentation.

    Calibrates a quantile threshold on a held-out calibration set,
    then produces prediction sets that contain the true class with
    probability >= 1 - alpha.

    Parameters
    ----------
    alpha:
        Significance level (e.g., 0.1 for 90% coverage).
    """

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self._quantile: float | None = None

    @property
    def is_calibrated(self) -> bool:
        """Whether calibrate() has been called."""
        return self._quantile is not None

    def calibrate(
        self,
        cal_scores: NDArray[np.float32],
        cal_labels: NDArray[np.int64],
    ) -> None:
        """Calibrate the conformal predictor on a holdout set.

        Uses the nonconformity score: 1 - p(y_true) for each voxel,
        then finds the (1 - alpha) quantile of these scores.

        Parameters
        ----------
        cal_scores:
            Softmax probabilities (N, C, D, H, W).
        cal_labels:
            Ground truth class indices (N, D, H, W).
        """
        # Vectorized nonconformity scores: 1 - p(y_true) for each voxel.
        # Use np.take_along_axis to gather the true-class probability at each voxel.
        # cal_labels: (N, D, H, W) -> (N, 1, D, H, W) for gather along class axis.
        labels_idx = cal_labels[:, np.newaxis, ...]  # (N, 1, D, H, W)
        true_class_probs = np.take_along_axis(cal_scores, labels_idx, axis=1)
        # true_class_probs: (N, 1, D, H, W) -> flatten
        scores_array = (1.0 - true_class_probs).ravel()

        # Quantile with finite-sample correction
        n = len(scores_array)
        level = min((n + 1) * (1 - self.alpha) / n, 1.0)
        self._quantile = float(np.quantile(scores_array, level))

        logger.info(
            "Conformal calibrated: quantile=%.4f, alpha=%.2f, n_voxels=%d",
            self._quantile,
            self.alpha,
            n,
        )

    def predict(
        self,
        test_scores: NDArray[np.float32],
    ) -> ConformalResult:
        """Produce prediction sets with coverage guarantee.

        A class c is included in the prediction set for a voxel if
        1 - p(c) <= quantile (equivalently, p(c) >= 1 - quantile).

        Parameters
        ----------
        test_scores:
            Softmax probabilities (B, C, D, H, W).

        Returns
        -------
        ConformalResult with boolean prediction sets.
        """
        if not self.is_calibrated:
            msg = "ConformalPredictor must be calibrated before predict()"
            raise RuntimeError(msg)

        # Include class c if 1 - p(c) <= quantile, i.e., p(c) >= 1 - quantile
        threshold = 1.0 - self._quantile
        prediction_sets = test_scores >= threshold

        return ConformalResult(
            prediction_sets=prediction_sets,
            quantile=self._quantile,
            alpha=self.alpha,
        )
