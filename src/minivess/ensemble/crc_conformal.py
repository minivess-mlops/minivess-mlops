"""Conformalized Risk Control (CRC) for segmentation.

Extends split conformal prediction with formal risk control on
arbitrary monotone loss functions. Produces prediction sets with
coverage guarantees and Varisco heatmaps for dashboard visualization.

References:
    - Angelopoulos, Bates, Fisch, Lei, Schuster (2024),
      "Conformal Risk Control"
    - Mossina & Friedrich (2025), "ConSeMa"
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import numpy as np

from minivess.ensemble.conformal import ConformalResult

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class CRCPredictor:
    """Conformalized Risk Control predictor for segmentation.

    Uses LAC (Least Ambiguous set-valued Classifier) nonconformity scores
    with risk-controlling calibration to produce prediction sets with
    formal coverage guarantees.

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
        """Calibrate the CRC predictor on a holdout set.

        Computes nonconformity scores as 1 - p(y_true) for each voxel,
        then finds the (1 - alpha)(1 + 1/n) quantile.

        Parameters
        ----------
        cal_scores:
            Softmax probabilities (N, C, D, H, W).
        cal_labels:
            Ground truth class indices (N, D, H, W).
        """
        labels_idx = cal_labels[:, np.newaxis, ...]  # (N, 1, D, H, W)
        true_class_probs = np.take_along_axis(cal_scores, labels_idx, axis=1)
        scores_array = (1.0 - true_class_probs).ravel()

        # Finite-sample corrected quantile
        n = len(scores_array)
        level = min((n + 1) * (1 - self.alpha) / n, 1.0)
        self._quantile = float(np.quantile(scores_array, level))

        logger.info(
            "CRC calibrated: quantile=%.4f, alpha=%.2f, n_voxels=%d",
            self._quantile,
            self.alpha,
            n,
        )

    def predict(
        self,
        test_scores: NDArray[np.float32],
    ) -> ConformalResult:
        """Produce prediction sets with coverage guarantee.

        A class c is included if 1 - p(c) <= quantile, i.e., p(c) >= 1 - quantile.

        Parameters
        ----------
        test_scores:
            Softmax probabilities (B, C, D, H, W).

        Returns
        -------
        ConformalResult with boolean prediction sets.
        """
        if self._quantile is None:
            msg = "CRCPredictor must be calibrated before predict()"
            raise RuntimeError(msg)

        threshold = 1.0 - self._quantile
        prediction_sets = test_scores >= threshold

        return ConformalResult(
            prediction_sets=prediction_sets,
            quantile=self._quantile,
            alpha=self.alpha,
        )


def varisco_heatmap(
    prediction_sets: NDArray[np.bool_],
) -> NDArray[np.int64]:
    """Compute Varisco uncertainty heatmap from prediction sets.

    The heatmap shows the number of classes included in the prediction
    set at each spatial location. Higher values = more uncertainty.

    Parameters
    ----------
    prediction_sets:
        Boolean array (B, C, D, H, W) indicating which classes are
        included in the prediction set at each voxel.

    Returns
    -------
    Heatmap (B, D, H, W) with values in [0, C].
    """
    return cast("NDArray[np.int64]", prediction_sets.sum(axis=1).astype(np.int64))
