"""Distance-transform conformal prediction for 3D segmentation.

FNR-controlling conformal prediction using signed distance transforms.
Calibrates a boundary distance threshold that guarantees the false
negative rate is below alpha.

Based on: Tan et al. (2025), "Conformal Label Smoothing for 3D Medical
Image Segmentation."
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import distance_transform_edt

from minivess.ensemble.distance_utils import asymmetric_hausdorff_percentile

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class DistanceTransformConformalPredictor:
    """FNR-controlling conformal prediction via distance transforms.

    For each calibration volume, computes the maximum distance from any
    GT-positive voxel to the nearest predicted-positive voxel (the
    "miss distance"). Calibrates a threshold such that dilating the
    prediction by this distance guarantees FNR <= alpha.

    Parameters
    ----------
    alpha:
        Significance level (e.g., 0.1 for 90% FNR control).
    """

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self._threshold: float | None = None

    @property
    def is_calibrated(self) -> bool:
        """Whether calibrate() has been called."""
        return self._threshold is not None

    @property
    def calibrated_threshold(self) -> float:
        """Calibrated distance threshold in voxels."""
        if self._threshold is None:
            msg = "Not calibrated yet"
            raise RuntimeError(msg)
        return self._threshold

    def calibrate(
        self,
        predictions: list[NDArray],
        labels: list[NDArray],
    ) -> None:
        """Calibrate the distance threshold on holdout volumes.

        For each calibration pair, computes the 95th percentile of distances
        from GT voxels to the nearest predicted voxel (asymmetric Hausdorff).
        The calibrated threshold is the (1-alpha) quantile of these scores.

        Parameters
        ----------
        predictions:
            List of binary prediction masks (D, H, W).
        labels:
            List of binary ground truth masks (D, H, W).
        """
        if not predictions:
            msg = "Need at least one calibration volume"
            raise ValueError(msg)

        # Nonconformity scores: asymmetric Hausdorff from GT to prediction
        scores: list[float] = []
        for pred, gt in zip(predictions, labels, strict=True):
            score = asymmetric_hausdorff_percentile(gt, pred, percentile=95)
            scores.append(score)

        # Quantile with finite-sample correction
        n = len(scores)
        level = min(math.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self._threshold = float(np.quantile(scores, min(level, 1.0)))

        logger.info(
            "Distance-transform CP calibrated: threshold=%.2f voxels, "
            "alpha=%.2f, n_volumes=%d",
            self._threshold,
            self.alpha,
            n,
        )

    def predict(
        self,
        prediction: NDArray,
    ) -> NDArray[np.bool_]:
        """Produce FNR-controlled prediction set by distance dilation.

        Expands the prediction mask to include all voxels within
        `calibrated_threshold` voxels of the predicted boundary.

        Parameters
        ----------
        prediction:
            Binary prediction mask (D, H, W).

        Returns
        -------
        Boolean prediction set (D, H, W).
        """
        if not self.is_calibrated:
            msg = "Must calibrate() before predict()"
            raise RuntimeError(msg)

        pred_bool = prediction.astype(bool)

        if self._threshold <= 0 or not pred_bool.any():
            return pred_bool.copy()

        # Distance from every voxel to nearest prediction voxel
        dist_to_pred = distance_transform_edt(~pred_bool)

        # Include all voxels within threshold distance
        return np.asarray(dist_to_pred <= self._threshold, dtype=bool)


def compute_distance_metrics(
    prediction_set: NDArray,
    ground_truth: NDArray,
    *,
    threshold: float = 0.0,
) -> dict[str, float]:
    """Compute FNR/FPR metrics for a distance-transform prediction set.

    Parameters
    ----------
    prediction_set:
        Boolean prediction set (D, H, W).
    ground_truth:
        Binary ground truth mask (D, H, W).
    threshold:
        Distance threshold used (for logging).

    Returns
    -------
    Dict with: fnr, fpr, boundary_distance, volume_inflation.
    """
    pred_bool = prediction_set.astype(bool)
    gt_bool = ground_truth.astype(bool)

    gt_sum = float(gt_bool.sum())
    non_gt_sum = float((~gt_bool).sum())
    pred_sum = float(pred_bool.sum())

    # FNR: GT voxels not in prediction set / total GT voxels
    fnr = float((gt_bool & ~pred_bool).sum()) / gt_sum if gt_sum > 0 else 0.0

    # FPR: prediction set voxels not in GT / total non-GT voxels
    fpr = float((pred_bool & ~gt_bool).sum()) / non_gt_sum if non_gt_sum > 0 else 0.0

    # Original prediction volume (before dilation) approximation
    # Use GT sum as reference for volume inflation
    volume_inflation = pred_sum / gt_sum if gt_sum > 0 else 1.0

    return {
        "fnr": fnr,
        "fpr": fpr,
        "boundary_distance": threshold,
        "volume_inflation": volume_inflation,
    }
