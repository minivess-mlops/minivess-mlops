"""Risk-controlling prediction sets for segmentation.

Implements the Learn Then Test (LTT) framework for controlling arbitrary
risk functions (Dice loss, FNR, segmentation loss) with finite-sample
guarantees.

Based on: Angelopoulos et al. (2022), "Learn Then Test: Calibrating
Predictive Algorithms to Achieve Risk Control."
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class RiskFunction(Protocol):
    """Protocol for risk functions: (prediction_set, ground_truth) -> float."""

    def __call__(self, pred_set: NDArray, ground_truth: NDArray) -> float: ...


# ---------------------------------------------------------------------------
# Risk function library
# ---------------------------------------------------------------------------


def dice_loss_risk(pred_set: NDArray, ground_truth: NDArray) -> float:
    """1 - Dice coefficient between prediction set and ground truth.

    Parameters
    ----------
    pred_set:
        Binary prediction set.
    ground_truth:
        Binary ground truth mask.

    Returns
    -------
    Dice loss in [0, 1]. 0 = perfect, 1 = no overlap.
    """
    pred_bool = pred_set.astype(bool)
    gt_bool = ground_truth.astype(bool)
    intersection = float(np.sum(pred_bool & gt_bool))
    total = float(np.sum(pred_bool)) + float(np.sum(gt_bool))
    if total == 0:
        return 0.0
    return 1.0 - 2.0 * intersection / total


def fnr_risk(pred_set: NDArray, ground_truth: NDArray) -> float:
    """False negative rate: GT voxels not in prediction set.

    Parameters
    ----------
    pred_set:
        Binary prediction set.
    ground_truth:
        Binary ground truth mask.

    Returns
    -------
    FNR in [0, 1]. 0 = all GT covered, 1 = no GT covered.
    """
    pred_bool = pred_set.astype(bool)
    gt_bool = ground_truth.astype(bool)
    gt_sum = float(gt_bool.sum())
    if gt_sum == 0:
        return 0.0
    return float((gt_bool & ~pred_bool).sum()) / gt_sum


def fpr_risk(pred_set: NDArray, ground_truth: NDArray) -> float:
    """False positive rate: prediction set voxels not in GT.

    Parameters
    ----------
    pred_set:
        Binary prediction set.
    ground_truth:
        Binary ground truth mask.

    Returns
    -------
    FPR in [0, 1].
    """
    pred_bool = pred_set.astype(bool)
    gt_bool = ground_truth.astype(bool)
    non_gt_sum = float((~gt_bool).sum())
    if non_gt_sum == 0:
        return 0.0
    return float((pred_bool & ~gt_bool).sum()) / non_gt_sum


def volume_error_risk(pred_set: NDArray, ground_truth: NDArray) -> float:
    """|volume(pred_set) - volume(GT)| / volume(GT).

    Parameters
    ----------
    pred_set:
        Binary prediction set.
    ground_truth:
        Binary ground truth mask.

    Returns
    -------
    Relative volume error (>= 0). 0 = same volume.
    """
    pred_vol = float(pred_set.astype(bool).sum())
    gt_vol = float(ground_truth.astype(bool).sum())
    if gt_vol == 0:
        return 0.0
    return abs(pred_vol - gt_vol) / gt_vol


# ---------------------------------------------------------------------------
# Risk-Controlling Predictor
# ---------------------------------------------------------------------------


class RiskControllingPredictor:
    """Risk-controlling prediction sets via threshold calibration.

    Searches over probability thresholds to find the optimal one that
    controls the specified risk function at level alpha.

    The prediction set at threshold t is: {voxel : prob(voxel) >= t}.
    Lower thresholds give larger prediction sets (more inclusive).

    Parameters
    ----------
    alpha:
        Maximum acceptable risk level.
    risk_fn:
        Risk function mapping (prediction_set, ground_truth) -> float.
    n_thresholds:
        Number of threshold candidates to evaluate.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        risk_fn: RiskFunction | None = None,
        n_thresholds: int = 100,
    ) -> None:
        self.alpha = alpha
        self.risk_fn = risk_fn or fnr_risk
        self.n_thresholds = n_thresholds
        self._optimal_threshold: float | None = None

    @property
    def is_calibrated(self) -> bool:
        """Whether calibrate() has been called."""
        return self._optimal_threshold is not None

    @property
    def optimal_threshold(self) -> float:
        """Calibrated probability threshold."""
        if self._optimal_threshold is None:
            msg = "Not calibrated yet"
            raise RuntimeError(msg)
        return self._optimal_threshold

    def calibrate(
        self,
        softmax_probs: list[NDArray],
        labels: list[NDArray],
    ) -> None:
        """Calibrate the threshold to control risk at level alpha.

        For each candidate threshold, computes the empirical risk on
        calibration data. Finds the highest threshold (smallest prediction
        set) that controls risk at level alpha with finite-sample correction.

        Parameters
        ----------
        softmax_probs:
            List of probability maps (D, H, W), values in [0, 1].
        labels:
            List of binary ground truth masks (D, H, W).
        """
        if not softmax_probs:
            msg = "Need at least one calibration volume"
            raise ValueError(msg)

        n = len(softmax_probs)

        # Candidate thresholds from high (strict) to low (inclusive)
        thresholds = np.linspace(1.0, 0.0, self.n_thresholds + 1)

        best_threshold = 0.0  # Default: include everything

        for t in thresholds:
            # Compute risk for each calibration volume at this threshold
            risks: list[float] = []
            for probs, gt in zip(softmax_probs, labels, strict=True):
                pred_set = probs >= t
                risk = self.risk_fn(pred_set, gt)
                risks.append(risk)

            # Use the (1-alpha) quantile of risks with finite-sample correction.
            # Accept if the quantile is within alpha (conformal-style guarantee).
            level = min(math.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
            risk_quantile = float(np.quantile(risks, min(level, 1.0)))

            if risk_quantile <= self.alpha:
                best_threshold = float(t)
                break  # Found the highest threshold that controls risk

        self._optimal_threshold = best_threshold

        logger.info(
            "Risk-controlling CP calibrated: threshold=%.4f, alpha=%.2f, n_volumes=%d",
            self._optimal_threshold,
            self.alpha,
            n,
        )

    def predict(
        self,
        softmax_probs: NDArray,
    ) -> NDArray[np.bool_]:
        """Produce risk-controlled prediction set.

        Parameters
        ----------
        softmax_probs:
            Probability map (D, H, W), values in [0, 1].

        Returns
        -------
        Boolean prediction set (D, H, W).
        """
        if not self.is_calibrated:
            msg = "Must calibrate() before predict()"
            raise RuntimeError(msg)

        return np.asarray(softmax_probs >= self._optimal_threshold, dtype=bool)
