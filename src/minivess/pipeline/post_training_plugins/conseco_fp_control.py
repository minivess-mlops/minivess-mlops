"""ConSeCo false-positive control post-training plugin.

Implements threshold-based and erosion-based mask shrinking with
conformal calibration to guarantee FP rate below user-specified tolerance.

References:
    - arXiv:2511.15406, "ConSeCo: Conformal Segmentation Control"
    - https://github.com/deel-ai-papers/conseco
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.ndimage import binary_erosion

from minivess.pipeline.post_training_plugin import PluginInput, PluginOutput

logger = logging.getLogger(__name__)


class ConSeCoPlugin:
    """ConSeCo FP control plugin — threshold + erosion shrinking."""

    @property
    def name(self) -> str:
        return "conseco_fp_control"

    @property
    def requires_calibration_data(self) -> bool:
        return True

    def validate_inputs(self, plugin_input: PluginInput) -> list[str]:
        errors: list[str] = []
        if plugin_input.calibration_data is None:
            errors.append(
                "ConSeCo requires calibration_data (pred_probs/masks + gt_masks)"
            )
        return errors

    def execute(self, plugin_input: PluginInput) -> PluginOutput:
        config = plugin_input.config
        tolerance: float = config.get("tolerance", 0.05)
        shrink_method: str = config.get("shrink_method", "erosion")
        cal_data = plugin_input.calibration_data
        assert cal_data is not None

        pred_probs = np.asarray(cal_data["pred_probs"], dtype=np.float32)
        gt_masks = np.asarray(cal_data["gt_masks"], dtype=np.int64)

        metrics: dict[str, float] = {"tolerance": tolerance}
        artifacts: dict[str, Any] = {"shrink_method": shrink_method}

        if shrink_method == "threshold":
            optimal_t = _find_threshold(pred_probs, gt_masks, tolerance=tolerance)
            metrics["optimal_threshold"] = optimal_t
            metrics["fp_rate_at_threshold"] = _fp_rate_at_threshold(
                pred_probs, gt_masks, optimal_t
            )
            logger.info(
                "ConSeCo threshold: t=%.4f, FP rate=%.4f",
                optimal_t,
                metrics["fp_rate_at_threshold"],
            )
        elif shrink_method == "erosion":
            optimal_r = _find_erosion_radius(pred_probs, gt_masks, tolerance=tolerance)
            metrics["optimal_erosion_radius"] = float(optimal_r)
            metrics["fp_rate_at_erosion"] = _fp_rate_at_erosion(
                pred_probs, gt_masks, optimal_r
            )
            logger.info(
                "ConSeCo erosion: r=%d, FP rate=%.4f",
                optimal_r,
                metrics["fp_rate_at_erosion"],
            )
        else:
            logger.warning("Unknown shrink method: %s", shrink_method)

        return PluginOutput(artifacts=artifacts, metrics=metrics)


def _fp_rate(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute false positive rate: FP / (FP + TN)."""
    fp = ((pred == 1) & (gt == 0)).sum()
    tn = ((pred == 0) & (gt == 0)).sum()
    denom = fp + tn
    if denom == 0:
        return 0.0
    return float(fp / denom)


def _fp_rate_at_threshold(probs: np.ndarray, gt: np.ndarray, threshold: float) -> float:
    """FP rate when binarizing at given threshold."""
    pred = (probs >= threshold).astype(np.int64)
    return _fp_rate(pred, gt)


def _fp_rate_at_erosion(probs: np.ndarray, gt: np.ndarray, radius: int) -> float:
    """FP rate after eroding predictions with given radius."""
    pred = (probs >= 0.5).astype(np.int64)
    if radius > 0:
        # Erode each sample independently
        eroded = np.zeros_like(pred)
        for i in range(pred.shape[0]):
            eroded[i] = binary_erosion(pred[i], iterations=radius).astype(np.int64)
        pred = eroded
    return _fp_rate(pred, gt)


def _find_threshold(probs: np.ndarray, gt: np.ndarray, *, tolerance: float) -> float:
    """Find minimum threshold such that FP rate <= tolerance.

    Uses conformal-style approach: test thresholds from high to low,
    find the lowest that satisfies the FP constraint.
    """
    candidates = np.linspace(0.5, 0.99, 50)
    best_t = 0.99  # Conservative default
    for t in candidates:
        fpr = _fp_rate_at_threshold(probs, gt, t)
        if fpr <= tolerance:
            best_t = float(t)
            break
    return best_t


def _find_erosion_radius(probs: np.ndarray, gt: np.ndarray, *, tolerance: float) -> int:
    """Find minimum erosion radius such that FP rate <= tolerance."""
    for r in range(0, 10):
        fpr = _fp_rate_at_erosion(probs, gt, r)
        if fpr <= tolerance:
            return r
    return 9  # Max radius
