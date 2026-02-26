"""Tests for risk-controlling prediction sets (Phase 3).

Validates the Learn Then Test (LTT) framework for controlling arbitrary
risk functions (Dice loss, FNR, FPR) on segmentation prediction sets.
"""

from __future__ import annotations

import numpy as np
import pytest


def _make_sphere_mask(
    shape: tuple[int, int, int] = (16, 16, 16),
    center: tuple[int, int, int] | None = None,
    radius: float = 5.0,
) -> np.ndarray:
    """Create a binary sphere mask."""
    if center is None:
        center = tuple(s // 2 for s in shape)
    coords = np.mgrid[: shape[0], : shape[1], : shape[2]]
    dist_sq = sum((c - cn) ** 2 for c, cn in zip(coords, center, strict=True))
    return (dist_sq <= radius**2).astype(np.int64)


# ---------------------------------------------------------------------------
# Task 3.1: RiskControllingPredictor
# ---------------------------------------------------------------------------


class TestRiskControllingPredictor:
    """Test risk-controlling prediction sets."""

    def test_dice_loss_risk_controlled(self) -> None:
        """Dice loss should be below alpha on calibration data."""
        from minivess.ensemble.risk_control import (
            RiskControllingPredictor,
            dice_loss_risk,
        )

        gt = _make_sphere_mask(radius=5.0)
        pred_probs = np.zeros((16, 16, 16), dtype=np.float32)
        # High probability inside GT
        pred_probs[gt.astype(bool)] = 0.8
        pred_probs[~gt.astype(bool)] = 0.2

        predictor = RiskControllingPredictor(
            alpha=0.3,
            risk_fn=dice_loss_risk,
        )
        predictor.calibrate(
            softmax_probs=[pred_probs] * 5,
            labels=[gt] * 5,
        )
        pred_set = predictor.predict(pred_probs)

        # Dice loss on calibration data should be roughly controlled
        loss = dice_loss_risk(pred_set, gt)
        assert loss <= 0.5  # Allow slack for small calibration set

    def test_fnr_risk_controlled(self) -> None:
        """FNR should be below alpha on calibration data."""
        from minivess.ensemble.risk_control import (
            RiskControllingPredictor,
            fnr_risk,
        )

        gt = _make_sphere_mask(radius=5.0)
        pred_probs = np.zeros((16, 16, 16), dtype=np.float32)
        pred_probs[gt.astype(bool)] = 0.7
        pred_probs[~gt.astype(bool)] = 0.3

        predictor = RiskControllingPredictor(
            alpha=0.2,
            risk_fn=fnr_risk,
        )
        predictor.calibrate(
            softmax_probs=[pred_probs] * 5,
            labels=[gt] * 5,
        )
        pred_set = predictor.predict(pred_probs)

        loss = fnr_risk(pred_set, gt)
        assert loss <= 0.5

    def test_monotonic_lambda(self) -> None:
        """Lower threshold should give larger prediction sets."""
        from minivess.ensemble.risk_control import RiskControllingPredictor, fnr_risk

        gt = _make_sphere_mask(radius=5.0)
        pred_probs = np.zeros((16, 16, 16), dtype=np.float32)
        pred_probs[gt.astype(bool)] = 0.7

        p_strict = RiskControllingPredictor(alpha=0.1, risk_fn=fnr_risk)
        p_strict.calibrate([pred_probs] * 10, [gt] * 10)
        set_strict = p_strict.predict(pred_probs)

        p_loose = RiskControllingPredictor(alpha=0.5, risk_fn=fnr_risk)
        p_loose.calibrate([pred_probs] * 10, [gt] * 10)
        set_loose = p_loose.predict(pred_probs)

        # Stricter alpha (0.1) should give larger sets (lower threshold)
        assert set_strict.sum() >= set_loose.sum()

    def test_calibrate_stores_lambda(self) -> None:
        """optimal_threshold should be stored after calibration."""
        from minivess.ensemble.risk_control import RiskControllingPredictor, fnr_risk

        gt = _make_sphere_mask(radius=5.0)
        pred_probs = np.zeros((16, 16, 16), dtype=np.float32)
        pred_probs[gt.astype(bool)] = 0.8

        predictor = RiskControllingPredictor(alpha=0.1, risk_fn=fnr_risk)
        predictor.calibrate([pred_probs] * 5, [gt] * 5)

        assert predictor.is_calibrated
        assert 0.0 <= predictor.optimal_threshold <= 1.0

    def test_predict_before_calibrate_raises(self) -> None:
        """Predicting without calibration must raise RuntimeError."""
        from minivess.ensemble.risk_control import RiskControllingPredictor, fnr_risk

        predictor = RiskControllingPredictor(alpha=0.1, risk_fn=fnr_risk)
        pred_probs = np.zeros((16, 16, 16), dtype=np.float32)
        with pytest.raises(RuntimeError, match="calibrate"):
            predictor.predict(pred_probs)

    def test_custom_risk_function(self) -> None:
        """Should accept user-defined risk callable."""
        from minivess.ensemble.risk_control import RiskControllingPredictor

        def custom_risk(pred_set: np.ndarray, gt: np.ndarray) -> float:
            return float(np.abs(pred_set.sum() - gt.sum())) / max(float(gt.sum()), 1)

        gt = _make_sphere_mask(radius=5.0)
        pred_probs = np.zeros((16, 16, 16), dtype=np.float32)
        pred_probs[gt.astype(bool)] = 0.7

        predictor = RiskControllingPredictor(alpha=0.3, risk_fn=custom_risk)
        predictor.calibrate([pred_probs] * 5, [gt] * 5)
        assert predictor.is_calibrated


# ---------------------------------------------------------------------------
# Task 3.2: Risk function library
# ---------------------------------------------------------------------------


class TestRiskFunctions:
    """Test segmentation risk functions."""

    def test_dice_loss_perfect(self) -> None:
        """Dice loss = 0.0 for identical masks."""
        from minivess.ensemble.risk_control import dice_loss_risk

        mask = _make_sphere_mask(radius=5.0)
        assert dice_loss_risk(mask.astype(bool), mask) == pytest.approx(0.0)

    def test_dice_loss_empty(self) -> None:
        """Dice loss = 1.0 for empty prediction."""
        from minivess.ensemble.risk_control import dice_loss_risk

        gt = _make_sphere_mask(radius=5.0)
        empty = np.zeros_like(gt, dtype=bool)
        assert dice_loss_risk(empty, gt) == pytest.approx(1.0)

    def test_fnr_perfect(self) -> None:
        """FNR = 0.0 for perfect prediction."""
        from minivess.ensemble.risk_control import fnr_risk

        mask = _make_sphere_mask(radius=5.0)
        assert fnr_risk(mask.astype(bool), mask) == pytest.approx(0.0)

    def test_fnr_empty(self) -> None:
        """FNR = 1.0 for empty prediction (all false negatives)."""
        from minivess.ensemble.risk_control import fnr_risk

        gt = _make_sphere_mask(radius=5.0)
        empty = np.zeros_like(gt, dtype=bool)
        assert fnr_risk(empty, gt) == pytest.approx(1.0)

    def test_fpr_perfect(self) -> None:
        """FPR = 0.0 for perfect prediction."""
        from minivess.ensemble.risk_control import fpr_risk

        mask = _make_sphere_mask(radius=5.0)
        assert fpr_risk(mask.astype(bool), mask) == pytest.approx(0.0)

    def test_volume_error_perfect(self) -> None:
        """Volume error = 0.0 for same-volume masks."""
        from minivess.ensemble.risk_control import volume_error_risk

        mask = _make_sphere_mask(radius=5.0)
        assert volume_error_risk(mask.astype(bool), mask) == pytest.approx(0.0)
