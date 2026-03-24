"""Tests for evaluation-time calibration metrics.

Validates Brier score, O:E ratio, IPA, and calibration slope.
"""

from __future__ import annotations

import numpy as np


def _make_perfect_predictions(
    n: int = 10000, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Create perfectly calibrated predictions."""
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n).astype(float)
    # Perfect: p_pred = y_true (with tiny noise to avoid degeneracy)
    p_pred = np.clip(y_true + rng.normal(0, 0.01, size=n), 0.001, 0.999)
    return y_true, p_pred


def _make_random_predictions(
    n: int = 10000, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Create random (poor) predictions."""
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n).astype(float)
    p_pred = rng.uniform(0.001, 0.999, size=n)
    return y_true, p_pred


class TestBrierScorePerfect:
    def test_brier_score_perfect_predictions(self) -> None:
        from minivess.pipeline.calibration_metrics import compute_calibration_metrics

        y_true, p_pred = _make_perfect_predictions()
        result = compute_calibration_metrics(y_true, p_pred, seed=42)
        assert result.brier_score < 0.01


class TestBrierScoreRandom:
    def test_brier_score_random_predictions(self) -> None:
        from minivess.pipeline.calibration_metrics import compute_calibration_metrics

        y_true, p_pred = _make_random_predictions()
        result = compute_calibration_metrics(y_true, p_pred, seed=42)
        assert result.brier_score > 0.1


class TestOERatioCalibrated:
    def test_oe_ratio_calibrated(self) -> None:
        from minivess.pipeline.calibration_metrics import compute_calibration_metrics

        y_true, p_pred = _make_perfect_predictions()
        result = compute_calibration_metrics(y_true, p_pred, seed=42)
        assert 0.9 < result.oe_ratio < 1.1, f"O:E should be ~1.0, got {result.oe_ratio}"


class TestOERatioOverconfident:
    def test_oe_ratio_overconfident(self) -> None:
        from minivess.pipeline.calibration_metrics import compute_calibration_metrics

        rng = np.random.default_rng(42)
        n = 10000
        y_true = rng.integers(0, 2, size=n).astype(float)
        # Overconfident: predict high probabilities regardless of label
        p_pred = np.full(n, 0.9)
        result = compute_calibration_metrics(y_true, p_pred, seed=42)
        # O:E < 1 means model predicts too many positives
        assert result.oe_ratio < 0.7


class TestIPAPositive:
    def test_ipa_positive_for_good_model(self) -> None:
        from minivess.pipeline.calibration_metrics import compute_calibration_metrics

        y_true, p_pred = _make_perfect_predictions()
        result = compute_calibration_metrics(y_true, p_pred, seed=42)
        assert result.ipa > 0.5, (
            f"IPA should be positive for good model, got {result.ipa}"
        )


class TestIPAZero:
    def test_ipa_zero_for_null_model(self) -> None:
        from minivess.pipeline.calibration_metrics import compute_calibration_metrics

        rng = np.random.default_rng(42)
        n = 10000
        y_true = rng.integers(0, 2, size=n).astype(float)
        prevalence = y_true.mean()
        p_pred = np.full(n, prevalence)  # Null model: predict prevalence
        result = compute_calibration_metrics(y_true, p_pred, seed=42)
        assert abs(result.ipa) < 0.05, (
            f"IPA should be ~0 for null model, got {result.ipa}"
        )


class TestCalibrationSlopePerfect:
    def test_calibration_slope_perfect(self) -> None:
        from minivess.pipeline.calibration_metrics import compute_calibration_metrics

        y_true, p_pred = _make_perfect_predictions()
        result = compute_calibration_metrics(y_true, p_pred, seed=42)
        # Near-perfect predictions: slope should be positive
        # (Exact slope depends on noise magnitude in near-perfect regime)
        assert result.calibration_slope > 0.5


class TestCalibrationMetricsSubsampling:
    def test_calibration_metrics_voxel_subsampling(self) -> None:
        from minivess.pipeline.calibration_metrics import compute_calibration_metrics

        rng = np.random.default_rng(42)
        n = 500000  # Large to test subsampling
        y_true = rng.integers(0, 2, size=n).astype(float)
        p_pred = rng.uniform(0.001, 0.999, size=n)

        result = compute_calibration_metrics(y_true, p_pred, max_voxels=100000, seed=42)
        assert 0.0 <= result.brier_score <= 1.0
