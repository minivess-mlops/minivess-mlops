"""Tests for MAPIE conformal prediction integration (Issue #7)."""

from __future__ import annotations

import numpy as np
import pytest


def _make_synthetic_probs(
    n_volumes: int = 2,
    n_classes: int = 3,
    spatial: tuple[int, int, int] = (4, 4, 2),
) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic softmax probs and labels for testing.

    Returns (probs, labels) where probs sums to 1 along class axis.
    """
    rng = np.random.default_rng(42)
    shape = (n_volumes, n_classes, *spatial)
    raw = rng.random(shape).astype(np.float32)
    # Normalize to softmax-like
    probs = raw / raw.sum(axis=1, keepdims=True)
    labels = probs.argmax(axis=1).astype(np.int64)
    return probs, labels


# ---------------------------------------------------------------------------
# T1: MapieConformalSegmentation creation
# ---------------------------------------------------------------------------


class TestMapieConformalSegmentation:
    """Test MAPIE-based conformal predictor for 3D segmentation."""

    def test_creates_with_default_alpha(self) -> None:
        """Should create with default 90% coverage (alpha=0.1)."""
        from minivess.ensemble.mapie_conformal import MapieConformalSegmentation

        predictor = MapieConformalSegmentation()
        assert predictor.alpha == 0.1

    def test_creates_with_custom_alpha(self) -> None:
        """Should accept custom alpha values."""
        from minivess.ensemble.mapie_conformal import MapieConformalSegmentation

        predictor = MapieConformalSegmentation(alpha=0.05)
        assert predictor.alpha == 0.05

    def test_not_calibrated_initially(self) -> None:
        """Should not be calibrated before calling calibrate()."""
        from minivess.ensemble.mapie_conformal import MapieConformalSegmentation

        predictor = MapieConformalSegmentation()
        assert predictor.is_calibrated is False


# ---------------------------------------------------------------------------
# T2: Calibration
# ---------------------------------------------------------------------------


class TestCalibration:
    """Test MAPIE calibration on holdout data."""

    def test_calibrate_marks_as_calibrated(self) -> None:
        """After calibration, is_calibrated should be True."""
        from minivess.ensemble.mapie_conformal import MapieConformalSegmentation

        predictor = MapieConformalSegmentation(alpha=0.1)
        probs, labels = _make_synthetic_probs(n_volumes=3)
        predictor.calibrate(probs, labels)
        assert predictor.is_calibrated is True

    def test_predict_before_calibrate_raises(self) -> None:
        """Predicting without calibration should raise RuntimeError."""
        from minivess.ensemble.mapie_conformal import MapieConformalSegmentation

        predictor = MapieConformalSegmentation()
        probs, _ = _make_synthetic_probs(n_volumes=1)
        with pytest.raises(RuntimeError, match="calibrate"):
            predictor.predict(probs)

    def test_calibrate_with_different_volumes(self) -> None:
        """Calibration should work with varying number of volumes."""
        from minivess.ensemble.mapie_conformal import MapieConformalSegmentation

        predictor = MapieConformalSegmentation()
        probs, labels = _make_synthetic_probs(n_volumes=5, spatial=(8, 8, 4))
        predictor.calibrate(probs, labels)
        assert predictor.is_calibrated is True


# ---------------------------------------------------------------------------
# T3: Prediction sets
# ---------------------------------------------------------------------------


class TestPredictionSets:
    """Test conformal prediction set generation."""

    def test_prediction_sets_shape(self) -> None:
        """Prediction sets should match input spatial dims."""
        from minivess.ensemble.mapie_conformal import MapieConformalSegmentation

        predictor = MapieConformalSegmentation(alpha=0.1)
        cal_probs, cal_labels = _make_synthetic_probs(n_volumes=3)
        predictor.calibrate(cal_probs, cal_labels)

        test_probs, _ = _make_synthetic_probs(n_volumes=2)
        result = predictor.predict(test_probs)
        # (B, C, D, H, W) boolean
        assert result.prediction_sets.shape == test_probs.shape

    def test_prediction_sets_are_boolean(self) -> None:
        """Prediction sets should be boolean arrays."""
        from minivess.ensemble.mapie_conformal import MapieConformalSegmentation

        predictor = MapieConformalSegmentation(alpha=0.1)
        cal_probs, cal_labels = _make_synthetic_probs(n_volumes=3)
        predictor.calibrate(cal_probs, cal_labels)

        test_probs, _ = _make_synthetic_probs(n_volumes=2)
        result = predictor.predict(test_probs)
        assert result.prediction_sets.dtype == np.bool_

    def test_prediction_sets_mostly_nonempty(self) -> None:
        """Most voxels should have at least one class in their prediction set."""
        from minivess.ensemble.mapie_conformal import MapieConformalSegmentation

        predictor = MapieConformalSegmentation(alpha=0.1)
        cal_probs, cal_labels = _make_synthetic_probs(n_volumes=10, spatial=(8, 8, 4))
        predictor.calibrate(cal_probs, cal_labels)

        test_probs, _ = _make_synthetic_probs(n_volumes=2, spatial=(8, 8, 4))
        result = predictor.predict(test_probs)
        # Most voxels should have at least one class
        nonempty = (result.prediction_sets.sum(axis=1) >= 1).mean()
        assert nonempty >= 0.8

    def test_higher_alpha_gives_smaller_sets(self) -> None:
        """Higher alpha (lower coverage) should produce smaller prediction sets."""
        from minivess.ensemble.mapie_conformal import MapieConformalSegmentation

        cal_probs, cal_labels = _make_synthetic_probs(n_volumes=10, spatial=(8, 8, 4))
        test_probs, _ = _make_synthetic_probs(n_volumes=2, spatial=(8, 8, 4))

        predictor_90 = MapieConformalSegmentation(alpha=0.1)
        predictor_90.calibrate(cal_probs, cal_labels)
        result_90 = predictor_90.predict(test_probs)

        predictor_50 = MapieConformalSegmentation(alpha=0.5)
        predictor_50.calibrate(cal_probs, cal_labels)
        result_50 = predictor_50.predict(test_probs)

        # 90% coverage sets should be >= 50% coverage sets on average
        width_90 = result_90.prediction_sets.sum(axis=1).mean()
        width_50 = result_50.prediction_sets.sum(axis=1).mean()
        assert width_90 >= width_50


# ---------------------------------------------------------------------------
# T4: Coverage metrics
# ---------------------------------------------------------------------------


class TestCoverageMetrics:
    """Test conformal coverage metric computation."""

    def test_compute_coverage_returns_metrics(self) -> None:
        """compute_coverage_metrics should return a ConformalMetrics."""
        from minivess.ensemble.mapie_conformal import (
            MapieConformalSegmentation,
            compute_coverage_metrics,
        )

        predictor = MapieConformalSegmentation(alpha=0.1)
        cal_probs, cal_labels = _make_synthetic_probs(n_volumes=5)
        predictor.calibrate(cal_probs, cal_labels)

        test_probs, test_labels = _make_synthetic_probs(n_volumes=2)
        result = predictor.predict(test_probs)
        metrics = compute_coverage_metrics(result.prediction_sets, test_labels)
        assert 0.0 <= metrics.coverage <= 1.0
        assert metrics.mean_set_size >= 0.0

    def test_coverage_near_target(self) -> None:
        """Empirical coverage should be close to 1 - alpha."""
        from minivess.ensemble.mapie_conformal import (
            MapieConformalSegmentation,
            compute_coverage_metrics,
        )

        # Use enough data for stable coverage
        cal_probs, cal_labels = _make_synthetic_probs(
            n_volumes=20, n_classes=2, spatial=(8, 8, 4)
        )
        test_probs, test_labels = _make_synthetic_probs(
            n_volumes=10, n_classes=2, spatial=(8, 8, 4)
        )

        predictor = MapieConformalSegmentation(alpha=0.1)
        predictor.calibrate(cal_probs, cal_labels)
        result = predictor.predict(test_probs)
        metrics = compute_coverage_metrics(result.prediction_sets, test_labels)
        # Coverage should be roughly >= 0.85 (allowing some slack)
        assert metrics.coverage >= 0.85

    def test_metrics_as_dict(self) -> None:
        """ConformalMetrics should be convertible to dict for MLflow logging."""
        from minivess.ensemble.mapie_conformal import (
            ConformalMetrics,
        )

        metrics = ConformalMetrics(coverage=0.92, mean_set_size=1.3)
        d = metrics.to_dict()
        assert d["conformal_coverage"] == 0.92
        assert d["conformal_mean_set_size"] == 1.3
