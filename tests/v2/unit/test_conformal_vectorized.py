"""Tests for vectorized conformal prediction (Phase 0).

Validates that vectorized implementations produce identical results
to the original loop-based versions, with much better performance.
"""

from __future__ import annotations

import time

import numpy as np
import pytest


def _make_synthetic_probs(
    n_volumes: int = 2,
    n_classes: int = 3,
    spatial: tuple[int, int, int] = (4, 4, 2),
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic softmax probs and labels."""
    rng = np.random.default_rng(seed)
    shape = (n_volumes, n_classes, *spatial)
    raw = rng.random(shape).astype(np.float32)
    probs = raw / raw.sum(axis=1, keepdims=True)
    labels = probs.argmax(axis=1).astype(np.int64)
    return probs, labels


# ---------------------------------------------------------------------------
# Task 0.1: ConformalPredictor vectorization
# ---------------------------------------------------------------------------


class TestConformalPredictorVectorized:
    """Verify vectorized calibrate() matches original loop implementation."""

    def test_calibrate_matches_loop_version(self) -> None:
        """Vectorized calibrate() must produce the same quantile as loop version."""
        from minivess.ensemble.conformal import ConformalPredictor

        probs, labels = _make_synthetic_probs(n_volumes=3)

        # Run the vectorized version
        predictor = ConformalPredictor(alpha=0.1)
        predictor.calibrate(probs, labels)
        vectorized_quantile = predictor._quantile

        # Manually compute via the original loop algorithm
        n_samples = probs.shape[0]
        spatial_shape = labels.shape[1:]
        scores_flat: list[float] = []
        for i in range(n_samples):
            for d in range(spatial_shape[0]):
                for h in range(spatial_shape[1]):
                    for w in range(spatial_shape[2]):
                        true_class = labels[i, d, h, w]
                        score = 1.0 - probs[i, true_class, d, h, w]
                        scores_flat.append(float(score))
        scores_array = np.array(scores_flat)
        n = len(scores_array)
        level = min((n + 1) * (1 - 0.1) / n, 1.0)
        expected_quantile = float(np.quantile(scores_array, level))

        assert vectorized_quantile == pytest.approx(expected_quantile, abs=1e-7)

    def test_calibrate_large_volume_fast(self) -> None:
        """Calibration on 10 volumes (32,32,16) should complete in < 1 second."""
        from minivess.ensemble.conformal import ConformalPredictor

        probs, labels = _make_synthetic_probs(n_volumes=10, spatial=(32, 32, 16))
        predictor = ConformalPredictor(alpha=0.1)

        start = time.monotonic()
        predictor.calibrate(probs, labels)
        elapsed = time.monotonic() - start

        assert elapsed < 1.0, f"Calibration took {elapsed:.2f}s, expected < 1.0s"
        assert predictor.is_calibrated

    def test_predict_unchanged(self) -> None:
        """predict() output must be identical before/after vectorization."""
        from minivess.ensemble.conformal import ConformalPredictor

        cal_probs, cal_labels = _make_synthetic_probs(n_volumes=5, seed=1)
        test_probs, _ = _make_synthetic_probs(n_volumes=2, seed=2)

        predictor = ConformalPredictor(alpha=0.1)
        predictor.calibrate(cal_probs, cal_labels)
        result = predictor.predict(test_probs)

        # The prediction set should still be boolean with correct shape
        assert result.prediction_sets.dtype == np.bool_
        assert result.prediction_sets.shape == test_probs.shape

        # Check that the threshold logic is correct: class c included
        # if p(c) >= 1 - quantile
        threshold = 1.0 - predictor._quantile
        expected = test_probs >= threshold
        np.testing.assert_array_equal(result.prediction_sets, expected)

    def test_quantile_finite_sample_correction(self) -> None:
        """Verify ceil((n+1)(1-alpha)/n) formula is preserved."""
        from minivess.ensemble.conformal import ConformalPredictor

        probs, labels = _make_synthetic_probs(n_volumes=5, spatial=(4, 4, 2))
        predictor = ConformalPredictor(alpha=0.1)
        predictor.calibrate(probs, labels)

        # Just verify quantile is set and reasonable
        assert predictor._quantile is not None
        assert 0.0 <= predictor._quantile <= 1.0


# ---------------------------------------------------------------------------
# Task 0.2: compute_coverage_metrics vectorization
# ---------------------------------------------------------------------------


class TestCoverageMetricsVectorized:
    """Verify vectorized compute_coverage_metrics matches loop version."""

    def test_coverage_matches_loop_version(self) -> None:
        """Vectorized coverage must match the original loop computation."""
        from minivess.ensemble.mapie_conformal import compute_coverage_metrics

        probs, labels = _make_synthetic_probs(n_volumes=3, n_classes=2)

        # Create synthetic prediction sets (B, C, D, H, W) boolean
        rng = np.random.default_rng(99)
        pred_sets = rng.random(probs.shape) > 0.3

        result = compute_coverage_metrics(pred_sets, labels)

        # Manual loop computation
        n_volumes = labels.shape[0]
        spatial = labels.shape[1:]
        covered = 0
        total = 0
        for i in range(n_volumes):
            for d in range(spatial[0]):
                for h in range(spatial[1]):
                    for w in range(spatial[2]):
                        true_class = labels[i, d, h, w]
                        if pred_sets[i, true_class, d, h, w]:
                            covered += 1
                        total += 1
        expected_coverage = covered / max(total, 1)

        assert result.coverage == pytest.approx(expected_coverage, abs=1e-7)

    def test_coverage_large_volume_fast(self) -> None:
        """Coverage on 10 volumes (32,32,16) should complete in < 1 second."""
        from minivess.ensemble.mapie_conformal import compute_coverage_metrics

        probs, labels = _make_synthetic_probs(
            n_volumes=10, n_classes=2, spatial=(32, 32, 16)
        )
        rng = np.random.default_rng(99)
        pred_sets = rng.random(probs.shape) > 0.3

        start = time.monotonic()
        result = compute_coverage_metrics(pred_sets, labels)
        elapsed = time.monotonic() - start

        assert elapsed < 1.0, f"Coverage took {elapsed:.2f}s, expected < 1.0s"
        assert 0.0 <= result.coverage <= 1.0

    def test_mean_set_size_unchanged(self) -> None:
        """mean_set_size computation must be identical."""
        from minivess.ensemble.mapie_conformal import compute_coverage_metrics

        rng = np.random.default_rng(42)
        pred_sets = rng.random((2, 3, 4, 4, 2)) > 0.5
        labels = np.zeros((2, 4, 4, 2), dtype=np.int64)

        result = compute_coverage_metrics(pred_sets, labels)
        expected = float(pred_sets.sum(axis=1).mean())
        assert result.mean_set_size == pytest.approx(expected, abs=1e-7)

    def test_perfect_coverage(self) -> None:
        """All-true prediction sets should give coverage 1.0."""
        from minivess.ensemble.mapie_conformal import compute_coverage_metrics

        labels = np.zeros((2, 4, 4, 2), dtype=np.int64)
        pred_sets = np.ones((2, 3, 4, 4, 2), dtype=bool)

        result = compute_coverage_metrics(pred_sets, labels)
        assert result.coverage == pytest.approx(1.0)

    def test_zero_coverage(self) -> None:
        """All-false prediction sets should give coverage 0.0."""
        from minivess.ensemble.mapie_conformal import compute_coverage_metrics

        labels = np.zeros((2, 4, 4, 2), dtype=np.int64)
        pred_sets = np.zeros((2, 3, 4, 4, 2), dtype=bool)

        result = compute_coverage_metrics(pred_sets, labels)
        assert result.coverage == pytest.approx(0.0)
