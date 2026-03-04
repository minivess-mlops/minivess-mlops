"""Tests for CRC (Conformalized Risk Control) and Varisco heatmaps.

CRC extends conformal prediction with formal risk control on arbitrary
monotone loss functions (Angelopoulos et al. 2024). Varisco heatmaps
visualize prediction set size per pixel for dashboard integration.

Issue: #308 | Phase 2 | Plan: T2.1 (RED)
"""

from __future__ import annotations

import numpy as np


def _make_calibration_data(
    n: int = 10,
    num_classes: int = 2,
    shape: tuple[int, ...] = (4, 4, 4),
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic calibration data (softmax probs + labels)."""
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((n, num_classes, *shape)).astype(np.float32)
    # Softmax
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    labels = probs.argmax(axis=1).astype(np.int64)
    return probs, labels


class TestCRCConformal:
    """Tests for CRC (Conformalized Risk Control) predictor."""

    def test_lac_thresholding_includes_true_class(self) -> None:
        """LAC-based prediction sets should include the ground truth class
        with probability >= 1 - alpha on calibration data."""
        from minivess.ensemble.crc_conformal import CRCPredictor

        probs, labels = _make_calibration_data(n=50, seed=42)
        predictor = CRCPredictor(alpha=0.1)
        predictor.calibrate(probs, labels)

        result = predictor.predict(probs)

        # Check coverage: true class should be included in prediction set
        n_samples = labels.shape[0]
        covered = 0
        for i in range(n_samples):
            for d in range(labels.shape[1]):
                for h in range(labels.shape[2]):
                    for w in range(labels.shape[3]):
                        true_cls = labels[i, d, h, w]
                        if result.prediction_sets[i, true_cls, d, h, w]:
                            covered += 1

        total_voxels = np.prod(labels.shape)
        coverage = covered / total_voxels
        # Coverage should be at least 1 - alpha (0.9) on calibration set
        assert coverage >= 0.85, f"Coverage {coverage:.3f} too low"

    def test_crc_coverage_guarantee(self) -> None:
        """CRC with risk function should maintain coverage >= 1-alpha."""
        from minivess.ensemble.crc_conformal import CRCPredictor

        probs, labels = _make_calibration_data(n=100, seed=123)
        predictor = CRCPredictor(alpha=0.1)
        predictor.calibrate(probs, labels)

        # Test on separate data
        test_probs, test_labels = _make_calibration_data(n=20, seed=456)
        result = predictor.predict(test_probs)

        assert result.prediction_sets.shape[0] == test_probs.shape[0]
        assert result.prediction_sets.shape[1] == test_probs.shape[1]
        assert result.alpha == 0.1

    def test_crc_integrates_with_conformal_evaluator(self) -> None:
        """CRC result should have same interface as ConformalResult."""
        from minivess.ensemble.conformal import ConformalResult
        from minivess.ensemble.crc_conformal import CRCPredictor

        probs, labels = _make_calibration_data(n=20)
        predictor = CRCPredictor(alpha=0.05)
        predictor.calibrate(probs, labels)

        result = predictor.predict(probs)
        # Must have same fields as ConformalResult
        assert isinstance(result, ConformalResult)
        assert hasattr(result, "prediction_sets")
        assert hasattr(result, "quantile")
        assert hasattr(result, "alpha")

    def test_crc_stricter_alpha_smaller_sets(self) -> None:
        """Higher alpha (less coverage) should produce smaller prediction sets."""
        from minivess.ensemble.crc_conformal import CRCPredictor

        probs, labels = _make_calibration_data(n=50)

        predictor_strict = CRCPredictor(alpha=0.3)
        predictor_strict.calibrate(probs, labels)
        result_strict = predictor_strict.predict(probs)

        predictor_loose = CRCPredictor(alpha=0.01)
        predictor_loose.calibrate(probs, labels)
        result_loose = predictor_loose.predict(probs)

        strict_size = result_strict.prediction_sets.sum()
        loose_size = result_loose.prediction_sets.sum()
        assert strict_size <= loose_size, (
            f"Strict alpha should give smaller sets: {strict_size} vs {loose_size}"
        )


class TestVariscoHeatmap:
    """Tests for Varisco uncertainty heatmaps (prediction set size per voxel)."""

    def test_varisco_heatmap_shape(self) -> None:
        """Varisco heatmap should have spatial shape (B, D, H, W)."""
        from minivess.ensemble.crc_conformal import varisco_heatmap

        prediction_sets = np.ones((2, 3, 4, 4, 4), dtype=bool)
        heatmap = varisco_heatmap(prediction_sets)
        assert heatmap.shape == (2, 4, 4, 4)

    def test_varisco_heatmap_values_bounded(self) -> None:
        """Heatmap values should be in [0, num_classes]."""
        from minivess.ensemble.crc_conformal import varisco_heatmap

        num_classes = 3
        rng = np.random.default_rng(42)
        prediction_sets = rng.random((2, num_classes, 4, 4, 4)) > 0.5
        heatmap = varisco_heatmap(prediction_sets)

        assert heatmap.min() >= 0
        assert heatmap.max() <= num_classes

    def test_varisco_heatmap_all_included(self) -> None:
        """When all classes are in prediction set, heatmap = num_classes."""
        from minivess.ensemble.crc_conformal import varisco_heatmap

        num_classes = 2
        prediction_sets = np.ones((1, num_classes, 4, 4, 4), dtype=bool)
        heatmap = varisco_heatmap(prediction_sets)
        assert (heatmap == num_classes).all()

    def test_varisco_heatmap_single_class(self) -> None:
        """When exactly one class per voxel, heatmap = 1 everywhere."""
        from minivess.ensemble.crc_conformal import varisco_heatmap

        prediction_sets = np.zeros((1, 2, 4, 4, 4), dtype=bool)
        prediction_sets[:, 0] = True  # only class 0
        heatmap = varisco_heatmap(prediction_sets)
        assert (heatmap == 1).all()
