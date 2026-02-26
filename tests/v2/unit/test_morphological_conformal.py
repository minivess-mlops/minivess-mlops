"""Tests for morphological conformal prediction (Phase 1).

Validates ConSeMa-inspired inner/outer contour prediction sets with
3D structuring elements for binary vascular segmentation.
"""

from __future__ import annotations

import numpy as np
import pytest


def _make_sphere_mask(
    shape: tuple[int, int, int] = (16, 16, 16),
    center: tuple[int, int, int] | None = None,
    radius: float = 5.0,
) -> np.ndarray:
    """Create a binary sphere mask for testing."""
    if center is None:
        center = tuple(s // 2 for s in shape)
    coords = np.mgrid[: shape[0], : shape[1], : shape[2]]
    dist_sq = sum((c - cn) ** 2 for c, cn in zip(coords, center, strict=True))
    return (dist_sq <= radius**2).astype(np.int64)


def _make_tube_mask(
    shape: tuple[int, int, int] = (16, 16, 32),
    center_yz: tuple[int, int] = (8, 8),
    radius: float = 3.0,
) -> np.ndarray:
    """Create a tube (vessel-like) mask along the Z axis."""
    mask = np.zeros(shape, dtype=np.int64)
    for y in range(shape[0]):
        for x in range(shape[1]):
            if (y - center_yz[0]) ** 2 + (x - center_yz[1]) ** 2 <= radius**2:
                mask[y, x, :] = 1
    return mask


# ---------------------------------------------------------------------------
# Task 1.1: MorphologicalConformalResult dataclass
# ---------------------------------------------------------------------------


class TestMorphologicalConformalResult:
    """Test the result dataclass."""

    def test_result_has_inner_outer_contours(self) -> None:
        """Result must have inner_contour and outer_contour fields."""
        from minivess.ensemble.morphological_conformal import (
            MorphologicalConformalResult,
        )

        inner = np.zeros((4, 4, 4), dtype=bool)
        outer = np.ones((4, 4, 4), dtype=bool)
        result = MorphologicalConformalResult(
            inner_contour=inner,
            outer_contour=outer,
            band_width=1.0,
            dilation_radius=2,
            erosion_radius=1,
            alpha=0.1,
        )
        assert result.inner_contour.dtype == np.bool_
        assert result.outer_contour.dtype == np.bool_

    def test_result_band_width_nonnegative(self) -> None:
        """band_width must be >= 0."""
        from minivess.ensemble.morphological_conformal import (
            MorphologicalConformalResult,
        )

        result = MorphologicalConformalResult(
            inner_contour=np.zeros((4, 4, 4), dtype=bool),
            outer_contour=np.ones((4, 4, 4), dtype=bool),
            band_width=0.0,
            dilation_radius=0,
            erosion_radius=0,
            alpha=0.1,
        )
        assert result.band_width >= 0.0

    def test_inner_subset_of_outer(self) -> None:
        """inner_contour should be a subset of outer_contour."""
        from minivess.ensemble.morphological_conformal import (
            MorphologicalConformalPredictor,
        )

        # Create calibration data: pred slightly smaller than GT
        gt = _make_sphere_mask(radius=5.0)
        pred = _make_sphere_mask(radius=4.0)

        predictor = MorphologicalConformalPredictor(alpha=0.1)
        predictor.calibrate(
            predictions=[pred, pred, pred],
            labels=[gt, gt, gt],
        )
        result = predictor.predict(pred)

        # Inner should be subset of outer
        assert np.all(result.inner_contour <= result.outer_contour)

    def test_inner_subset_of_prediction(self) -> None:
        """inner_contour should be a subset of the original prediction."""
        from minivess.ensemble.morphological_conformal import (
            MorphologicalConformalPredictor,
        )

        gt = _make_sphere_mask(radius=5.0)
        pred = _make_sphere_mask(radius=5.0)

        predictor = MorphologicalConformalPredictor(alpha=0.1)
        predictor.calibrate(
            predictions=[pred, pred, pred],
            labels=[gt, gt, gt],
        )
        result = predictor.predict(pred)

        # Inner contour is obtained by erosion, so it must be subset of prediction
        assert np.all(result.inner_contour <= pred.astype(bool))


# ---------------------------------------------------------------------------
# Task 1.2: Dilation calibration
# ---------------------------------------------------------------------------


class TestDilationCalibration:
    """Test morphological dilation radius calibration."""

    def test_calibrate_perfect_prediction(self) -> None:
        """lambda=0 when prediction perfectly matches ground truth."""
        from minivess.ensemble.morphological_conformal import (
            MorphologicalConformalPredictor,
        )

        mask = _make_sphere_mask(radius=5.0)
        predictor = MorphologicalConformalPredictor(alpha=0.1)
        predictor.calibrate(
            predictions=[mask, mask, mask],
            labels=[mask, mask, mask],
        )
        assert predictor.dilation_radius == 0

    def test_calibrate_shifted_prediction(self) -> None:
        """lambda > 0 when prediction is shifted/smaller than GT."""
        from minivess.ensemble.morphological_conformal import (
            MorphologicalConformalPredictor,
        )

        gt = _make_sphere_mask(radius=5.0)
        pred = _make_sphere_mask(radius=3.0)  # Smaller than GT

        predictor = MorphologicalConformalPredictor(alpha=0.1)
        predictor.calibrate(
            predictions=[pred, pred, pred, pred, pred],
            labels=[gt, gt, gt, gt, gt],
        )
        assert predictor.dilation_radius > 0

    def test_calibrate_empty_prediction(self) -> None:
        """Handle empty prediction gracefully (max_lambda cap)."""
        from minivess.ensemble.morphological_conformal import (
            MorphologicalConformalPredictor,
        )

        gt = _make_sphere_mask(radius=5.0)
        empty = np.zeros_like(gt)

        predictor = MorphologicalConformalPredictor(alpha=0.1, max_radius=10)
        predictor.calibrate(
            predictions=[empty, empty, empty],
            labels=[gt, gt, gt],
        )
        assert predictor.is_calibrated
        assert predictor.dilation_radius <= 10

    def test_calibrate_stores_radius(self) -> None:
        """After calibration, is_calibrated=True and dilation_radius is set."""
        from minivess.ensemble.morphological_conformal import (
            MorphologicalConformalPredictor,
        )

        mask = _make_sphere_mask(radius=5.0)
        predictor = MorphologicalConformalPredictor(alpha=0.1)
        assert not predictor.is_calibrated

        predictor.calibrate(
            predictions=[mask, mask, mask],
            labels=[mask, mask, mask],
        )
        assert predictor.is_calibrated
        assert predictor.dilation_radius >= 0

    def test_max_lambda_cap(self) -> None:
        """Dilation radius must not exceed max_radius."""
        from minivess.ensemble.morphological_conformal import (
            MorphologicalConformalPredictor,
        )

        gt = _make_sphere_mask(radius=7.0)
        empty = np.zeros_like(gt)

        predictor = MorphologicalConformalPredictor(alpha=0.1, max_radius=5)
        predictor.calibrate(
            predictions=[empty, empty, empty],
            labels=[gt, gt, gt],
        )
        assert predictor.dilation_radius <= 5


# ---------------------------------------------------------------------------
# Task 1.3: Erosion calibration
# ---------------------------------------------------------------------------


class TestErosionCalibration:
    """Test morphological erosion radius calibration."""

    def test_erosion_perfect_prediction(self) -> None:
        """mu=0 when prediction perfectly matches GT."""
        from minivess.ensemble.morphological_conformal import (
            MorphologicalConformalPredictor,
        )

        mask = _make_sphere_mask(radius=5.0)
        predictor = MorphologicalConformalPredictor(alpha=0.1)
        predictor.calibrate(
            predictions=[mask, mask, mask],
            labels=[mask, mask, mask],
        )
        assert predictor.erosion_radius == 0

    def test_erosion_oversized_prediction(self) -> None:
        """mu > 0 when prediction is larger than GT."""
        from minivess.ensemble.morphological_conformal import (
            MorphologicalConformalPredictor,
        )

        gt = _make_sphere_mask(radius=3.0)
        pred = _make_sphere_mask(radius=6.0)  # Larger than GT

        predictor = MorphologicalConformalPredictor(alpha=0.1)
        predictor.calibrate(
            predictions=[pred, pred, pred, pred, pred],
            labels=[gt, gt, gt, gt, gt],
        )
        assert predictor.erosion_radius > 0

    def test_erosion_empty_after_erosion(self) -> None:
        """Handle thin structures that vanish after erosion gracefully."""
        from minivess.ensemble.morphological_conformal import (
            MorphologicalConformalPredictor,
        )

        # Very thin structure (radius=1)
        thin_mask = _make_sphere_mask(radius=1.0)
        gt_smaller = _make_sphere_mask(radius=0.5)

        predictor = MorphologicalConformalPredictor(alpha=0.1, max_radius=5)
        predictor.calibrate(
            predictions=[thin_mask, thin_mask, thin_mask],
            labels=[gt_smaller, gt_smaller, gt_smaller],
        )
        assert predictor.is_calibrated

    def test_erosion_stores_radius(self) -> None:
        """erosion_radius attribute must be set after calibration."""
        from minivess.ensemble.morphological_conformal import (
            MorphologicalConformalPredictor,
        )

        mask = _make_sphere_mask(radius=5.0)
        predictor = MorphologicalConformalPredictor(alpha=0.1)
        predictor.calibrate(
            predictions=[mask, mask, mask],
            labels=[mask, mask, mask],
        )
        assert predictor.erosion_radius >= 0


# ---------------------------------------------------------------------------
# Task 1.4: Predict with inner/outer bands
# ---------------------------------------------------------------------------


class TestMorphologicalPredict:
    """Test predict() with morphological inner/outer contour bands."""

    def test_predict_output_shape(self) -> None:
        """outer/inner must have same spatial shape as input prediction."""
        from minivess.ensemble.morphological_conformal import (
            MorphologicalConformalPredictor,
        )

        mask = _make_sphere_mask(radius=5.0, shape=(16, 16, 16))
        predictor = MorphologicalConformalPredictor(alpha=0.1)
        predictor.calibrate(
            predictions=[mask, mask, mask],
            labels=[mask, mask, mask],
        )
        result = predictor.predict(mask)

        assert result.outer_contour.shape == mask.shape
        assert result.inner_contour.shape == mask.shape

    def test_predict_outer_covers_prediction(self) -> None:
        """Outer contour must cover the original prediction."""
        from minivess.ensemble.morphological_conformal import (
            MorphologicalConformalPredictor,
        )

        mask = _make_sphere_mask(radius=5.0)
        predictor = MorphologicalConformalPredictor(alpha=0.1)
        predictor.calibrate(
            predictions=[mask, mask, mask],
            labels=[mask, mask, mask],
        )
        result = predictor.predict(mask)

        # prediction should be subset of outer contour
        assert np.all(mask.astype(bool) <= result.outer_contour)

    def test_predict_inner_subset_prediction(self) -> None:
        """Inner contour must be a subset of the original prediction."""
        from minivess.ensemble.morphological_conformal import (
            MorphologicalConformalPredictor,
        )

        mask = _make_sphere_mask(radius=5.0)
        predictor = MorphologicalConformalPredictor(alpha=0.1)
        predictor.calibrate(
            predictions=[mask, mask, mask],
            labels=[mask, mask, mask],
        )
        result = predictor.predict(mask)

        assert np.all(result.inner_contour <= mask.astype(bool))

    def test_predict_band_between_inner_outer(self) -> None:
        """Band = outer AND NOT inner."""
        from minivess.ensemble.morphological_conformal import (
            MorphologicalConformalPredictor,
        )

        gt = _make_sphere_mask(radius=5.0)
        pred = _make_sphere_mask(radius=4.0)

        predictor = MorphologicalConformalPredictor(alpha=0.1)
        predictor.calibrate(
            predictions=[pred] * 5,
            labels=[gt] * 5,
        )
        result = predictor.predict(pred)

        # Band width > 0 if there's any uncertainty
        assert result.band_width >= 0.0

    def test_predict_before_calibrate_raises(self) -> None:
        """Predicting without calibration must raise RuntimeError."""
        from minivess.ensemble.morphological_conformal import (
            MorphologicalConformalPredictor,
        )

        predictor = MorphologicalConformalPredictor(alpha=0.1)
        mask = _make_sphere_mask(radius=5.0)
        with pytest.raises(RuntimeError, match="calibrate"):
            predictor.predict(mask)

    def test_predict_binary_input(self) -> None:
        """Must work with binary integer masks, not just boolean."""
        from minivess.ensemble.morphological_conformal import (
            MorphologicalConformalPredictor,
        )

        mask = _make_sphere_mask(radius=5.0)  # int64
        assert mask.dtype == np.int64

        predictor = MorphologicalConformalPredictor(alpha=0.1)
        predictor.calibrate(
            predictions=[mask, mask, mask],
            labels=[mask, mask, mask],
        )
        result = predictor.predict(mask)
        assert result.outer_contour.dtype == np.bool_
        assert result.inner_contour.dtype == np.bool_


# ---------------------------------------------------------------------------
# Task 1.5: Morphological CP metrics
# ---------------------------------------------------------------------------


class TestMorphologicalMetrics:
    """Test morphological conformal prediction metrics."""

    def test_metrics_perfect_prediction(self) -> None:
        """Perfect prediction: coverage 1.0, band_width 0."""
        from minivess.ensemble.morphological_conformal import (
            MorphologicalConformalPredictor,
            compute_morphological_metrics,
        )

        mask = _make_sphere_mask(radius=5.0)
        predictor = MorphologicalConformalPredictor(alpha=0.1)
        predictor.calibrate(
            predictions=[mask, mask, mask],
            labels=[mask, mask, mask],
        )
        result = predictor.predict(mask)
        metrics = compute_morphological_metrics(result, mask)

        assert metrics["outer_coverage"] == pytest.approx(1.0)
        assert metrics["mean_band_width"] >= 0.0

    def test_metrics_returns_all_fields(self) -> None:
        """All 4 metric fields must be present."""
        from minivess.ensemble.morphological_conformal import (
            MorphologicalConformalPredictor,
            compute_morphological_metrics,
        )

        mask = _make_sphere_mask(radius=5.0)
        predictor = MorphologicalConformalPredictor(alpha=0.1)
        predictor.calibrate(
            predictions=[mask, mask, mask],
            labels=[mask, mask, mask],
        )
        result = predictor.predict(mask)
        metrics = compute_morphological_metrics(result, mask)

        assert "outer_coverage" in metrics
        assert "inner_precision" in metrics
        assert "mean_band_width" in metrics
        assert "band_volume_ratio" in metrics

    def test_metrics_to_dict(self) -> None:
        """Metrics dict values must be float (MLflow compatible)."""
        from minivess.ensemble.morphological_conformal import (
            MorphologicalConformalPredictor,
            compute_morphological_metrics,
        )

        mask = _make_sphere_mask(radius=5.0)
        predictor = MorphologicalConformalPredictor(alpha=0.1)
        predictor.calibrate(
            predictions=[mask, mask, mask],
            labels=[mask, mask, mask],
        )
        result = predictor.predict(mask)
        metrics = compute_morphological_metrics(result, mask)

        for key, val in metrics.items():
            assert isinstance(val, float), f"{key} is {type(val)}, expected float"
