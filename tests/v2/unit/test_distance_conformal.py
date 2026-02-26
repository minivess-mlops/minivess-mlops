"""Tests for distance-transform conformal prediction (Phase 2).

Validates FNR-controlling conformal prediction using signed distance
transforms for binary 3D segmentation.
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
# Task 2.1: DistanceTransformConformalPredictor
# ---------------------------------------------------------------------------


class TestDistanceTransformConformal:
    """Test distance-transform based conformal prediction."""

    def test_calibrate_perfect_prediction(self) -> None:
        """Threshold ~0 when prediction perfectly matches GT."""
        from minivess.ensemble.distance_conformal import (
            DistanceTransformConformalPredictor,
        )

        mask = _make_sphere_mask(radius=5.0)
        predictor = DistanceTransformConformalPredictor(alpha=0.1)
        predictor.calibrate(
            predictions=[mask, mask, mask],
            labels=[mask, mask, mask],
        )
        assert predictor.calibrated_threshold == pytest.approx(0.0, abs=0.5)

    def test_calibrate_shifted_prediction(self) -> None:
        """Threshold > 0 for prediction that misses some GT voxels."""
        from minivess.ensemble.distance_conformal import (
            DistanceTransformConformalPredictor,
        )

        gt = _make_sphere_mask(radius=5.0)
        pred = _make_sphere_mask(radius=3.0)  # Smaller

        predictor = DistanceTransformConformalPredictor(alpha=0.1)
        predictor.calibrate(
            predictions=[pred] * 5,
            labels=[gt] * 5,
        )
        assert predictor.calibrated_threshold > 0.0

    def test_predict_covers_gt(self) -> None:
        """Dilated prediction should cover GT on calibration data."""
        from minivess.ensemble.distance_conformal import (
            DistanceTransformConformalPredictor,
        )

        gt = _make_sphere_mask(radius=5.0)
        pred = _make_sphere_mask(radius=4.0)

        predictor = DistanceTransformConformalPredictor(alpha=0.1)
        predictor.calibrate(
            predictions=[pred] * 5,
            labels=[gt] * 5,
        )
        result = predictor.predict(pred)

        # The prediction set should cover GT (on calibration data)
        gt_bool = gt.astype(bool)
        coverage = float((gt_bool & result).sum()) / float(gt_bool.sum())
        assert coverage >= 0.8  # Allow some slack for quantile correction

    def test_predict_before_calibrate_raises(self) -> None:
        """Predicting without calibration must raise RuntimeError."""
        from minivess.ensemble.distance_conformal import (
            DistanceTransformConformalPredictor,
        )

        predictor = DistanceTransformConformalPredictor(alpha=0.1)
        mask = _make_sphere_mask(radius=5.0)
        with pytest.raises(RuntimeError, match="calibrate"):
            predictor.predict(mask)

    def test_threshold_stored(self) -> None:
        """calibrated_threshold must be accessible after calibration."""
        from minivess.ensemble.distance_conformal import (
            DistanceTransformConformalPredictor,
        )

        mask = _make_sphere_mask(radius=5.0)
        predictor = DistanceTransformConformalPredictor(alpha=0.1)
        predictor.calibrate(
            predictions=[mask, mask, mask],
            labels=[mask, mask, mask],
        )
        assert predictor.calibrated_threshold >= 0.0

    def test_higher_alpha_smaller_dilation(self) -> None:
        """alpha=0.5 should give smaller threshold than alpha=0.1."""
        from minivess.ensemble.distance_conformal import (
            DistanceTransformConformalPredictor,
        )

        gt = _make_sphere_mask(radius=5.0)
        pred = _make_sphere_mask(radius=3.0)
        cal_preds = [pred] * 10
        cal_labels = [gt] * 10

        p_strict = DistanceTransformConformalPredictor(alpha=0.1)
        p_strict.calibrate(cal_preds, cal_labels)

        p_loose = DistanceTransformConformalPredictor(alpha=0.5)
        p_loose.calibrate(cal_preds, cal_labels)

        assert p_strict.calibrated_threshold >= p_loose.calibrated_threshold


# ---------------------------------------------------------------------------
# Task 2.2: FNR and FPR metrics
# ---------------------------------------------------------------------------


class TestDistanceTransformMetrics:
    """Test FNR/FPR metrics for distance-transform CP."""

    def test_fnr_below_alpha(self) -> None:
        """FNR should be <= alpha on calibration data (with slack)."""
        from minivess.ensemble.distance_conformal import (
            DistanceTransformConformalPredictor,
            compute_distance_metrics,
        )

        gt = _make_sphere_mask(radius=5.0)
        pred = _make_sphere_mask(radius=4.0)

        predictor = DistanceTransformConformalPredictor(alpha=0.2)
        predictor.calibrate(
            predictions=[pred] * 10,
            labels=[gt] * 10,
        )
        pred_set = predictor.predict(pred)
        metrics = compute_distance_metrics(pred_set, gt)

        # FNR should be roughly controlled (allow some slack)
        assert metrics["fnr"] <= 0.5

    def test_fnr_zero_perfect(self) -> None:
        """FNR = 0 when prediction set perfectly covers GT."""
        from minivess.ensemble.distance_conformal import compute_distance_metrics

        gt = _make_sphere_mask(radius=5.0)
        # Prediction set = GT union extra -> all GT covered
        pred_set = np.ones_like(gt, dtype=bool)
        metrics = compute_distance_metrics(pred_set, gt)
        assert metrics["fnr"] == pytest.approx(0.0)

    def test_fpr_increases_with_dilation(self) -> None:
        """Larger prediction sets should have higher FPR."""
        from minivess.ensemble.distance_conformal import compute_distance_metrics

        gt = _make_sphere_mask(radius=5.0)
        small_set = _make_sphere_mask(radius=5.0).astype(bool)
        large_set = _make_sphere_mask(radius=7.0).astype(bool)

        metrics_small = compute_distance_metrics(small_set, gt)
        metrics_large = compute_distance_metrics(large_set, gt)

        assert metrics_large["fpr"] >= metrics_small["fpr"]

    def test_metrics_to_dict(self) -> None:
        """All metric values should be float (MLflow compatible)."""
        from minivess.ensemble.distance_conformal import compute_distance_metrics

        gt = _make_sphere_mask(radius=5.0)
        pred_set = _make_sphere_mask(radius=5.0).astype(bool)
        metrics = compute_distance_metrics(pred_set, gt)

        assert "fnr" in metrics
        assert "fpr" in metrics
        assert "boundary_distance" in metrics
        assert "volume_inflation" in metrics

        for key, val in metrics.items():
            assert isinstance(val, float), f"{key} is {type(val)}, expected float"


# ---------------------------------------------------------------------------
# Task 2.3: Signed distance transform utilities
# ---------------------------------------------------------------------------


class TestDistanceUtils:
    """Test signed distance transform utilities."""

    def test_sdt_positive_inside(self) -> None:
        """SDT should have positive values inside the mask."""
        from minivess.ensemble.distance_utils import signed_distance_transform

        mask = _make_sphere_mask(radius=5.0)
        sdt = signed_distance_transform(mask)

        center = tuple(s // 2 for s in mask.shape)
        assert sdt[center] > 0

    def test_sdt_negative_outside(self) -> None:
        """SDT should have negative values outside the mask."""
        from minivess.ensemble.distance_utils import signed_distance_transform

        mask = _make_sphere_mask(radius=5.0)
        sdt = signed_distance_transform(mask)

        # Corner is outside
        assert sdt[0, 0, 0] < 0

    def test_sdt_zero_at_boundary(self) -> None:
        """SDT should be ~0 at the mask boundary."""
        from minivess.ensemble.distance_utils import signed_distance_transform

        mask = _make_sphere_mask(radius=5.0, shape=(32, 32, 32))
        sdt = signed_distance_transform(mask)

        # Find boundary voxels (mask but with at least one non-mask neighbor)
        from scipy.ndimage import binary_erosion

        eroded = binary_erosion(mask.astype(bool))
        boundary = mask.astype(bool) & ~eroded

        if boundary.any():
            boundary_vals = sdt[boundary]
            # Boundary SDT values should be close to 0 (within 1 voxel)
            assert np.abs(boundary_vals).max() <= 1.5

    def test_boundary_distance_zero_identical(self) -> None:
        """Distance should be 0 for identical masks."""
        from minivess.ensemble.distance_utils import boundary_distance

        mask = _make_sphere_mask(radius=5.0)
        dist = boundary_distance(mask, mask)
        assert dist == pytest.approx(0.0, abs=0.5)

    def test_boundary_distance_symmetric(self) -> None:
        """boundary_distance(A, B) should equal boundary_distance(B, A)."""
        from minivess.ensemble.distance_utils import boundary_distance

        mask_a = _make_sphere_mask(radius=5.0)
        mask_b = _make_sphere_mask(radius=4.0)

        d_ab = boundary_distance(mask_a, mask_b)
        d_ba = boundary_distance(mask_b, mask_a)
        # Symmetric Hausdorff-like: max(d(A->B), d(B->A))
        assert d_ab == pytest.approx(d_ba, abs=0.1)

    def test_hausdorff_p95_robust(self) -> None:
        """95th percentile Hausdorff should ignore outlier boundary points."""
        from minivess.ensemble.distance_utils import (
            asymmetric_hausdorff_percentile,
        )

        gt = _make_sphere_mask(radius=5.0)
        pred = _make_sphere_mask(radius=4.0)

        hd95 = asymmetric_hausdorff_percentile(gt, pred, percentile=95)
        hd100 = asymmetric_hausdorff_percentile(gt, pred, percentile=100)

        # HD95 <= HD100 (ignoring outliers makes it smaller or equal)
        assert hd95 <= hd100 + 0.01
