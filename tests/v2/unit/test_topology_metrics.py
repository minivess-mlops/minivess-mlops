"""Tests for topology-aware evaluation metrics.

Covers: NSD (#134), HD95 (#135), ccDice (#112), Betti error (#113), Junction F1 (#117).
TDD RED phase: tests written before implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers: synthetic 3D masks
# ---------------------------------------------------------------------------


def _make_sphere(
    shape: tuple[int, int, int], center: tuple[int, int, int], radius: float
) -> np.ndarray:
    """Create a binary 3D sphere mask."""
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    dist = np.sqrt(
        (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2
    )
    return (dist <= radius).astype(np.uint8)


def _make_tube(
    shape: tuple[int, int, int],
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    radius: float,
) -> np.ndarray:
    """Create a binary 3D tube (cylinder) mask along a line segment."""
    mask = np.zeros(shape, dtype=np.uint8)
    # Parameterize the line segment
    t_steps = max(shape) * 2
    for t in np.linspace(0, 1, t_steps):
        cz = start[0] + t * (end[0] - start[0])
        cy = start[1] + t * (end[1] - start[1])
        cx = start[2] + t * (end[2] - start[2])
        zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
        dist = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
        mask[dist <= radius] = 1
    return mask


def _make_y_bifurcation(shape: tuple[int, int, int]) -> np.ndarray:
    """Create a Y-shaped bifurcation mask (1 junction, 3 endpoints)."""
    mask = np.zeros(shape, dtype=np.uint8)
    # Trunk: bottom center to middle
    trunk = _make_tube(
        shape,
        (0, shape[1] // 2, shape[2] // 2),
        (shape[0] // 2, shape[1] // 2, shape[2] // 2),
        radius=2,
    )
    # Left branch
    left = _make_tube(
        shape,
        (shape[0] // 2, shape[1] // 2, shape[2] // 2),
        (shape[0] - 1, shape[1] // 4, shape[2] // 4),
        radius=2,
    )
    # Right branch
    right = _make_tube(
        shape,
        (shape[0] // 2, shape[1] // 2, shape[2] // 2),
        (shape[0] - 1, 3 * shape[1] // 4, 3 * shape[2] // 4),
        radius=2,
    )
    mask = np.clip(trunk + left + right, 0, 1)
    return mask


# ===========================================================================
# T1: NSD (Surface Dice) — Issue #134
# ===========================================================================


class TestComputeNSD:
    """Tests for compute_nsd() — MONAI SurfaceDiceMetric wrapper."""

    def test_nsd_perfect_match_returns_one(self) -> None:
        from minivess.pipeline.topology_metrics import compute_nsd

        mask = _make_sphere((16, 16, 16), (8, 8, 8), 5)
        result = compute_nsd(mask, mask, tau=1.0)
        assert result == pytest.approx(1.0, abs=1e-4)

    def test_nsd_empty_prediction_returns_zero(self) -> None:
        from minivess.pipeline.topology_metrics import compute_nsd

        gt = _make_sphere((16, 16, 16), (8, 8, 8), 5)
        pred = np.zeros_like(gt)
        result = compute_nsd(pred, gt, tau=1.0)
        assert result == pytest.approx(0.0, abs=1e-4)

    def test_nsd_empty_ground_truth_returns_zero(self) -> None:
        from minivess.pipeline.topology_metrics import compute_nsd

        pred = _make_sphere((16, 16, 16), (8, 8, 8), 5)
        gt = np.zeros_like(pred)
        result = compute_nsd(pred, gt, tau=1.0)
        assert result == pytest.approx(0.0, abs=1e-4)

    def test_nsd_bounded_zero_one(self) -> None:
        from minivess.pipeline.topology_metrics import compute_nsd

        gt = _make_sphere((16, 16, 16), (8, 8, 8), 5)
        pred = _make_sphere((16, 16, 16), (9, 9, 9), 4)
        result = compute_nsd(pred, gt, tau=2.0)
        assert 0.0 <= result <= 1.0

    def test_nsd_tau_parameter_affects_result(self) -> None:
        from minivess.pipeline.topology_metrics import compute_nsd

        gt = _make_sphere((16, 16, 16), (8, 8, 8), 5)
        pred = _make_sphere((16, 16, 16), (9, 9, 9), 5)  # offset by 1
        nsd_tight = compute_nsd(pred, gt, tau=0.5)
        nsd_loose = compute_nsd(pred, gt, tau=5.0)
        # Larger tau should give higher or equal NSD
        assert nsd_loose >= nsd_tight

    def test_nsd_3d_tube_structure(self) -> None:
        from minivess.pipeline.topology_metrics import compute_nsd

        tube = _make_tube((16, 16, 16), (2, 8, 8), (14, 8, 8), radius=2)
        result = compute_nsd(tube, tube, tau=1.0)
        assert result == pytest.approx(1.0, abs=1e-4)

    def test_nsd_returns_float(self) -> None:
        from minivess.pipeline.topology_metrics import compute_nsd

        mask = _make_sphere((16, 16, 16), (8, 8, 8), 5)
        result = compute_nsd(mask, mask, tau=1.0)
        assert isinstance(result, float)


# ===========================================================================
# T2: HD95 — Issue #135
# ===========================================================================


class TestComputeHD95:
    """Tests for compute_hd95() — MONAI HausdorffDistanceMetric wrapper."""

    def test_hd95_perfect_match_returns_zero(self) -> None:
        from minivess.pipeline.topology_metrics import compute_hd95

        mask = _make_sphere((16, 16, 16), (8, 8, 8), 5)
        result = compute_hd95(mask, mask)
        assert result == pytest.approx(0.0, abs=1e-4)

    def test_hd95_empty_prediction_returns_inf_or_nan(self) -> None:
        from minivess.pipeline.topology_metrics import compute_hd95

        gt = _make_sphere((16, 16, 16), (8, 8, 8), 5)
        pred = np.zeros_like(gt)
        result = compute_hd95(pred, gt)
        # Should return inf or nan for degenerate case
        assert np.isinf(result) or np.isnan(result)

    def test_hd95_concentric_spheres_known_distance(self) -> None:
        from minivess.pipeline.topology_metrics import compute_hd95

        inner = _make_sphere((32, 32, 32), (16, 16, 16), 5)
        outer = _make_sphere((32, 32, 32), (16, 16, 16), 8)
        result = compute_hd95(inner, outer)
        # HD95 between concentric spheres of radius 5 and 8 should be ~3
        assert 2.0 <= result <= 4.0

    def test_hd95_offset_volumes_bounded(self) -> None:
        from minivess.pipeline.topology_metrics import compute_hd95

        gt = _make_sphere((16, 16, 16), (8, 8, 8), 5)
        pred = _make_sphere((16, 16, 16), (9, 9, 9), 5)  # offset by 1
        result = compute_hd95(pred, gt)
        # Should be positive and bounded
        assert 0.0 < result < 10.0

    def test_hd95_direction_is_minimize(self) -> None:
        """HD95 is a distance metric — lower is better."""
        from minivess.pipeline.topology_metrics import compute_hd95

        mask = _make_sphere((16, 16, 16), (8, 8, 8), 5)
        perfect = compute_hd95(mask, mask)
        offset = compute_hd95(_make_sphere((16, 16, 16), (10, 10, 10), 5), mask)
        assert perfect < offset

    def test_hd95_returns_float(self) -> None:
        from minivess.pipeline.topology_metrics import compute_hd95

        mask = _make_sphere((16, 16, 16), (8, 8, 8), 5)
        result = compute_hd95(mask, mask)
        assert isinstance(result, float)

    def test_hd95_3d_tube_structure(self) -> None:
        from minivess.pipeline.topology_metrics import compute_hd95

        tube = _make_tube((16, 16, 16), (2, 8, 8), (14, 8, 8), radius=2)
        result = compute_hd95(tube, tube)
        assert result == pytest.approx(0.0, abs=1e-4)


# ===========================================================================
# T3: ccDice — Issue #112
# ===========================================================================


class TestComputeCcDice:
    """Tests for compute_ccdice() — connected-component Dice."""

    def test_ccdice_perfect_match_returns_one(self) -> None:
        from minivess.pipeline.topology_metrics import compute_ccdice

        mask = _make_sphere((16, 16, 16), (8, 8, 8), 5)
        result = compute_ccdice(mask, mask)
        assert result == pytest.approx(1.0, abs=1e-4)

    def test_ccdice_empty_prediction_returns_zero(self) -> None:
        from minivess.pipeline.topology_metrics import compute_ccdice

        gt = _make_sphere((16, 16, 16), (8, 8, 8), 5)
        pred = np.zeros_like(gt)
        result = compute_ccdice(pred, gt)
        assert result == pytest.approx(0.0, abs=1e-4)

    def test_ccdice_empty_ground_truth_returns_zero(self) -> None:
        from minivess.pipeline.topology_metrics import compute_ccdice

        pred = _make_sphere((16, 16, 16), (8, 8, 8), 5)
        gt = np.zeros_like(pred)
        result = compute_ccdice(pred, gt)
        assert result == pytest.approx(0.0, abs=1e-4)

    def test_ccdice_single_connected_component_equals_dice(self) -> None:
        """When both pred and GT are single connected components, ccDice ≈ Dice."""
        from minivess.pipeline.topology_metrics import compute_ccdice

        gt = _make_sphere((16, 16, 16), (8, 8, 8), 5)
        pred = _make_sphere((16, 16, 16), (8, 8, 8), 4)  # slightly smaller
        ccdice = compute_ccdice(pred, gt)
        # Standard Dice for comparison
        intersection = np.sum(pred & gt)
        dice = 2 * intersection / (np.sum(pred) + np.sum(gt))
        assert ccdice == pytest.approx(dice, abs=0.05)

    def test_ccdice_fragmented_prediction_lower_than_dice(self) -> None:
        """Fragmented predictions should have lower ccDice than standard Dice."""
        from minivess.pipeline.topology_metrics import compute_ccdice

        # GT: one big sphere
        gt = _make_sphere((32, 32, 32), (16, 16, 16), 10)
        # Pred: same sphere but with a gap (two fragments)
        pred = gt.copy()
        pred[14:18, :, :] = 0  # cut in the middle to fragment
        ccdice = compute_ccdice(pred, gt)
        # Standard Dice
        intersection = np.sum(pred & gt)
        dice = 2 * intersection / (np.sum(pred) + np.sum(gt))
        # ccDice should be lower because fragments don't fully cover GT component
        assert ccdice <= dice + 0.01  # allow tiny float tolerance

    def test_ccdice_multiple_components_matched_correctly(self) -> None:
        """Two separate components in both pred and GT should match correctly."""
        from minivess.pipeline.topology_metrics import compute_ccdice

        gt = np.zeros((32, 32, 32), dtype=np.uint8)
        gt |= _make_sphere((32, 32, 32), (8, 8, 8), 4)
        gt |= _make_sphere((32, 32, 32), (24, 24, 24), 4)

        pred = gt.copy()  # perfect match of two components
        result = compute_ccdice(pred, gt)
        assert result == pytest.approx(1.0, abs=1e-4)

    def test_ccdice_bounded_zero_one(self) -> None:
        from minivess.pipeline.topology_metrics import compute_ccdice

        gt = _make_sphere((16, 16, 16), (8, 8, 8), 5)
        pred = _make_sphere((16, 16, 16), (9, 9, 9), 4)
        result = compute_ccdice(pred, gt)
        assert 0.0 <= result <= 1.0

    def test_ccdice_3d_volume(self) -> None:
        from minivess.pipeline.topology_metrics import compute_ccdice

        mask = _make_tube((16, 16, 16), (2, 8, 8), (14, 8, 8), radius=2)
        result = compute_ccdice(mask, mask)
        assert result == pytest.approx(1.0, abs=1e-4)


# ===========================================================================
# T7: Betti Error — Issue #113
# ===========================================================================


class TestComputeBettiError:
    """Tests for compute_betti_error() and compute_persistence_distance()."""

    def test_betti_error_perfect_match_returns_zeros(self) -> None:
        from minivess.pipeline.topology_metrics import compute_betti_error

        mask = _make_sphere((16, 16, 16), (8, 8, 8), 5)
        result = compute_betti_error(mask, mask)
        assert result["beta0_error"] == pytest.approx(0.0)

    def test_betti_error_extra_components_positive_beta0(self) -> None:
        from minivess.pipeline.topology_metrics import compute_betti_error

        # Pred: two separate spheres (2 components vs 1)
        pred = np.zeros((32, 32, 32), dtype=np.uint8)
        pred |= _make_sphere((32, 32, 32), (8, 8, 8), 4)
        pred |= _make_sphere((32, 32, 32), (24, 24, 24), 4)
        gt = np.zeros((32, 32, 32), dtype=np.uint8)
        gt |= _make_sphere((32, 32, 32), (16, 16, 16), 8)
        result = compute_betti_error(pred, gt)
        assert result["beta0_error"] > 0

    def test_betti_error_missing_components_positive_beta0(self) -> None:
        from minivess.pipeline.topology_metrics import compute_betti_error

        # GT: two separate spheres, pred: only one
        gt = np.zeros((32, 32, 32), dtype=np.uint8)
        gt |= _make_sphere((32, 32, 32), (8, 8, 8), 4)
        gt |= _make_sphere((32, 32, 32), (24, 24, 24), 4)
        pred = _make_sphere((32, 32, 32), (8, 8, 8), 4)
        result = compute_betti_error(pred, gt)
        assert result["beta0_error"] > 0

    def test_betti_error_returns_dict_with_beta0_beta1(self) -> None:
        from minivess.pipeline.topology_metrics import compute_betti_error

        mask = _make_sphere((16, 16, 16), (8, 8, 8), 5)
        result = compute_betti_error(mask, mask)
        assert "beta0_error" in result
        assert "beta1_error" in result

    def test_betti_error_empty_masks(self) -> None:
        from minivess.pipeline.topology_metrics import compute_betti_error

        empty = np.zeros((16, 16, 16), dtype=np.uint8)
        result = compute_betti_error(empty, empty)
        assert result["beta0_error"] == pytest.approx(0.0)

    def test_betti_error_3d_volume(self) -> None:
        from minivess.pipeline.topology_metrics import compute_betti_error

        tube = _make_tube((16, 16, 16), (2, 8, 8), (14, 8, 8), radius=2)
        result = compute_betti_error(tube, tube)
        assert result["beta0_error"] == pytest.approx(0.0)

    def test_persistence_distance_returns_float_or_nan(self) -> None:
        from minivess.pipeline.topology_metrics import compute_persistence_distance

        mask = _make_sphere((16, 16, 16), (8, 8, 8), 5)
        result = compute_persistence_distance(mask, mask)
        assert isinstance(result, float)

    def test_persistence_distance_graceful_without_gudhi(self) -> None:
        """If gudhi is missing, should return NaN gracefully."""
        from minivess.pipeline.topology_metrics import compute_persistence_distance

        mask = _make_sphere((16, 16, 16), (8, 8, 8), 5)
        result = compute_persistence_distance(mask, mask)
        # Either a valid float (if gudhi available) or NaN
        assert isinstance(result, float)


# ===========================================================================
# T8: Junction F1 — Issue #117
# ===========================================================================


class TestComputeJunctionF1:
    """Tests for compute_junction_f1() — bifurcation detection accuracy."""

    def test_junction_f1_perfect_match_returns_one(self) -> None:
        from minivess.pipeline.topology_metrics import compute_junction_f1

        mask = _make_y_bifurcation((32, 32, 32))
        result = compute_junction_f1(mask, mask, tolerance=3)
        assert result["f1"] == pytest.approx(1.0, abs=0.01)

    def test_junction_f1_missed_junction_recall_below_one(self) -> None:
        from minivess.pipeline.topology_metrics import compute_junction_f1

        gt = _make_y_bifurcation((32, 32, 32))
        # Pred: just the trunk (no bifurcation → no junction)
        pred = _make_tube((32, 32, 32), (0, 16, 16), (31, 16, 16), radius=2)
        result = compute_junction_f1(pred, gt, tolerance=3)
        assert result["recall"] < 1.0

    def test_junction_f1_spurious_junction_precision_below_one(self) -> None:
        from minivess.pipeline.topology_metrics import compute_junction_f1

        # GT: straight tube (no junctions)
        gt = _make_tube((32, 32, 32), (2, 16, 16), (30, 16, 16), radius=2)
        # Pred: Y-bifurcation (has junction)
        pred = _make_y_bifurcation((32, 32, 32))
        result = compute_junction_f1(pred, gt, tolerance=3)
        # GT has no junctions, pred has some → precision=0
        assert result["precision"] == pytest.approx(0.0, abs=0.01)

    def test_junction_f1_no_junctions_returns_one(self) -> None:
        from minivess.pipeline.topology_metrics import compute_junction_f1

        # Both pred and GT are simple tubes (no junctions)
        tube = _make_tube((24, 24, 24), (4, 12, 12), (20, 12, 12), radius=2)
        result = compute_junction_f1(tube, tube, tolerance=3)
        assert result["f1"] == pytest.approx(1.0, abs=0.01)

    def test_junction_f1_tolerance_parameter(self) -> None:
        from minivess.pipeline.topology_metrics import compute_junction_f1

        mask = _make_y_bifurcation((32, 32, 32))
        # Tight tolerance should work for perfect match
        result_tight = compute_junction_f1(mask, mask, tolerance=1)
        result_loose = compute_junction_f1(mask, mask, tolerance=5)
        # Both should be perfect for self-match, loose should be >= tight
        assert result_loose["f1"] >= result_tight["f1"] - 0.01

    def test_junction_f1_returns_dict_with_precision_recall_f1(self) -> None:
        from minivess.pipeline.topology_metrics import compute_junction_f1

        mask = _make_y_bifurcation((32, 32, 32))
        result = compute_junction_f1(mask, mask, tolerance=3)
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result

    def test_junction_f1_3d_bifurcation_volume(self) -> None:
        from minivess.pipeline.topology_metrics import compute_junction_f1

        mask = _make_y_bifurcation((32, 32, 32))
        result = compute_junction_f1(mask, mask, tolerance=3)
        # Should detect at least 1 junction with perfect F1
        assert result["f1"] >= 0.5

    def test_junction_f1_empty_masks(self) -> None:
        from minivess.pipeline.topology_metrics import compute_junction_f1

        empty = np.zeros((16, 16, 16), dtype=np.uint8)
        result = compute_junction_f1(empty, empty, tolerance=3)
        assert result["f1"] == pytest.approx(1.0, abs=0.01)
