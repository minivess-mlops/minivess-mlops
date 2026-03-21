"""Tests for comprehensive calibration metrics module.

Tests both Tier 1 (fast) and Tier 2 (comprehensive) metrics.
"""

from __future__ import annotations

import numpy as np
import pytest


def _perfect_calibration(
    n: int = 1000, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate perfectly calibrated predictions: probs ~ labels."""
    rng = np.random.default_rng(seed)
    probs = rng.uniform(0.0, 1.0, size=n)
    labels = (rng.uniform(0.0, 1.0, size=n) < probs).astype(np.float64)
    return probs, labels


def _overconfident(n: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Generate overconfident predictions: probs always high, labels mixed."""
    probs = np.full(n, 0.9)
    labels = np.zeros(n)
    labels[: int(n * 0.5)] = 1.0
    return probs, labels


class TestECE:
    def test_perfect_calibration_low_ece(self) -> None:
        from minivess.pipeline.calibration_metrics import compute_ece

        probs, labels = _perfect_calibration()
        ece = compute_ece(probs, labels)
        assert 0.0 <= ece <= 0.1  # Should be very low

    def test_overconfident_high_ece(self) -> None:
        from minivess.pipeline.calibration_metrics import compute_ece

        probs, labels = _overconfident()
        ece = compute_ece(probs, labels)
        assert ece > 0.2  # Overconfident -> high ECE

    def test_ece_in_range(self) -> None:
        from minivess.pipeline.calibration_metrics import compute_ece

        probs, labels = _perfect_calibration()
        ece = compute_ece(probs, labels)
        assert 0.0 <= ece <= 1.0


class TestMCE:
    def test_mce_geq_ece(self) -> None:
        from minivess.pipeline.calibration_metrics import compute_ece, compute_mce

        probs, labels = _overconfident()
        ece = compute_ece(probs, labels)
        mce = compute_mce(probs, labels)
        assert mce >= ece  # MCE is always >= ECE


class TestRMSCE:
    def test_rmsce_in_range(self) -> None:
        from minivess.pipeline.calibration_metrics import compute_rmsce

        probs, labels = _perfect_calibration()
        rmsce = compute_rmsce(probs, labels)
        assert 0.0 <= rmsce <= 1.0


class TestBrierScore:
    def test_perfect_predictions_low_brier(self) -> None:
        from minivess.pipeline.calibration_metrics import compute_brier_score

        labels = np.array([0.0, 1.0, 0.0, 1.0])
        probs = np.array([0.0, 1.0, 0.0, 1.0])
        assert compute_brier_score(probs, labels) == pytest.approx(0.0)

    def test_worst_predictions_high_brier(self) -> None:
        from minivess.pipeline.calibration_metrics import compute_brier_score

        labels = np.array([0.0, 1.0, 0.0, 1.0])
        probs = np.array([1.0, 0.0, 1.0, 0.0])
        assert compute_brier_score(probs, labels) == pytest.approx(1.0)


class TestNLL:
    def test_nll_in_range(self) -> None:
        from minivess.pipeline.calibration_metrics import compute_nll

        probs, labels = _perfect_calibration()
        nll = compute_nll(probs, labels)
        assert nll >= 0.0

    def test_perfect_predictions_low_nll(self) -> None:
        from minivess.pipeline.calibration_metrics import compute_nll

        labels = np.array([0.0, 1.0])
        probs = np.array([0.01, 0.99])
        nll = compute_nll(probs, labels)
        assert nll < 0.1


class TestOverconfidenceError:
    def test_overconfident_has_positive_oe(self) -> None:
        from minivess.pipeline.calibration_metrics import compute_overconfidence_error

        probs, labels = _overconfident()
        oe = compute_overconfidence_error(probs, labels)
        assert oe > 0.0

    def test_oe_in_range(self) -> None:
        from minivess.pipeline.calibration_metrics import compute_overconfidence_error

        probs, labels = _perfect_calibration()
        oe = compute_overconfidence_error(probs, labels)
        assert 0.0 <= oe <= 1.0


class TestDebiasedECE:
    def test_debiased_ece_less_than_or_equal_ece(self) -> None:
        from minivess.pipeline.calibration_metrics import (
            compute_debiased_ece,
            compute_ece,
        )

        probs, labels = _perfect_calibration(n=500)
        ece = compute_ece(probs, labels)
        dece = compute_debiased_ece(probs, labels)
        # Debiased should be <= ECE (it subtracts the bias)
        assert dece <= ece + 0.01  # small tolerance


class TestACE:
    def test_ace_in_range(self) -> None:
        from minivess.pipeline.calibration_metrics import compute_ace

        probs, labels = _perfect_calibration()
        ace = compute_ace(probs, labels)
        assert 0.0 <= ace <= 1.0


class TestBAECE:
    def test_ba_ece_flat_input_returns_value(self) -> None:
        from minivess.pipeline.calibration_metrics import compute_ba_ece

        probs, labels = _perfect_calibration(n=1000)
        ba_ece = compute_ba_ece(probs, labels)
        assert 0.0 <= ba_ece <= 1.0

    def test_ba_ece_3d_input(self) -> None:
        from minivess.pipeline.calibration_metrics import compute_ba_ece

        # Create a simple 3D volume
        rng = np.random.default_rng(42)
        labels_3d = np.zeros((8, 8, 8))
        labels_3d[2:6, 2:6, 2:6] = 1.0  # Cube in center
        probs_3d = (
            labels_3d * 0.8
            + (1 - labels_3d) * 0.2
            + rng.normal(0, 0.05, labels_3d.shape)
        )
        probs_3d = np.clip(probs_3d, 0.0, 1.0)
        ba_ece = compute_ba_ece(probs_3d, labels_3d)
        assert 0.0 <= ba_ece <= 1.0


class TestBrierMap:
    def test_brier_map_shape(self) -> None:
        from minivess.pipeline.calibration_metrics import compute_brier_map

        probs = np.array([0.9, 0.1, 0.5])
        labels = np.array([1.0, 0.0, 1.0])
        bmap = compute_brier_map(probs, labels)
        assert bmap.shape == probs.shape

    def test_brier_map_perfect(self) -> None:
        from minivess.pipeline.calibration_metrics import compute_brier_map

        probs = np.array([0.0, 1.0])
        labels = np.array([0.0, 1.0])
        bmap = compute_brier_map(probs, labels)
        np.testing.assert_array_almost_equal(bmap, [0.0, 0.0])


class TestNLLMap:
    def test_nll_map_shape(self) -> None:
        from minivess.pipeline.calibration_metrics import compute_nll_map

        probs = np.array([0.9, 0.1, 0.5])
        labels = np.array([1.0, 0.0, 1.0])
        nmap = compute_nll_map(probs, labels)
        assert nmap.shape == probs.shape


class TestComputeAllCalibrationMetrics:
    def test_fast_tier_returns_tier1_only(self) -> None:
        from minivess.pipeline.calibration_metrics import (
            compute_all_calibration_metrics,
        )

        probs, labels = _perfect_calibration()
        result = compute_all_calibration_metrics(probs, labels, tier="fast")
        assert "ece" in result
        assert "mce" in result
        assert "brier" in result
        assert "nll" in result
        # Tier 2 should NOT be in fast
        assert "ace" not in result

    def test_comprehensive_tier_returns_all(self) -> None:
        from minivess.pipeline.calibration_metrics import (
            compute_all_calibration_metrics,
        )

        probs, labels = _perfect_calibration()
        result = compute_all_calibration_metrics(probs, labels, tier="comprehensive")
        assert "ece" in result
        assert "ace" in result
        assert "ba_ece" in result
        # All values should be float
        for key, val in result.items():
            assert isinstance(val, float), f"{key} is not float: {type(val)}"

    def test_all_metrics_finite(self) -> None:
        from minivess.pipeline.calibration_metrics import (
            compute_all_calibration_metrics,
        )

        probs, labels = _perfect_calibration()
        result = compute_all_calibration_metrics(probs, labels, tier="comprehensive")
        for key, val in result.items():
            assert np.isfinite(val), f"{key} is not finite: {val}"

    def test_invalid_tier_raises(self) -> None:
        from minivess.pipeline.calibration_metrics import (
            compute_all_calibration_metrics,
        )

        probs, labels = _perfect_calibration()
        with pytest.raises(ValueError, match="tier must be"):
            compute_all_calibration_metrics(probs, labels, tier="invalid")
