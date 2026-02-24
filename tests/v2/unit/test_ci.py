"""Tests for confidence interval reporting (Issue #6)."""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# T1: ConfidenceInterval dataclass
# ---------------------------------------------------------------------------


class TestConfidenceInterval:
    """Test ConfidenceInterval dataclass."""

    def test_construction(self) -> None:
        """ConfidenceInterval should store point estimate and bounds."""
        from minivess.pipeline.ci import ConfidenceInterval

        ci = ConfidenceInterval(
            point_estimate=0.85,
            lower=0.82,
            upper=0.88,
            confidence_level=0.95,
            method="percentile_bootstrap",
        )
        assert ci.point_estimate == 0.85
        assert ci.lower == 0.82
        assert ci.upper == 0.88
        assert ci.confidence_level == 0.95
        assert ci.method == "percentile_bootstrap"

    def test_to_dict(self) -> None:
        """to_dict should return flat dict for MLflow logging."""
        from minivess.pipeline.ci import ConfidenceInterval

        ci = ConfidenceInterval(
            point_estimate=0.85,
            lower=0.82,
            upper=0.88,
            confidence_level=0.95,
            method="percentile_bootstrap",
        )
        d = ci.to_dict("dice")
        assert d["dice"] == 0.85
        assert d["dice_ci_lower"] == 0.82
        assert d["dice_ci_upper"] == 0.88
        assert d["dice_ci_level"] == 0.95

    def test_width(self) -> None:
        """width property should return upper - lower."""
        from minivess.pipeline.ci import ConfidenceInterval

        ci = ConfidenceInterval(
            point_estimate=0.85,
            lower=0.80,
            upper=0.90,
            confidence_level=0.95,
            method="percentile_bootstrap",
        )
        assert abs(ci.width - 0.10) < 1e-9

    def test_lower_must_not_exceed_upper(self) -> None:
        """Should raise if lower > upper."""
        from minivess.pipeline.ci import ConfidenceInterval

        with pytest.raises(ValueError, match="lower.*upper"):
            ConfidenceInterval(
                point_estimate=0.85,
                lower=0.90,
                upper=0.80,
                confidence_level=0.95,
                method="percentile_bootstrap",
            )


# ---------------------------------------------------------------------------
# T2: Percentile bootstrap CI
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    """Test percentile bootstrap confidence intervals."""

    def test_known_distribution(self) -> None:
        """Bootstrap CI of a normal sample should contain the true mean."""
        from minivess.pipeline.ci import bootstrap_ci

        rng = np.random.default_rng(42)
        samples = rng.normal(loc=0.85, scale=0.05, size=100)
        ci = bootstrap_ci(samples, statistic=np.mean, confidence_level=0.95)
        assert ci.lower <= 0.85 <= ci.upper
        assert ci.method == "percentile_bootstrap"
        assert ci.confidence_level == 0.95

    def test_more_samples_narrower_ci(self) -> None:
        """Larger sample size should yield narrower CI."""
        from minivess.pipeline.ci import bootstrap_ci

        rng = np.random.default_rng(42)
        small = rng.normal(loc=0.85, scale=0.05, size=20)
        large = rng.normal(loc=0.85, scale=0.05, size=200)

        ci_small = bootstrap_ci(small, statistic=np.mean)
        ci_large = bootstrap_ci(large, statistic=np.mean)
        assert ci_large.width < ci_small.width

    def test_higher_confidence_wider_ci(self) -> None:
        """99% CI should be wider than 90% CI."""
        from minivess.pipeline.ci import bootstrap_ci

        rng = np.random.default_rng(42)
        samples = rng.normal(loc=0.85, scale=0.05, size=100)

        ci_90 = bootstrap_ci(samples, statistic=np.mean, confidence_level=0.90)
        ci_99 = bootstrap_ci(samples, statistic=np.mean, confidence_level=0.99)
        assert ci_99.width > ci_90.width

    def test_custom_statistic(self) -> None:
        """Should work with any callable statistic (e.g., median)."""
        from minivess.pipeline.ci import bootstrap_ci

        rng = np.random.default_rng(42)
        samples = rng.normal(loc=0.85, scale=0.05, size=100)
        ci = bootstrap_ci(samples, statistic=np.median)
        assert ci.lower < ci.point_estimate < ci.upper

    def test_reproducible_with_seed(self) -> None:
        """Same seed should give identical CIs."""
        from minivess.pipeline.ci import bootstrap_ci

        rng = np.random.default_rng(42)
        samples = rng.normal(loc=0.85, scale=0.05, size=50)

        ci1 = bootstrap_ci(samples, statistic=np.mean, seed=123)
        ci2 = bootstrap_ci(samples, statistic=np.mean, seed=123)
        assert ci1.lower == ci2.lower
        assert ci1.upper == ci2.upper


# ---------------------------------------------------------------------------
# T3: BCa bootstrap CI
# ---------------------------------------------------------------------------


class TestBcaBootstrapCI:
    """Test BCa bootstrap for small samples."""

    def test_known_distribution(self) -> None:
        """BCa CI should contain the true mean."""
        from minivess.pipeline.ci import bca_bootstrap_ci

        rng = np.random.default_rng(42)
        samples = rng.normal(loc=0.85, scale=0.05, size=30)
        ci = bca_bootstrap_ci(samples, statistic=np.mean, confidence_level=0.95)
        assert ci.lower <= 0.85 <= ci.upper
        assert ci.method == "bca_bootstrap"

    def test_small_sample(self) -> None:
        """BCa should work with very small samples (n=10)."""
        from minivess.pipeline.ci import bca_bootstrap_ci

        rng = np.random.default_rng(42)
        samples = rng.normal(loc=0.85, scale=0.05, size=10)
        ci = bca_bootstrap_ci(samples, statistic=np.mean, confidence_level=0.95)
        assert ci.lower < ci.upper
        assert ci.confidence_level == 0.95


# ---------------------------------------------------------------------------
# T4: compute_metrics_with_ci — integration
# ---------------------------------------------------------------------------


class TestComputeMetricsWithCI:
    """Test integrated metric + CI computation."""

    def test_dice_with_ci(self) -> None:
        """compute_metrics_with_ci should return CI for Dice scores."""
        from minivess.pipeline.ci import compute_metrics_with_ci

        rng = np.random.default_rng(42)
        per_sample_dice = rng.normal(loc=0.85, scale=0.05, size=50)
        per_sample_dice = np.clip(per_sample_dice, 0.0, 1.0)

        result = compute_metrics_with_ci({"dice": per_sample_dice})
        assert "dice" in result
        ci = result["dice"]
        assert ci.lower <= ci.point_estimate <= ci.upper
        assert ci.confidence_level == 0.95

    def test_multiple_metrics(self) -> None:
        """Should compute CIs for multiple metrics simultaneously."""
        from minivess.pipeline.ci import compute_metrics_with_ci

        rng = np.random.default_rng(42)
        metrics = {
            "dice": rng.normal(loc=0.85, scale=0.05, size=50),
            "hd95": rng.normal(loc=3.5, scale=1.0, size=50),
        }

        result = compute_metrics_with_ci(metrics)
        assert len(result) == 2
        assert "dice" in result
        assert "hd95" in result

    def test_to_flat_dict(self) -> None:
        """Should convert all CIs to flat dict for MLflow logging."""
        from minivess.pipeline.ci import compute_metrics_with_ci

        rng = np.random.default_rng(42)
        result = compute_metrics_with_ci(
            {"dice": rng.normal(loc=0.85, scale=0.05, size=50)}
        )
        flat = {}
        for name, ci in result.items():
            flat.update(ci.to_dict(name))
        assert "dice" in flat
        assert "dice_ci_lower" in flat
        assert "dice_ci_upper" in flat
        assert "dice_ci_level" in flat
