"""Property-based tests using Hypothesis (Code Review R2.6).

Tests mathematical invariants that should hold for ANY valid input:
- bootstrap_ci: lower <= point_estimate <= upper
- ConfidenceInterval: width >= 0, lower <= upper
- temperature_scale: output is valid probability distribution
- compute_prediction_risk: risk >= 0, shape preserved
- Dice score via SegmentationMetrics: result in [0, 1]
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

# ---------------------------------------------------------------------------
# T1: ConfidenceInterval invariants
# ---------------------------------------------------------------------------


class TestConfidenceIntervalProperties:
    """Property-based tests for ConfidenceInterval."""

    @given(
        lower=st.floats(min_value=-10.0, max_value=10.0),
        width=st.floats(min_value=0.0, max_value=10.0),
    )
    @settings(deadline=None)
    def test_width_is_nonnegative(self, lower: float, width: float) -> None:
        """CI width should always be >= 0."""
        from minivess.pipeline.ci import ConfidenceInterval

        upper = lower + width
        point = (lower + upper) / 2
        ci = ConfidenceInterval(
            point_estimate=point,
            lower=lower,
            upper=upper,
            confidence_level=0.95,
            method="test",
        )
        assert ci.width >= 0.0

    @given(
        lower=st.floats(min_value=0.0, max_value=5.0),
        upper=st.floats(min_value=0.0, max_value=5.0),
    )
    @settings(deadline=None)
    def test_lower_exceeds_upper_raises(self, lower: float, upper: float) -> None:
        """ConfidenceInterval should reject lower > upper."""
        from minivess.pipeline.ci import ConfidenceInterval

        if lower > upper:
            with pytest.raises(ValueError, match="lower"):
                ConfidenceInterval(
                    point_estimate=0.5,
                    lower=lower,
                    upper=upper,
                    confidence_level=0.95,
                    method="test",
                )


# ---------------------------------------------------------------------------
# T2: bootstrap_ci ordering invariant
# ---------------------------------------------------------------------------


class TestBootstrapCIProperties:
    """Property-based tests for bootstrap_ci."""

    @given(
        data=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=5, max_value=50),
            elements=st.floats(min_value=0.0, max_value=1.0),
        ),
    )
    @settings(max_examples=20, deadline=5000)
    def test_ci_bounds_ordering(self, data: np.ndarray) -> None:
        """bootstrap_ci should always produce lower <= point <= upper."""
        from minivess.pipeline.ci import bootstrap_ci

        ci = bootstrap_ci(data, n_resamples=200, seed=42)
        assert ci.lower <= ci.point_estimate <= ci.upper

    @given(
        level=st.floats(min_value=0.50, max_value=0.99),
    )
    @settings(max_examples=10, deadline=5000)
    def test_ci_width_increases_with_confidence(self, level: float) -> None:
        """Higher confidence levels should not produce narrower CIs."""
        from minivess.pipeline.ci import bootstrap_ci

        rng = np.random.default_rng(42)
        data = rng.random(30)
        ci_low = bootstrap_ci(data, confidence_level=0.80, n_resamples=500, seed=42)
        ci_high = bootstrap_ci(data, confidence_level=level, n_resamples=500, seed=42)
        if level > 0.80:
            assert ci_high.width >= ci_low.width * 0.9  # Allow small numerical noise


# ---------------------------------------------------------------------------
# T3: temperature_scale properties
# ---------------------------------------------------------------------------


class TestTemperatureScaleProperties:
    """Property-based tests for temperature scaling."""

    @given(
        temp=st.floats(min_value=0.1, max_value=10.0),
    )
    @settings(max_examples=20)
    def test_output_sums_to_one(self, temp: float) -> None:
        """temperature_scale output should sum to ~1.0 along class axis."""
        from minivess.ensemble.calibration import temperature_scale

        rng = np.random.default_rng(42)
        logits = rng.standard_normal((5, 3))
        probs = temperature_scale(logits, temp)
        sums = probs.sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)

    @given(
        temp=st.floats(min_value=0.1, max_value=10.0),
    )
    @settings(max_examples=20)
    def test_output_in_01_range(self, temp: float) -> None:
        """temperature_scale output should be in [0, 1]."""
        from minivess.ensemble.calibration import temperature_scale

        rng = np.random.default_rng(42)
        logits = rng.standard_normal((5, 3))
        probs = temperature_scale(logits, temp)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)


# ---------------------------------------------------------------------------
# T4: compute_prediction_risk properties
# ---------------------------------------------------------------------------


class TestPredictionRiskProperties:
    """Property-based tests for prediction risk computation."""

    @given(
        data=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=100),
            elements=st.floats(min_value=0.0, max_value=1.0),
        ),
    )
    @settings(max_examples=20)
    def test_risk_is_nonnegative(self, data: np.ndarray) -> None:
        """Prediction risk should always be >= 0."""
        from minivess.observability.pprm import compute_prediction_risk

        labels = np.zeros_like(data)
        risk = compute_prediction_risk(data, labels)
        assert np.all(risk >= 0.0)

    @given(
        data=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=100),
            elements=st.floats(min_value=0.0, max_value=1.0),
        ),
    )
    @settings(max_examples=20)
    def test_risk_shape_preserved(self, data: np.ndarray) -> None:
        """Risk output should have same shape as input."""
        from minivess.observability.pprm import compute_prediction_risk

        labels = np.ones_like(data)
        risk = compute_prediction_risk(data, labels)
        assert risk.shape == data.shape


# ---------------------------------------------------------------------------
# T5: Dice score via SegmentationMetrics
# ---------------------------------------------------------------------------


class TestDiceScoreProperties:
    """Property-based tests for Dice score invariants."""

    @given(
        size=st.integers(min_value=4, max_value=16),
    )
    @settings(max_examples=10, deadline=5000)
    def test_dice_in_zero_one(self, size: int) -> None:
        """Dice score should always be in [0, 1]."""
        from minivess.pipeline.metrics import SegmentationMetrics

        metrics = SegmentationMetrics(num_classes=2)
        pred = torch.randint(0, 2, (1, size, size, size))
        label = torch.randint(0, 2, (1, size, size, size))
        metrics.update(pred, label)
        result = metrics.compute()
        assert 0.0 <= result.values["dice"] <= 1.0

    def test_dice_perfect_prediction_is_one(self) -> None:
        """Dice should be 1.0 when prediction exactly matches label."""
        from minivess.pipeline.metrics import SegmentationMetrics

        metrics = SegmentationMetrics(num_classes=2)
        label = torch.zeros(1, 16, 16, 16, dtype=torch.long)
        label[0, 4:12, 4:12, 4:12] = 1
        metrics.update(label, label)
        result = metrics.compute()
        assert result.values["dice"] == pytest.approx(1.0, abs=1e-5)
