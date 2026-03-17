"""Tests for epoch-0 cost prediction (issue #717).

Verifies estimate_cost_from_first_epoch() extrapolates total job cost
from a single measured epoch time.
"""

from __future__ import annotations


class TestEstimateCostFromFirstEpoch:
    """estimate_cost_from_first_epoch must predict total job cost."""

    def test_predict_basic(self) -> None:
        """22 min/epoch × 5 epochs × $0.19/hr (no setup) = $0.3483."""
        from minivess.observability.infrastructure_timing import (
            estimate_cost_from_first_epoch,
        )

        result = estimate_cost_from_first_epoch(
            epoch_seconds=22 * 60,  # 22 min
            max_epochs=5,
            num_folds=1,
            hourly_rate_usd=0.19,
            setup_minutes=0,  # exclude setup for clean calculation
        )
        # 22*60*5 = 6600s = 1.833h × $0.19 = $0.3483
        assert abs(result["est/total_cost"] - 0.3483) < 0.01

    def test_predict_with_folds(self) -> None:
        """22 min × 5 epochs × 3 folds × $0.19/hr (no setup) = $1.045."""
        from minivess.observability.infrastructure_timing import (
            estimate_cost_from_first_epoch,
        )

        result = estimate_cost_from_first_epoch(
            epoch_seconds=22 * 60,
            max_epochs=5,
            num_folds=3,
            hourly_rate_usd=0.19,
            setup_minutes=0,
        )
        assert abs(result["est/total_cost"] - 1.045) < 0.01

    def test_predict_includes_setup_overhead(self) -> None:
        """Setup overhead adds to total cost."""
        from minivess.observability.infrastructure_timing import (
            estimate_cost_from_first_epoch,
        )

        result_no_setup = estimate_cost_from_first_epoch(
            epoch_seconds=60,
            max_epochs=10,
            num_folds=1,
            hourly_rate_usd=1.0,
            setup_minutes=0,
        )
        result_with_setup = estimate_cost_from_first_epoch(
            epoch_seconds=60,
            max_epochs=10,
            num_folds=1,
            hourly_rate_usd=1.0,
            setup_minutes=5,
        )
        assert result_with_setup["est/total_cost"] > result_no_setup["est/total_cost"]

    def test_predict_zero_for_local(self) -> None:
        """Local runs (hourly_rate=0) have zero cost."""
        from minivess.observability.infrastructure_timing import (
            estimate_cost_from_first_epoch,
        )

        result = estimate_cost_from_first_epoch(
            epoch_seconds=60,
            max_epochs=100,
            num_folds=3,
            hourly_rate_usd=0.0,
        )
        assert result["est/total_cost"] == 0.0

    def test_result_has_required_keys(self) -> None:
        """Result must include cost, duration, and per-epoch estimates."""
        from minivess.observability.infrastructure_timing import (
            estimate_cost_from_first_epoch,
        )

        result = estimate_cost_from_first_epoch(
            epoch_seconds=120,
            max_epochs=50,
            num_folds=3,
            hourly_rate_usd=0.19,
        )
        for key in (
            "est/total_cost",
            "est/total_hours",
            "est/cost_per_epoch",
            "est/epoch_seconds",
        ):
            assert key in result, f"Missing key: {key}"
