"""Tests for cost estimate edge cases.

T8 from double-check plan: zero epochs, zero rate, zero folds, invalid env.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


class TestEstimateCostEdgeCases:
    """estimate_cost_from_first_epoch must return finite values for all edge cases."""

    def test_zero_epoch_seconds(self) -> None:
        from minivess.observability.infrastructure_timing import (
            estimate_cost_from_first_epoch,
        )

        result = estimate_cost_from_first_epoch(
            epoch_seconds=0.0, max_epochs=100, num_folds=3, hourly_rate_usd=0.55
        )
        assert all(math.isfinite(v) for v in result.values()), f"Non-finite: {result}"
        assert result["est/epoch_seconds"] == 0.0

    def test_zero_hourly_rate(self) -> None:
        from minivess.observability.infrastructure_timing import (
            estimate_cost_from_first_epoch,
        )

        result = estimate_cost_from_first_epoch(
            epoch_seconds=120.0, max_epochs=100, num_folds=3, hourly_rate_usd=0.0
        )
        assert result["est/total_cost"] == 0.0
        assert result["est/cost_per_epoch"] == 0.0
        assert result["est/total_hours"] > 0  # Still has time, just $0

    def test_zero_folds(self) -> None:
        from minivess.observability.infrastructure_timing import (
            estimate_cost_from_first_epoch,
        )

        result = estimate_cost_from_first_epoch(
            epoch_seconds=120.0, max_epochs=100, num_folds=0, hourly_rate_usd=0.55
        )
        assert all(math.isfinite(v) for v in result.values()), f"Non-finite: {result}"

    def test_normal_case(self) -> None:
        from minivess.observability.infrastructure_timing import (
            estimate_cost_from_first_epoch,
        )

        result = estimate_cost_from_first_epoch(
            epoch_seconds=60.0, max_epochs=100, num_folds=3, hourly_rate_usd=0.55
        )
        assert result["est/total_cost"] > 0
        assert result["est/total_hours"] > 0
        assert result["est/cost_per_epoch"] > 0
        assert all(math.isfinite(v) for v in result.values())


class TestGetHourlyRate:
    """get_hourly_rate_usd must handle invalid env gracefully."""

    def test_default_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from minivess.observability.infrastructure_timing import get_hourly_rate_usd

        monkeypatch.delenv("INSTANCE_HOURLY_USD", raising=False)
        assert get_hourly_rate_usd() == 0.0

    def test_valid_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from minivess.observability.infrastructure_timing import get_hourly_rate_usd

        monkeypatch.setenv("INSTANCE_HOURLY_USD", "1.23")
        assert get_hourly_rate_usd() == 1.23

    def test_invalid_value_warns(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from minivess.observability.infrastructure_timing import get_hourly_rate_usd

        monkeypatch.setenv("INSTANCE_HOURLY_USD", "not-a-number")
        with caplog.at_level(logging.WARNING):
            rate = get_hourly_rate_usd()
        assert rate == 0.0
        assert any("invalid" in r.message.lower() for r in caplog.records)
