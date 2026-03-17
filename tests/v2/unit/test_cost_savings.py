"""Tests for spot vs on-demand savings computation.

PR-E T2 (Issue #831): Aggregate cost data and compute spot savings,
per-model breakdown, and debug vs production cost split.

TDD Phase: RED — tests written before implementation.
"""

from __future__ import annotations

import pytest


def _make_cost_records() -> list[dict[str, float | str]]:
    """Create synthetic cost records from factorial runs."""
    return [
        {
            "run_id": "run_1",
            "model": "dynunet",
            "phase": "training",
            "gpu_hours": 2.0,
            "spot_cost_usd": 1.60,
            "ondemand_hourly_rate": 1.20,
        },
        {
            "run_id": "run_2",
            "model": "dynunet",
            "phase": "training",
            "gpu_hours": 1.5,
            "spot_cost_usd": 1.20,
            "ondemand_hourly_rate": 1.20,
        },
        {
            "run_id": "run_3",
            "model": "segresnet",
            "phase": "training",
            "gpu_hours": 3.0,
            "spot_cost_usd": 2.40,
            "ondemand_hourly_rate": 1.20,
        },
        {
            "run_id": "run_4",
            "model": "segresnet",
            "phase": "debug",
            "gpu_hours": 0.5,
            "spot_cost_usd": 0.40,
            "ondemand_hourly_rate": 1.20,
        },
    ]


class TestSpotSavingsComputation:
    """Compute total spot savings."""

    def test_spot_savings_computation(self) -> None:
        """Total spot cost is sum of all spot_cost_usd."""
        from minivess.observability.cost_logging import compute_spot_savings

        records = _make_cost_records()
        summary = compute_spot_savings(records)

        # Total spot: 1.60 + 1.20 + 2.40 + 0.40 = 5.60
        assert summary.total_spot_cost_usd == pytest.approx(5.60, abs=0.01)

    def test_spot_savings_ondemand_total(self) -> None:
        """On-demand cost = sum(hours * ondemand_hourly_rate)."""
        from minivess.observability.cost_logging import compute_spot_savings

        records = _make_cost_records()
        summary = compute_spot_savings(records)

        # On-demand: (2.0 + 1.5 + 3.0 + 0.5) * 1.20 = 8.40
        assert summary.total_ondemand_cost_usd == pytest.approx(8.40, abs=0.01)


class TestSpotSavingsPercentage:
    """Verify savings percentage calculation."""

    def test_spot_savings_percentage(self) -> None:
        """Savings = (ondemand - spot) / ondemand * 100."""
        from minivess.observability.cost_logging import compute_spot_savings

        records = _make_cost_records()
        summary = compute_spot_savings(records)

        # (8.40 - 5.60) / 8.40 * 100 = 33.33%
        assert summary.savings_pct == pytest.approx(33.33, abs=0.1)

    def test_zero_ondemand_no_division_error(self) -> None:
        """Zero on-demand cost produces 0% savings, not ZeroDivisionError."""
        from minivess.observability.cost_logging import compute_spot_savings

        records = [
            {
                "run_id": "run_1",
                "model": "test",
                "phase": "training",
                "gpu_hours": 0.0,
                "spot_cost_usd": 0.0,
                "ondemand_hourly_rate": 0.0,
            }
        ]
        summary = compute_spot_savings(records)
        assert summary.savings_pct == 0.0


class TestCostBreakdownByPhase:
    """Debug vs production cost breakdown."""

    def test_cost_breakdown_by_phase(self) -> None:
        """Costs are split between debug and training phases."""
        from minivess.observability.cost_logging import compute_spot_savings

        records = _make_cost_records()
        summary = compute_spot_savings(records)

        assert "debug" in summary.cost_by_phase
        assert "training" in summary.cost_by_phase
        assert summary.cost_by_phase["debug"] == pytest.approx(0.40, abs=0.01)
        assert summary.cost_by_phase["training"] == pytest.approx(5.20, abs=0.01)


class TestCostBreakdownByModel:
    """Per-model cost breakdown."""

    def test_cost_breakdown_by_model(self) -> None:
        """Costs are aggregated per model family."""
        from minivess.observability.cost_logging import compute_spot_savings

        records = _make_cost_records()
        summary = compute_spot_savings(records)

        assert "dynunet" in summary.cost_by_model
        assert "segresnet" in summary.cost_by_model
        # dynunet: 1.60 + 1.20 = 2.80
        assert summary.cost_by_model["dynunet"] == pytest.approx(2.80, abs=0.01)
        # segresnet: 2.40 + 0.40 = 2.80
        assert summary.cost_by_model["segresnet"] == pytest.approx(2.80, abs=0.01)

    def test_cost_by_model_sums_to_total(self) -> None:
        """Per-model costs sum to total spot cost."""
        from minivess.observability.cost_logging import compute_spot_savings

        records = _make_cost_records()
        summary = compute_spot_savings(records)

        model_sum = sum(summary.cost_by_model.values())
        assert model_sum == pytest.approx(summary.total_spot_cost_usd, abs=0.01)


class TestCostSummaryDataclass:
    """CostSummary dataclass fields."""

    def test_cost_summary_dataclass(self) -> None:
        """CostSummary has all required fields."""
        from minivess.observability.cost_logging import CostSummary

        summary = CostSummary(
            total_spot_cost_usd=5.60,
            total_ondemand_cost_usd=8.40,
            savings_pct=33.33,
            cost_by_phase={"training": 5.20, "debug": 0.40},
            cost_by_model={"dynunet": 2.80, "segresnet": 2.80},
            total_gpu_hours=7.0,
        )

        assert summary.total_spot_cost_usd == 5.60
        assert summary.total_gpu_hours == 7.0
        assert len(summary.cost_by_model) == 2
