"""Tests for RunAnalytics cost aggregation methods (issue #735).

Verifies cost_by_model_family(), cost_trends(), and break_even_analysis()
using in-memory DataFrames (no live MLflow required).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest


@pytest.fixture()
def runs_df() -> pd.DataFrame:
    """Create a DataFrame mimicking load_experiment_runs() output."""
    now = datetime.now(UTC)
    return pd.DataFrame(
        [
            {
                "run_id": "r1",
                "run_name": "fold_0_dynunet",
                "status": "FINISHED",
                "start_time": now.timestamp() - 3600,
                "end_time": now.timestamp() - 3000,
                "param_model_family": "dynunet",
                "metric_cost_total_usd": 0.50,
                "metric_cost_effective_gpu_rate": 0.22,
                "metric_cost_setup_fraction": 0.15,
                "metric_cost_break_even_epochs": 3,
                "metric_cost_epochs_to_amortize_setup": 8,
            },
            {
                "run_id": "r2",
                "run_name": "fold_0_dynunet_2",
                "status": "FINISHED",
                "start_time": now.timestamp() - 2500,
                "end_time": now.timestamp() - 2000,
                "param_model_family": "dynunet",
                "metric_cost_total_usd": 0.60,
                "metric_cost_effective_gpu_rate": 0.24,
                "metric_cost_setup_fraction": 0.12,
                "metric_cost_break_even_epochs": 2,
                "metric_cost_epochs_to_amortize_setup": 7,
            },
            {
                "run_id": "r3",
                "run_name": "fold_0_sam3",
                "status": "FINISHED",
                "start_time": now.timestamp() - 1500,
                "end_time": now.timestamp() - 1000,
                "param_model_family": "sam3_vanilla",
                "metric_cost_total_usd": 1.20,
                "metric_cost_effective_gpu_rate": 0.40,
                "metric_cost_setup_fraction": 0.08,
                "metric_cost_break_even_epochs": 5,
                "metric_cost_epochs_to_amortize_setup": 12,
            },
            {
                "run_id": "r4",
                "run_name": "fold_0_vesselfm",
                "status": "FINISHED",
                "start_time": now.timestamp() - 500,
                "end_time": now.timestamp(),
                "param_model_family": "vesselfm",
                "metric_cost_total_usd": 0.80,
                "metric_cost_effective_gpu_rate": 0.30,
                "metric_cost_setup_fraction": 0.10,
                "metric_cost_break_even_epochs": 4,
                "metric_cost_epochs_to_amortize_setup": 10,
            },
        ]
    )


@pytest.fixture()
def analytics() -> object:
    """Create a RunAnalytics instance with mocked MLflow client."""
    from unittest.mock import patch

    with (
        patch("minivess.observability.analytics.MlflowClient"),
        patch(
            "minivess.observability.analytics.resolve_tracking_uri",
            return_value="mlruns",
        ),
    ):
        from minivess.observability.analytics import RunAnalytics

        return RunAnalytics(tracking_uri="mlruns")


class TestCostByModelFamily:
    """cost_by_model_family aggregates cost metrics by model family."""

    def test_cost_by_model_family(
        self, analytics: object, runs_df: pd.DataFrame
    ) -> None:
        result = analytics.cost_by_model_family(runs_df)  # type: ignore[attr-defined]
        assert "model_family" in result.columns
        assert "total_cost_usd_mean" in result.columns
        assert "total_cost_usd_std" in result.columns
        assert "effective_rate_mean" in result.columns
        # dynunet has 2 runs, sam3_vanilla 1, vesselfm 1 -> 3 families
        assert len(result) == 3

    def test_cost_by_model_family_empty(self, analytics: object) -> None:
        empty_df = pd.DataFrame()
        result = analytics.cost_by_model_family(empty_df)  # type: ignore[attr-defined]
        assert len(result) == 0

    def test_cost_by_model_family_multiple_families(
        self, analytics: object, runs_df: pd.DataFrame
    ) -> None:
        result = analytics.cost_by_model_family(runs_df)  # type: ignore[attr-defined]
        families = set(result["model_family"].tolist())
        assert families == {"dynunet", "sam3_vanilla", "vesselfm"}


class TestCostTrends:
    """cost_trends computes cumulative cost over time."""

    def test_cost_trends(self, analytics: object, runs_df: pd.DataFrame) -> None:
        result = analytics.cost_trends(runs_df)  # type: ignore[attr-defined]
        assert "run_id" in result.columns
        assert "start_time" in result.columns
        assert "metric_cost_total_usd" in result.columns
        assert "cumulative_cost_usd" in result.columns
        # Should be ordered by start_time and cumulative increases
        cumulative = result["cumulative_cost_usd"].tolist()
        for i in range(1, len(cumulative)):
            assert cumulative[i] >= cumulative[i - 1]


class TestBreakEvenAnalysis:
    """break_even_analysis aggregates amortization metrics."""

    def test_break_even_analysis(
        self, analytics: object, runs_df: pd.DataFrame
    ) -> None:
        result = analytics.break_even_analysis(runs_df)  # type: ignore[attr-defined]
        assert "model_family" in result.columns
        assert "avg_break_even_epochs" in result.columns
        assert "avg_epochs_to_amortize" in result.columns
        assert len(result) == 3
