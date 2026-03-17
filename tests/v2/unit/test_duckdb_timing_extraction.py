"""Tests for DuckDB extraction of infrastructure timing and cost data.

Verifies that setup_* and cost_* prefixes are in _EXCLUDED_METRIC_PREFIXES
so they don't pollute the training_metrics table.

Issue: #683
"""

from __future__ import annotations


class TestExcludedMetricPrefixes:
    """Tests for _EXCLUDED_METRIC_PREFIXES in duckdb_extraction.py."""

    def test_excluded_metric_prefixes_include_setup(self) -> None:
        """_EXCLUDED_METRIC_PREFIXES contains 'setup_'."""
        from minivess.pipeline.duckdb_extraction import _EXCLUDED_METRIC_PREFIXES

        assert any(prefix == "setup_" for prefix in _EXCLUDED_METRIC_PREFIXES), (
            f"'setup_' not found in {_EXCLUDED_METRIC_PREFIXES}"
        )

    def test_excluded_metric_prefixes_include_cost(self) -> None:
        """_EXCLUDED_METRIC_PREFIXES contains 'cost_'."""
        from minivess.pipeline.duckdb_extraction import _EXCLUDED_METRIC_PREFIXES

        assert any(prefix == "cost_" for prefix in _EXCLUDED_METRIC_PREFIXES), (
            f"'cost_' not found in {_EXCLUDED_METRIC_PREFIXES}"
        )

    def test_should_exclude_setup_metric(self) -> None:
        """setup_total_seconds is excluded from training metrics."""
        from minivess.pipeline.duckdb_extraction import _should_include_training_metric

        assert not _should_include_training_metric("setup_total_seconds")

    def test_should_exclude_cost_metric(self) -> None:
        """cost_total_usd is excluded from training metrics."""
        from minivess.pipeline.duckdb_extraction import _should_include_training_metric

        assert not _should_include_training_metric("cost/total_usd")

    def test_should_include_training_metric(self) -> None:
        """train_loss is still included in training metrics."""
        from minivess.pipeline.duckdb_extraction import _should_include_training_metric

        assert _should_include_training_metric("train_loss")
