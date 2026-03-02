"""Tests for cross-model comparison infrastructure (SAM-12).

Verifies that cross_model_comparison() can compare different model families
(e.g., DynUNet vs SAM3 variants) using the same metric comparison machinery.
"""

from __future__ import annotations

import pytest

from minivess.pipeline.comparison import (
    ComparisonTable,
    MetricSummary,
    cross_model_comparison,
    find_best_model,
    format_cross_model_markdown,
)


def _make_summary(mean: float, std: float = 0.01) -> MetricSummary:
    return MetricSummary(
        mean=mean,
        std=std,
        ci_lower=mean - 0.05,
        ci_upper=mean + 0.05,
        per_fold=[mean - 0.01, mean, mean + 0.01],
    )


def _make_model_results() -> dict[str, dict[str, MetricSummary]]:
    """Create synthetic results for 3 model families."""
    return {
        "dynunet_cbdice_cldice": {
            "dsc": _make_summary(0.824),
            "cldice": _make_summary(0.906),
            "hd95": _make_summary(3.2),
        },
        "sam3_vanilla": {
            "dsc": _make_summary(0.45),
            "cldice": _make_summary(0.38),
            "hd95": _make_summary(12.5),
        },
        "sam3_topolora": {
            "dsc": _make_summary(0.52),
            "cldice": _make_summary(0.55),
            "hd95": _make_summary(9.1),
        },
        "sam3_hybrid": {
            "dsc": _make_summary(0.61),
            "cldice": _make_summary(0.63),
            "hd95": _make_summary(7.3),
        },
    }


class TestCrossModelComparison:
    """cross_model_comparison() builds ComparisonTable from model family results."""

    def test_builds_table_from_model_results(self) -> None:
        results = _make_model_results()
        table = cross_model_comparison(results)
        assert isinstance(table, ComparisonTable)
        assert len(table.losses) == 4  # 4 model families
        assert "dsc" in table.metric_names

    def test_empty_input_returns_empty_table(self) -> None:
        table = cross_model_comparison({})
        assert table.losses == []
        assert table.metric_names == []

    def test_model_names_preserved(self) -> None:
        results = _make_model_results()
        table = cross_model_comparison(results)
        names = {lr.loss_name for lr in table.losses}
        assert "dynunet_cbdice_cldice" in names
        assert "sam3_vanilla" in names
        assert "sam3_topolora" in names
        assert "sam3_hybrid" in names

    def test_metric_values_correct(self) -> None:
        results = _make_model_results()
        table = cross_model_comparison(results)
        dynunet = next(
            lr for lr in table.losses if lr.loss_name == "dynunet_cbdice_cldice"
        )
        assert abs(dynunet.metrics["dsc"].mean - 0.824) < 1e-6


class TestFindBestModel:
    """find_best_model() identifies best model across families."""

    def test_finds_best_by_dsc(self) -> None:
        results = _make_model_results()
        table = cross_model_comparison(results)
        best = find_best_model(table, "dsc", higher_is_better=True)
        assert best == "dynunet_cbdice_cldice"

    def test_finds_best_by_hd95_lower_better(self) -> None:
        results = _make_model_results()
        table = cross_model_comparison(results)
        best = find_best_model(table, "hd95", higher_is_better=False)
        assert best == "dynunet_cbdice_cldice"

    def test_raises_on_empty_table(self) -> None:
        table = ComparisonTable(losses=[], metric_names=[])
        with pytest.raises(ValueError, match="no models"):
            find_best_model(table, "dsc")


class TestFormatCrossModelMarkdown:
    """format_cross_model_markdown() formats table with model names."""

    def test_contains_model_names(self) -> None:
        results = _make_model_results()
        table = cross_model_comparison(results)
        md = format_cross_model_markdown(table)
        assert "dynunet_cbdice_cldice" in md
        assert "sam3_vanilla" in md

    def test_empty_table(self) -> None:
        table = ComparisonTable(losses=[], metric_names=[])
        md = format_cross_model_markdown(table)
        assert "No results" in md
