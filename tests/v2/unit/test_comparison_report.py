"""Tests for extended comparison reporting (comparison_report.py).

RED phase tests -- written before implementation.
Tests cover ModelComparisonEntry, ExtendedComparisonTable, build_extended_comparison,
format_extended_markdown, compute_significance_matrix, format_significance_markdown,
find_best_model_overall, and edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from minivess.pipeline.ci import ConfidenceInterval
from minivess.pipeline.comparison_report import (
    ExtendedComparisonTable,
    ModelComparisonEntry,
    build_extended_comparison,
    compute_significance_matrix,
    find_best_model_overall,
    format_extended_markdown,
    format_significance_markdown,
)
from minivess.pipeline.evaluation import FoldResult
from minivess.pipeline.evaluation_runner import EvaluationResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ci(value: float, *, spread: float = 0.05) -> ConfidenceInterval:
    """Create a ConfidenceInterval centred on *value*."""
    return ConfidenceInterval(
        point_estimate=value,
        lower=value - spread,
        upper=value + spread,
        confidence_level=0.95,
        method="percentile",
    )


def _make_fold_result(
    dsc: float,
    cldsc: float,
    masd: float,
    *,
    n_volumes: int = 5,
) -> FoldResult:
    """Create a FoldResult with per-volume metrics (slightly noisy)."""
    rng = np.random.default_rng(42)
    per_volume: dict[str, list[float]] = {
        "dsc": (rng.normal(dsc, 0.02, n_volumes)).tolist(),
        "centreline_dsc": (rng.normal(cldsc, 0.02, n_volumes)).tolist(),
        "measured_masd": (np.abs(rng.normal(masd, 0.3, n_volumes))).tolist(),
    }
    aggregated: dict[str, ConfidenceInterval] = {
        "dsc": _make_ci(dsc),
        "centreline_dsc": _make_ci(cldsc),
        "measured_masd": _make_ci(masd, spread=0.5),
    }
    return FoldResult(per_volume_metrics=per_volume, aggregated=aggregated)


def _make_eval_result(
    model_name: str,
    dataset_name: str,
    subset_name: str,
    dsc: float,
    cldsc: float,
    masd: float,
) -> EvaluationResult:
    """Create an EvaluationResult with dummy metrics."""
    return EvaluationResult(
        model_name=model_name,
        dataset_name=dataset_name,
        subset_name=subset_name,
        fold_result=_make_fold_result(dsc, cldsc, masd),
        predictions_dir=None,
        uncertainty_maps_dir=None,
    )


def _build_all_results() -> dict[str, dict[str, dict[str, EvaluationResult]]]:
    """Build a sample all_results dict with 3 models and 1 dataset.

    Structure: model_name -> dataset_name -> subset_name -> EvaluationResult
    """
    return {
        "dice_ce_fold0": {
            "minivess": {
                "all": _make_eval_result(
                    "dice_ce_fold0", "minivess", "all", 0.82, 0.75, 2.5
                ),
            },
        },
        "cbdice_fold0": {
            "minivess": {
                "all": _make_eval_result(
                    "cbdice_fold0", "minivess", "all", 0.85, 0.80, 1.8
                ),
            },
        },
        "ensemble_mean": {
            "minivess": {
                "all": _make_eval_result(
                    "ensemble_mean", "minivess", "all", 0.88, 0.84, 1.2
                ),
            },
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestModelComparisonEntry:
    def test_dataclass_fields(self) -> None:
        """ModelComparisonEntry stores model_name, model_type, strategy, and metrics."""
        entry = ModelComparisonEntry(
            model_name="dice_ce_fold0",
            model_type="single",
            strategy=None,
            per_dataset_metrics={"minivess": {"dsc": 0.85}},
            overall_metrics={"dsc": 0.85},
        )
        assert entry.model_name == "dice_ce_fold0"
        assert entry.model_type == "single"
        assert entry.strategy is None
        assert entry.per_dataset_metrics["minivess"]["dsc"] == pytest.approx(0.85)
        assert entry.overall_metrics["dsc"] == pytest.approx(0.85)

    def test_ensemble_entry_has_strategy(self) -> None:
        """Ensemble entries carry a strategy name."""
        entry = ModelComparisonEntry(
            model_name="ensemble_mean",
            model_type="ensemble",
            strategy="mean",
            per_dataset_metrics={},
            overall_metrics={"dsc": 0.90},
        )
        assert entry.model_type == "ensemble"
        assert entry.strategy == "mean"


class TestExtendedComparisonTable:
    def test_dataclass_fields(self) -> None:
        """ExtendedComparisonTable stores entries, metric_names, dataset_names, primary_metric."""
        table = ExtendedComparisonTable(
            entries=[],
            metric_names=["dsc", "centreline_dsc"],
            dataset_names=["minivess"],
            primary_metric="dsc",
            primary_metric_direction="maximize",
        )
        assert table.entries == []
        assert table.metric_names == ["dsc", "centreline_dsc"]
        assert table.dataset_names == ["minivess"]
        assert table.primary_metric == "dsc"
        assert table.primary_metric_direction == "maximize"


class TestBuildExtendedComparison:
    def test_builds_from_all_results(self) -> None:
        """build_extended_comparison produces an ExtendedComparisonTable from results."""
        all_results = _build_all_results()
        table = build_extended_comparison(all_results)
        assert isinstance(table, ExtendedComparisonTable)
        assert len(table.entries) == 3
        model_names = [e.model_name for e in table.entries]
        assert "dice_ce_fold0" in model_names
        assert "cbdice_fold0" in model_names
        assert "ensemble_mean" in model_names

    def test_sorts_by_primary_metric(self) -> None:
        """Entries are sorted by the primary metric (descending for maximize)."""
        all_results = _build_all_results()
        table = build_extended_comparison(
            all_results,
            primary_metric="dsc",
            primary_metric_direction="maximize",
        )
        # The model with the highest DSC should be first
        values = [e.overall_metrics.get("dsc", 0.0) for e in table.entries]
        assert values == sorted(values, reverse=True)

    def test_sorts_ascending_for_minimize(self) -> None:
        """Entries are sorted ascending when primary_metric_direction=minimize."""
        all_results = _build_all_results()
        table = build_extended_comparison(
            all_results,
            primary_metric="measured_masd",
            primary_metric_direction="minimize",
        )
        values = [e.overall_metrics.get("measured_masd", 0.0) for e in table.entries]
        assert values == sorted(values)

    def test_overall_metrics_aggregated_across_datasets(self) -> None:
        """Overall metrics are the mean of per-dataset metrics."""
        # Make results with 2 datasets
        results: dict[str, dict[str, dict[str, EvaluationResult]]] = {
            "model_a": {
                "ds1": {
                    "all": _make_eval_result("model_a", "ds1", "all", 0.80, 0.70, 2.0),
                },
                "ds2": {
                    "all": _make_eval_result("model_a", "ds2", "all", 0.90, 0.80, 1.0),
                },
            },
        }
        table = build_extended_comparison(results)
        entry = table.entries[0]
        # Mean of 0.80 and 0.90 = 0.85
        assert entry.overall_metrics["dsc"] == pytest.approx(0.85, abs=0.01)

    def test_dataset_names_collected(self) -> None:
        """Table dataset_names includes all datasets seen."""
        results: dict[str, dict[str, dict[str, EvaluationResult]]] = {
            "model_a": {
                "ds1": {
                    "all": _make_eval_result("model_a", "ds1", "all", 0.80, 0.70, 2.0),
                },
                "ds2": {
                    "all": _make_eval_result("model_a", "ds2", "all", 0.90, 0.80, 1.0),
                },
            },
        }
        table = build_extended_comparison(results)
        assert "ds1" in table.dataset_names
        assert "ds2" in table.dataset_names

    def test_metric_names_collected(self) -> None:
        """Table metric_names includes all metrics from FoldResult aggregated."""
        all_results = _build_all_results()
        table = build_extended_comparison(all_results)
        assert "dsc" in table.metric_names
        assert "centreline_dsc" in table.metric_names
        assert "measured_masd" in table.metric_names


class TestFormatExtendedMarkdown:
    def test_overall_table_present(self) -> None:
        """Markdown includes an overall comparison table."""
        all_results = _build_all_results()
        table = build_extended_comparison(all_results)
        md = format_extended_markdown(table)
        assert "Overall" in md or "overall" in md.lower()
        assert "|" in md
        # All model names should appear
        assert "dice_ce_fold0" in md
        assert "ensemble_mean" in md

    def test_per_dataset_sections(self) -> None:
        """Markdown includes per-dataset breakdown when include_per_dataset=True."""
        all_results = _build_all_results()
        table = build_extended_comparison(all_results)
        md = format_extended_markdown(table, include_per_dataset=True)
        assert "minivess" in md.lower()

    def test_per_dataset_excluded(self) -> None:
        """Per-dataset sections are excluded when include_per_dataset=False."""
        all_results = _build_all_results()
        table = build_extended_comparison(all_results)
        md_with = format_extended_markdown(table, include_per_dataset=True)
        md_without = format_extended_markdown(table, include_per_dataset=False)
        # The version without per-dataset should be shorter
        assert len(md_without) < len(md_with)

    def test_empty_table_markdown(self) -> None:
        """Empty table produces a sensible message."""
        table = ExtendedComparisonTable(
            entries=[],
            metric_names=[],
            dataset_names=[],
            primary_metric="dsc",
            primary_metric_direction="maximize",
        )
        md = format_extended_markdown(table)
        assert isinstance(md, str)
        assert len(md) > 0


class TestComputeSignificanceMatrix:
    def test_returns_pvalue_dict(self) -> None:
        """compute_significance_matrix returns {(a, b): p_value} dict."""
        all_results = _build_all_results()
        table = build_extended_comparison(
            all_results,
            primary_metric="dsc",
            primary_metric_direction="maximize",
        )
        pvalues = compute_significance_matrix(table, n_resamples=200, seed=42)
        assert isinstance(pvalues, dict)
        # For 3 models, we expect C(3,2)=3 pairs
        assert len(pvalues) == 3
        for key, pval in pvalues.items():
            assert isinstance(key, tuple)
            assert len(key) == 2
            assert 0.0 <= pval <= 1.0

    def test_identical_models_high_pvalue(self) -> None:
        """Two identical models should have a high p-value (no significant difference)."""
        # Use two identical models
        results: dict[str, dict[str, dict[str, EvaluationResult]]] = {
            "model_a": {
                "minivess": {
                    "all": _make_eval_result(
                        "model_a", "minivess", "all", 0.85, 0.80, 1.5
                    ),
                },
            },
            "model_b": {
                "minivess": {
                    "all": _make_eval_result(
                        "model_b", "minivess", "all", 0.85, 0.80, 1.5
                    ),
                },
            },
        }
        table = build_extended_comparison(
            results,
            primary_metric="dsc",
            primary_metric_direction="maximize",
        )
        pvalues = compute_significance_matrix(table, n_resamples=500, seed=42)
        pair_key = ("model_a", "model_b")
        reverse_key = ("model_b", "model_a")
        key = pair_key if pair_key in pvalues else reverse_key
        assert pvalues[key] > 0.3  # Should not be significant


class TestFormatSignificanceMarkdown:
    def test_produces_table_with_stars(self) -> None:
        """format_significance_markdown produces a table with *, **, or ns."""
        pvalues = {
            ("model_a", "model_b"): 0.001,
            ("model_a", "model_c"): 0.03,
            ("model_b", "model_c"): 0.50,
        }
        model_names = ["model_a", "model_b", "model_c"]
        md = format_significance_markdown(pvalues, model_names)
        assert "**" in md  # p < 0.01
        assert "*" in md  # p < 0.05
        assert "ns" in md  # not significant

    def test_uses_custom_alpha(self) -> None:
        """Custom alpha changes what counts as significant."""
        pvalues = {("a", "b"): 0.08}
        # With alpha=0.10, p=0.08 is significant
        md = format_significance_markdown(pvalues, ["a", "b"], alpha=0.10)
        assert "ns" not in md or "*" in md  # Should show * not ns


class TestFindBestModelOverall:
    def test_finds_best_model(self) -> None:
        """find_best_model_overall returns the model with the best primary metric."""
        all_results = _build_all_results()
        table = build_extended_comparison(
            all_results,
            primary_metric="dsc",
            primary_metric_direction="maximize",
        )
        best = find_best_model_overall(table)
        # ensemble_mean has dsc=0.88 (highest)
        assert best == "ensemble_mean"

    def test_finds_best_minimize(self) -> None:
        """find_best_model_overall with minimize returns lowest metric model."""
        all_results = _build_all_results()
        table = build_extended_comparison(
            all_results,
            primary_metric="measured_masd",
            primary_metric_direction="minimize",
        )
        best = find_best_model_overall(table)
        # ensemble_mean has masd=1.2 (lowest)
        assert best == "ensemble_mean"


class TestEmptyResultsHandled:
    def test_empty_all_results(self) -> None:
        """Empty all_results produces an empty table without error."""
        table = build_extended_comparison({})
        assert isinstance(table, ExtendedComparisonTable)
        assert len(table.entries) == 0

    def test_empty_table_find_best_raises(self) -> None:
        """find_best_model_overall on empty table raises ValueError."""
        table = ExtendedComparisonTable(
            entries=[],
            metric_names=[],
            dataset_names=[],
            primary_metric="dsc",
            primary_metric_direction="maximize",
        )
        with pytest.raises(ValueError, match="[Nn]o entries|[Ee]mpty"):
            find_best_model_overall(table)

    def test_empty_table_significance_matrix(self) -> None:
        """compute_significance_matrix on empty table returns empty dict."""
        table = ExtendedComparisonTable(
            entries=[],
            metric_names=[],
            dataset_names=[],
            primary_metric="dsc",
            primary_metric_direction="maximize",
        )
        pvalues = compute_significance_matrix(table, n_resamples=100, seed=42)
        assert pvalues == {}
