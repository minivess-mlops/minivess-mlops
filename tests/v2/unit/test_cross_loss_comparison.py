"""Tests for cross-loss comparison with paired bootstrap testing.

RED phase tests — written before implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from minivess.pipeline.ci import ConfidenceInterval
from minivess.pipeline.comparison import (
    ComparisonTable,
    LossResult,
    build_comparison_table,
    find_best_loss,
    format_comparison_markdown,
    paired_bootstrap_test,
)
from minivess.pipeline.evaluation import FoldResult


def _make_fold_result(dsc: float, cldsc: float, masd: float) -> FoldResult:
    """Create a minimal FoldResult for testing.

    Uses the actual FoldResult API: per_volume_metrics is dict[str, list[float]].
    """
    return FoldResult(
        per_volume_metrics={
            "dsc": [dsc],
            "centreline_dsc": [cldsc],
            "measured_masd": [masd],
        },
        aggregated={
            "dsc": ConfidenceInterval(
                point_estimate=dsc,
                lower=dsc - 0.05,
                upper=dsc + 0.05,
                confidence_level=0.95,
                method="percentile_bootstrap",
            ),
            "centreline_dsc": ConfidenceInterval(
                point_estimate=cldsc,
                lower=cldsc - 0.05,
                upper=cldsc + 0.05,
                confidence_level=0.95,
                method="percentile_bootstrap",
            ),
            "measured_masd": ConfidenceInterval(
                point_estimate=masd,
                lower=masd - 0.5,
                upper=masd + 0.5,
                confidence_level=0.95,
                method="percentile_bootstrap",
            ),
        },
    )


class TestComparisonTable:
    def test_build_from_eval_results(self) -> None:
        """Build comparison table from eval results dict."""
        eval_results: dict[str, list[FoldResult]] = {
            "dice_ce": [
                _make_fold_result(0.85, 0.80, 1.2),
                _make_fold_result(0.83, 0.78, 1.4),
            ],
            "cbdice": [
                _make_fold_result(0.87, 0.82, 1.1),
                _make_fold_result(0.86, 0.81, 1.0),
            ],
        }
        table = build_comparison_table(eval_results)
        assert isinstance(table, ComparisonTable)
        assert len(table.losses) == 2
        loss_names = [lr.loss_name for lr in table.losses]
        assert "dice_ce" in loss_names
        assert "cbdice" in loss_names

    def test_loss_result_has_metrics(self) -> None:
        """Each LossResult contains metric names with mean and CI."""
        eval_results: dict[str, list[FoldResult]] = {
            "dice_ce": [_make_fold_result(0.85, 0.80, 1.2)],
        }
        table = build_comparison_table(eval_results)
        result = table.losses[0]
        assert isinstance(result, LossResult)
        assert "dsc" in result.metrics
        assert result.metrics["dsc"].mean == pytest.approx(0.85, abs=0.01)

    def test_empty_results_handled(self) -> None:
        """Empty eval_results returns empty table."""
        table = build_comparison_table({})
        assert len(table.losses) == 0

    def test_num_folds_is_correct(self) -> None:
        """LossResult records the correct number of folds."""
        eval_results: dict[str, list[FoldResult]] = {
            "dice_ce": [
                _make_fold_result(0.85, 0.80, 1.2),
                _make_fold_result(0.83, 0.78, 1.4),
                _make_fold_result(0.84, 0.79, 1.3),
            ],
        }
        table = build_comparison_table(eval_results)
        assert table.losses[0].num_folds == 3

    def test_metric_names_populated(self) -> None:
        """ComparisonTable.metric_names contains all metric keys."""
        eval_results: dict[str, list[FoldResult]] = {
            "dice_ce": [_make_fold_result(0.85, 0.80, 1.2)],
        }
        table = build_comparison_table(eval_results)
        assert "dsc" in table.metric_names
        assert "centreline_dsc" in table.metric_names
        assert "measured_masd" in table.metric_names

    def test_multi_fold_mean_is_averaged(self) -> None:
        """Mean across folds is the average of per-fold point estimates."""
        eval_results: dict[str, list[FoldResult]] = {
            "dice_ce": [
                _make_fold_result(0.80, 0.75, 1.0),
                _make_fold_result(0.90, 0.85, 2.0),
            ],
        }
        table = build_comparison_table(eval_results)
        result = table.losses[0]
        assert result.metrics["dsc"].mean == pytest.approx(0.85, abs=0.01)


class TestBestLoss:
    def test_find_best_by_dsc(self) -> None:
        """Highest DSC wins (higher_is_better=True)."""
        eval_results: dict[str, list[FoldResult]] = {
            "dice_ce": [_make_fold_result(0.85, 0.80, 1.2)],
            "cbdice": [_make_fold_result(0.90, 0.85, 1.0)],
        }
        table = build_comparison_table(eval_results)
        best = find_best_loss(table, metric="dsc", higher_is_better=True)
        assert best == "cbdice"

    def test_find_best_by_masd(self) -> None:
        """Lowest MASD wins (higher_is_better=False)."""
        eval_results: dict[str, list[FoldResult]] = {
            "dice_ce": [_make_fold_result(0.85, 0.80, 1.2)],
            "cbdice": [_make_fold_result(0.90, 0.85, 2.0)],
        }
        table = build_comparison_table(eval_results)
        best = find_best_loss(table, metric="measured_masd", higher_is_better=False)
        assert best == "dice_ce"

    def test_find_best_single_loss(self) -> None:
        """Single loss is always best by any metric."""
        eval_results: dict[str, list[FoldResult]] = {
            "dice_ce": [_make_fold_result(0.85, 0.80, 1.2)],
        }
        table = build_comparison_table(eval_results)
        best = find_best_loss(table, metric="dsc", higher_is_better=True)
        assert best == "dice_ce"

    def test_find_best_tie_broken_by_first(self) -> None:
        """Ties: returns first loss encountered with that value."""
        eval_results: dict[str, list[FoldResult]] = {
            "dice_ce": [_make_fold_result(0.85, 0.80, 1.2)],
            "cbdice": [_make_fold_result(0.85, 0.80, 1.2)],
        }
        table = build_comparison_table(eval_results)
        best = find_best_loss(table, metric="dsc", higher_is_better=True)
        # Any of the tied losses is valid; just ensure it's one of them
        assert best in {"dice_ce", "cbdice"}


class TestPairedBootstrap:
    def test_paired_bootstrap_returns_pvalue(self) -> None:
        """Paired bootstrap test returns p-value between 0 and 1."""
        scores_a = np.array([0.85, 0.83, 0.87, 0.84, 0.86])
        scores_b = np.array([0.90, 0.88, 0.91, 0.89, 0.92])
        p_value = paired_bootstrap_test(scores_a, scores_b, n_resamples=1000, seed=42)
        assert 0.0 <= p_value <= 1.0

    def test_identical_scores_high_pvalue(self) -> None:
        """Identical scores -> p-value close to 1 (no significant difference)."""
        scores = np.array([0.85, 0.83, 0.87, 0.84, 0.86])
        p_value = paired_bootstrap_test(scores, scores, n_resamples=1000, seed=42)
        assert p_value > 0.5

    def test_very_different_scores_low_pvalue(self) -> None:
        """Very different scores -> p-value close to 0 (highly significant)."""
        scores_a = np.array([0.10, 0.12, 0.11, 0.09, 0.13])
        scores_b = np.array([0.90, 0.92, 0.91, 0.89, 0.93])
        p_value = paired_bootstrap_test(scores_a, scores_b, n_resamples=1000, seed=42)
        assert p_value < 0.05

    def test_reproducible_with_same_seed(self) -> None:
        """Same seed produces identical results."""
        scores_a = np.array([0.85, 0.83, 0.87, 0.84, 0.86])
        scores_b = np.array([0.88, 0.86, 0.89, 0.87, 0.90])
        p1 = paired_bootstrap_test(scores_a, scores_b, n_resamples=500, seed=99)
        p2 = paired_bootstrap_test(scores_a, scores_b, n_resamples=500, seed=99)
        assert p1 == pytest.approx(p2)

    def test_length_mismatch_raises(self) -> None:
        """Mismatched array lengths raise ValueError."""
        scores_a = np.array([0.85, 0.83, 0.87])
        scores_b = np.array([0.90, 0.88])
        with pytest.raises(ValueError, match="same length"):
            paired_bootstrap_test(scores_a, scores_b)


class TestFormatMarkdown:
    def test_format_produces_table(self) -> None:
        """format_comparison_markdown produces a markdown table with pipe chars."""
        eval_results: dict[str, list[FoldResult]] = {
            "dice_ce": [_make_fold_result(0.85, 0.80, 1.2)],
            "cbdice": [_make_fold_result(0.90, 0.85, 1.0)],
        }
        table = build_comparison_table(eval_results)
        md = format_comparison_markdown(table)
        assert "dice_ce" in md
        assert "cbdice" in md
        assert "|" in md  # markdown table formatting

    def test_format_empty_table(self) -> None:
        """format_comparison_markdown on empty table returns a string."""
        table = build_comparison_table({})
        md = format_comparison_markdown(table)
        assert isinstance(md, str)

    def test_format_contains_metric_names(self) -> None:
        """Markdown includes metric names in header."""
        eval_results: dict[str, list[FoldResult]] = {
            "dice_ce": [_make_fold_result(0.85, 0.80, 1.2)],
        }
        table = build_comparison_table(eval_results)
        md = format_comparison_markdown(table)
        assert "dsc" in md
