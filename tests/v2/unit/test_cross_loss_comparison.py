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
    PairwiseComparison,
    build_comparison_table,
    cohens_d,
    compute_all_pairwise_comparisons,
    find_best_loss,
    format_comparison_markdown,
    format_significance_matrix_markdown,
    holm_bonferroni_correction,
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


class TestHolmBonferroni:
    def test_single_pvalue_not_significant(self) -> None:
        """Single p-value above alpha is not significant."""
        result = holm_bonferroni_correction([0.10], alpha=0.05)
        assert len(result) == 1
        adj_p, is_sig = result[0]
        assert not is_sig
        assert adj_p == pytest.approx(0.10, abs=1e-9)

    def test_single_pvalue_significant(self) -> None:
        """Single p-value below alpha is significant."""
        result = holm_bonferroni_correction([0.02], alpha=0.05)
        assert len(result) == 1
        adj_p, is_sig = result[0]
        assert is_sig
        assert adj_p == pytest.approx(0.02, abs=1e-9)

    def test_multiple_pvalues_ordered(self) -> None:
        """With three tests, smallest p-value significant, largest not."""
        # 3 tests: alpha thresholds are 0.05/3, 0.05/2, 0.05/1
        p_values = [0.001, 0.03, 0.20]
        result = holm_bonferroni_correction(p_values, alpha=0.05)
        assert len(result) == 3
        # The p_values list is returned in the original order
        # p=0.001 should be significant (smallest, well below 0.05/3 ≈ 0.0167)
        # p=0.03 should be significant (0.03 < 0.05/2 = 0.025)? No: 0.03 > 0.025, not sig
        # Actually: sorted: [0.001, 0.03, 0.20]
        # i=0: compare 0.001 with 0.05/3=0.0167 -> reject
        # i=1: compare 0.03 with 0.05/2=0.025 -> fail (0.03 > 0.025), stop
        # i=2: not rejected (stopped)
        # p=0.001 -> significant
        adj_p_001, is_sig_001 = result[p_values.index(0.001)]
        adj_p_020, is_sig_020 = result[p_values.index(0.20)]
        assert is_sig_001
        assert not is_sig_020

    def test_all_significant(self) -> None:
        """All very small p-values are all significant after correction."""
        p_values = [0.001, 0.002, 0.003]
        result = holm_bonferroni_correction(p_values, alpha=0.05)
        assert all(is_sig for _, is_sig in result)

    def test_none_significant(self) -> None:
        """Large p-values are all non-significant after correction."""
        p_values = [0.3, 0.5, 0.8]
        result = holm_bonferroni_correction(p_values, alpha=0.05)
        assert not any(is_sig for _, is_sig in result)

    def test_adjusted_pvalues_clamped_to_one(self) -> None:
        """Adjusted p-values never exceed 1.0."""
        p_values = [0.9, 0.95, 0.99]
        result = holm_bonferroni_correction(p_values, alpha=0.05)
        for adj_p, _ in result:
            assert adj_p <= 1.0


class TestCohensD:
    def test_identical_scores_zero_d(self) -> None:
        """Identical arrays produce Cohen's d of 0."""
        scores = np.array([0.80, 0.82, 0.85, 0.78, 0.83])
        d = cohens_d(scores, scores)
        assert d == pytest.approx(0.0, abs=1e-9)

    def test_large_difference_large_d(self) -> None:
        """Clearly separated distributions produce large |d|."""
        scores_a = np.array([0.90, 0.91, 0.89, 0.92, 0.88])
        scores_b = np.array([0.10, 0.11, 0.09, 0.12, 0.08])
        d = cohens_d(scores_a, scores_b)
        assert abs(d) > 5.0

    def test_small_difference_small_d(self) -> None:
        """Slightly different distributions produce small |d|."""
        rng = np.random.default_rng(0)
        scores_a = rng.normal(0.80, 0.05, 100)
        scores_b = rng.normal(0.81, 0.05, 100)
        d = cohens_d(scores_a, scores_b)
        assert abs(d) < 1.0

    def test_zero_variance_returns_zero(self) -> None:
        """Constant arrays (zero variance) return d=0.0 without error."""
        scores_a = np.array([0.5, 0.5, 0.5])
        scores_b = np.array([0.5, 0.5, 0.5])
        d = cohens_d(scores_a, scores_b)
        assert d == pytest.approx(0.0, abs=1e-9)

    def test_sign_reflects_direction(self) -> None:
        """d > 0 when mean_a > mean_b, d < 0 when mean_a < mean_b."""
        a = np.array([0.9, 0.85, 0.88])
        b = np.array([0.7, 0.72, 0.68])
        assert cohens_d(a, b) > 0
        assert cohens_d(b, a) < 0


class TestPairwiseComparison:
    def test_dataclass_fields(self) -> None:
        """PairwiseComparison has required fields."""
        pc = PairwiseComparison(
            loss_a="dice_ce",
            loss_b="cbdice",
            metric="dsc",
            p_value=0.03,
            adjusted_p_value=0.06,
            is_significant=False,
            effect_size=1.2,
            direction="A > B",
        )
        assert pc.loss_a == "dice_ce"
        assert pc.loss_b == "cbdice"
        assert pc.metric == "dsc"
        assert pc.p_value == pytest.approx(0.03)
        assert pc.adjusted_p_value == pytest.approx(0.06)
        assert not pc.is_significant
        assert pc.effect_size == pytest.approx(1.2)
        assert pc.direction == "A > B"

    def test_compute_all_pairwise_4_losses_gives_6_pairs(self) -> None:
        """C(4, 2) = 6 pairwise comparisons for 4 losses."""
        eval_results: dict[str, list[FoldResult]] = {
            "loss_a": [
                _make_fold_result(0.80, 0.75, 1.5),
                _make_fold_result(0.82, 0.77, 1.4),
                _make_fold_result(0.81, 0.76, 1.6),
            ],
            "loss_b": [
                _make_fold_result(0.85, 0.80, 1.2),
                _make_fold_result(0.83, 0.78, 1.3),
                _make_fold_result(0.84, 0.79, 1.1),
            ],
            "loss_c": [
                _make_fold_result(0.78, 0.73, 1.8),
                _make_fold_result(0.79, 0.74, 1.7),
                _make_fold_result(0.77, 0.72, 1.9),
            ],
            "loss_d": [
                _make_fold_result(0.88, 0.83, 1.0),
                _make_fold_result(0.87, 0.82, 1.1),
                _make_fold_result(0.89, 0.84, 0.9),
            ],
        }
        table = build_comparison_table(eval_results)
        comparisons = compute_all_pairwise_comparisons(table, metric="dsc")
        assert len(comparisons) == 6

    def test_adjusted_pvalues_present(self) -> None:
        """All PairwiseComparison objects have a non-negative adjusted_p_value."""
        eval_results: dict[str, list[FoldResult]] = {
            "loss_a": [
                _make_fold_result(0.80, 0.75, 1.5),
                _make_fold_result(0.82, 0.77, 1.4),
            ],
            "loss_b": [
                _make_fold_result(0.85, 0.80, 1.2),
                _make_fold_result(0.83, 0.78, 1.3),
            ],
        }
        table = build_comparison_table(eval_results)
        comparisons = compute_all_pairwise_comparisons(table, metric="dsc")
        for cmp in comparisons:
            assert 0.0 <= cmp.adjusted_p_value <= 1.0

    def test_effect_sizes_present(self) -> None:
        """All PairwiseComparison objects have a finite effect_size."""
        eval_results: dict[str, list[FoldResult]] = {
            "loss_a": [
                _make_fold_result(0.80, 0.75, 1.5),
                _make_fold_result(0.82, 0.77, 1.4),
            ],
            "loss_b": [
                _make_fold_result(0.85, 0.80, 1.2),
                _make_fold_result(0.83, 0.78, 1.3),
            ],
        }
        table = build_comparison_table(eval_results)
        comparisons = compute_all_pairwise_comparisons(table, metric="dsc")
        for cmp in comparisons:
            assert np.isfinite(cmp.effect_size)

    def test_direction_field_populated(self) -> None:
        """direction field is one of the three valid strings."""
        eval_results: dict[str, list[FoldResult]] = {
            "loss_a": [
                _make_fold_result(0.80, 0.75, 1.5),
                _make_fold_result(0.82, 0.77, 1.4),
            ],
            "loss_b": [
                _make_fold_result(0.85, 0.80, 1.2),
                _make_fold_result(0.83, 0.78, 1.3),
            ],
        }
        table = build_comparison_table(eval_results)
        comparisons = compute_all_pairwise_comparisons(table, metric="dsc")
        valid_directions = {"A > B", "A < B", "A \u2248 B"}
        for cmp in comparisons:
            assert cmp.direction in valid_directions


class TestSignificanceMatrixMarkdown:
    def _make_table_with_two_losses(self) -> ComparisonTable:
        eval_results: dict[str, list[FoldResult]] = {
            "loss_a": [
                _make_fold_result(0.80, 0.75, 1.5),
                _make_fold_result(0.82, 0.77, 1.4),
                _make_fold_result(0.81, 0.76, 1.6),
            ],
            "loss_b": [
                _make_fold_result(0.85, 0.80, 1.2),
                _make_fold_result(0.83, 0.78, 1.3),
                _make_fold_result(0.84, 0.79, 1.1),
            ],
        }
        return build_comparison_table(eval_results)

    def test_format_produces_table(self) -> None:
        """format_significance_matrix_markdown returns a string with pipe chars."""
        table = self._make_table_with_two_losses()
        comparisons = compute_all_pairwise_comparisons(table, metric="dsc")
        md = format_significance_matrix_markdown(comparisons)
        assert isinstance(md, str)
        assert "|" in md

    def test_stars_for_significant_pairs(self) -> None:
        """Significant pairs show a star (*) marker in the output."""
        # Construct very different scores so comparison will be significant
        eval_results: dict[str, list[FoldResult]] = {
            "loss_low": [
                _make_fold_result(0.10, 0.09, 5.0),
                _make_fold_result(0.11, 0.10, 4.8),
                _make_fold_result(0.09, 0.08, 5.2),
            ],
            "loss_high": [
                _make_fold_result(0.95, 0.90, 0.1),
                _make_fold_result(0.94, 0.89, 0.2),
                _make_fold_result(0.96, 0.91, 0.15),
            ],
        }
        table = build_comparison_table(eval_results)
        comparisons = compute_all_pairwise_comparisons(
            table, metric="dsc", n_resamples=500, seed=42
        )
        md = format_significance_matrix_markdown(comparisons)
        # At least one comparison should be significant and marked with *
        assert "*" in md
