"""Tests for biostatistics LaTeX table generation (Phase 6, Task 6.1)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from minivess.pipeline.biostatistics_tables import generate_tables
from minivess.pipeline.biostatistics_types import (
    PairwiseResult,
    RankingResult,
    VarianceDecompositionResult,
)

if TYPE_CHECKING:
    from pathlib import Path


def _make_pairwise() -> list[PairwiseResult]:
    return [
        PairwiseResult(
            condition_a="dice_ce",
            condition_b="tversky",
            metric="val_dice",
            p_value=0.02,
            p_adjusted=0.04,
            correction_method="holm",
            significant=True,
            cohens_d=0.5,
            cliffs_delta=0.3,
            vda=0.65,
            bayesian_left=0.7,
            bayesian_rope=0.2,
            bayesian_right=0.1,
        ),
        PairwiseResult(
            condition_a="dice_ce",
            condition_b="cbdice",
            metric="val_dice",
            p_value=0.15,
            p_adjusted=0.30,
            correction_method="holm",
            significant=False,
            cohens_d=0.2,
            cliffs_delta=0.1,
            vda=0.55,
        ),
    ]


def _make_variance() -> list[VarianceDecompositionResult]:
    return [
        VarianceDecompositionResult(
            metric="val_dice",
            friedman_statistic=10.5,
            friedman_p=0.005,
            nemenyi_matrix=None,
            icc_value=0.85,
            icc_ci_lower=0.75,
            icc_ci_upper=0.92,
            icc_type="ICC2",
        ),
    ]


def _make_rankings() -> list[RankingResult]:
    return [
        RankingResult(
            metric="val_dice",
            condition_ranks={"dice_ce": 1.0, "tversky": 2.0, "cbdice": 3.0},
            mean_ranks={"dice_ce": 1.0, "tversky": 2.0, "cbdice": 3.0},
            cd_value=0.5,
        ),
    ]


class TestGenerateTables:
    def test_comparison_table_has_booktabs(self, tmp_path: Path) -> None:
        tables = generate_tables(
            pairwise=_make_pairwise(),
            variance=_make_variance(),
            rankings=_make_rankings(),
            output_dir=tmp_path / "tables",
        )
        assert len(tables) >= 1
        # Find the comparison table
        comparison = [t for t in tables if "comparison" in t.table_id]
        assert len(comparison) >= 1
        content = comparison[0].path.read_text(encoding="utf-8")
        assert "\\toprule" in content
        assert "\\bottomrule" in content

    def test_best_values_are_bold(self, tmp_path: Path) -> None:
        tables = generate_tables(
            pairwise=_make_pairwise(),
            variance=_make_variance(),
            rankings=_make_rankings(),
            output_dir=tmp_path / "tables",
        )
        # Effect sizes table should bold the largest effect
        effect_tables = [t for t in tables if "effect" in t.table_id]
        if effect_tables:
            content = effect_tables[0].path.read_text(encoding="utf-8")
            assert "\\textbf" in content

    def test_significance_markers_present(self, tmp_path: Path) -> None:
        tables = generate_tables(
            pairwise=_make_pairwise(),
            variance=_make_variance(),
            rankings=_make_rankings(),
            output_dir=tmp_path / "tables",
        )
        comparison = [t for t in tables if "comparison" in t.table_id]
        assert len(comparison) >= 1
        content = comparison[0].path.read_text(encoding="utf-8")
        # Significant results should have a marker
        assert "*" in content or "\\dag" in content or "$^{*}$" in content

    def test_tables_generate_expected_count(self, tmp_path: Path) -> None:
        tables = generate_tables(
            pairwise=_make_pairwise(),
            variance=_make_variance(),
            rankings=_make_rankings(),
            output_dir=tmp_path / "tables",
        )
        # Should produce at least: comparison, effect sizes, variance decomposition
        assert len(tables) >= 3

    def test_tables_are_latex_format(self, tmp_path: Path) -> None:
        tables = generate_tables(
            pairwise=_make_pairwise(),
            variance=_make_variance(),
            rankings=_make_rankings(),
            output_dir=tmp_path / "tables",
        )
        for t in tables:
            assert t.format == "latex"
            assert t.path.suffix == ".tex"
