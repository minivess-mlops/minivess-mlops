"""Tests for biostatistics figure generation (Phase 5, Tasks 5.1-5.2)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

from minivess.pipeline.biostatistics_figures import (
    generate_figures,
    write_sidecar,
)
from minivess.pipeline.biostatistics_types import (
    FactorialAnovaResult,
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
        ),
    ]


def _make_variance() -> list[VarianceDecompositionResult]:
    return [
        VarianceDecompositionResult(
            metric="val_dice",
            friedman_statistic=10.5,
            friedman_p=0.005,
            nemenyi_matrix={
                "dice_ce": {"dice_ce": 1.0, "tversky": 0.04},
                "tversky": {"dice_ce": 0.04, "tversky": 1.0},
            },
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
            condition_ranks={"dice_ce": 1.0, "tversky": 2.0},
            mean_ranks={"dice_ce": 1.0, "tversky": 2.0},
            cd_value=0.5,
        ),
    ]


def _make_per_volume_data() -> dict[str, dict[str, dict[int, np.ndarray]]]:
    rng = np.random.default_rng(42)
    return {
        "val_dice": {
            "dice_ce": {0: rng.normal(0.82, 0.05, 20)},
            "tversky": {0: rng.normal(0.78, 0.05, 20)},
        },
    }


class TestGenerateFigures:
    def test_figure_generates_png_and_svg(self, tmp_path: Path) -> None:
        figures = generate_figures(
            per_volume_data=_make_per_volume_data(),
            pairwise=_make_pairwise(),
            variance=_make_variance(),
            rankings=_make_rankings(),
            output_dir=tmp_path / "figures",
        )
        assert len(figures) >= 1
        for fig in figures:
            # At least PNG should exist
            png_paths = [p for p in fig.paths if str(p).endswith(".png")]
            assert len(png_paths) >= 1
            for p in png_paths:
                assert p.exists()

    def test_json_sidecar_has_required_fields(self, tmp_path: Path) -> None:
        figures = generate_figures(
            per_volume_data=_make_per_volume_data(),
            pairwise=_make_pairwise(),
            variance=_make_variance(),
            rankings=_make_rankings(),
            output_dir=tmp_path / "figures",
        )
        for fig in figures:
            if fig.sidecar_path is not None and fig.sidecar_path.exists():
                data = json.loads(fig.sidecar_path.read_text(encoding="utf-8"))
                assert "figure_id" in data
                assert "title" in data
                assert "generated_at" in data

    def test_figure_count_matches_minimum(self, tmp_path: Path) -> None:
        figures = generate_figures(
            per_volume_data=_make_per_volume_data(),
            pairwise=_make_pairwise(),
            variance=_make_variance(),
            rankings=_make_rankings(),
            output_dir=tmp_path / "figures",
        )
        # Must produce at least effect size heatmap + forest plot
        assert len(figures) >= 2

    def test_output_dir_is_not_hardcoded(self, tmp_path: Path) -> None:
        custom_dir = tmp_path / "custom_output" / "figs"
        figures = generate_figures(
            per_volume_data=_make_per_volume_data(),
            pairwise=_make_pairwise(),
            variance=_make_variance(),
            rankings=_make_rankings(),
            output_dir=custom_dir,
        )
        for fig in figures:
            for p in fig.paths:
                assert str(custom_dir) in str(p)


class TestWriteSidecar:
    def test_sidecar_json_is_valid_json(self, tmp_path: Path) -> None:
        sidecar_data = {
            "figure_id": "test_fig",
            "title": "Test Figure",
            "generated_at": "2026-03-07T12:00:00+00:00",
            "data": {"conditions": ["a", "b"], "values": [0.8, 0.7]},
        }
        path = tmp_path / "test_sidecar.json"
        write_sidecar(sidecar_data, path)

        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["figure_id"] == "test_fig"

    def test_sidecar_round_trips_through_json(self, tmp_path: Path) -> None:
        sidecar_data = {
            "figure_id": "roundtrip",
            "title": "Roundtrip Test",
            "generated_at": "2026-03-07T12:00:00+00:00",
            "data": {"values": [1.0, 2.0, 3.0]},
        }
        path = tmp_path / "roundtrip.json"
        write_sidecar(sidecar_data, path)

        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded == sidecar_data


def _make_anova_result() -> FactorialAnovaResult:
    return FactorialAnovaResult(
        metric="cldice",
        n_models=2,
        n_losses=2,
        f_values={"Model": 5.2, "Loss": 3.8, "Model:Loss": 1.1},
        p_values={"Model": 0.03, "Loss": 0.06, "Model:Loss": 0.35},
        eta_squared_partial={"Model": 0.12, "Loss": 0.09, "Model:Loss": 0.02},
        omega_squared={"Model": 0.10, "Loss": 0.07, "Model:Loss": 0.01},
    )


class TestAnovaFigureWiring:
    """Task 2.13: verify generate_figures() produces ANOVA figures when given anova_results."""

    def test_interaction_plot_generated_when_anova_provided(
        self, tmp_path: Path
    ) -> None:
        per_volume = _make_per_volume_data()
        # Add cldice data to match the ANOVA result metric
        rng = np.random.default_rng(99)
        per_volume["cldice"] = {
            "dynunet__dice_ce": {0: rng.normal(0.85, 0.03, 10)},
            "mambavesselnet__dice_ce": {0: rng.normal(0.80, 0.04, 10)},
        }
        figures = generate_figures(
            per_volume_data=per_volume,
            pairwise=_make_pairwise(),
            variance=_make_variance(),
            rankings=_make_rankings(),
            output_dir=tmp_path / "figures",
            anova_results=[_make_anova_result()],
        )
        # Should have more figures than without ANOVA results
        fig_ids = [f.figure_id for f in figures]
        assert "interaction_plot" in fig_ids
        assert "variance_lollipop" in fig_ids

    def test_no_anova_figures_when_anova_results_is_none(
        self, tmp_path: Path
    ) -> None:
        figures_without = generate_figures(
            per_volume_data=_make_per_volume_data(),
            pairwise=_make_pairwise(),
            variance=_make_variance(),
            rankings=_make_rankings(),
            output_dir=tmp_path / "figures",
        )
        fig_ids = [f.figure_id for f in figures_without]
        assert "interaction_plot" not in fig_ids
        assert "variance_lollipop" not in fig_ids
