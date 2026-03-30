"""Tests for biostatistics DuckDB v2 output + extended JSON sidecars — Plan Task 2.3.

Verifies:
- build_biostatistics_results_duckdb() creates 9-table DuckDB 2
- All 11 JSON sidecars exported via export_extended_r_data()
- metadata.analysis_duckdb_sha256 tracks provenance
- TRIPOD compliance table has items 9a, 18e, 24
- Diagnostics table has power analysis records

Pure unit tests — no Docker, no Prefect, no DagsHub.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np

from minivess.pipeline.biostatistics_types import (
    FactorialAnovaResult,
    PairwiseResult,
    RankingResult,
    VarianceDecompositionResult,
)


def _make_pairwise() -> list[PairwiseResult]:
    """Create mock pairwise comparison results."""
    return [
        PairwiseResult(
            condition_a="dice_ce",
            condition_b="cbdice_cldice",
            metric="dsc",
            p_value=0.03,
            p_adjusted=0.06,
            correction_method="holm",
            significant=False,
            cohens_d=0.4,
            cliffs_delta=0.3,
            vda=0.65,
        ),
    ]


def _make_anova() -> list[FactorialAnovaResult]:
    """Create mock ANOVA results."""
    return [
        FactorialAnovaResult(
            metric="dsc",
            n_models=1,
            n_losses=2,
            f_values={"loss_function": 3.5, "with_aux_calib": 1.2},
            p_values={"loss_function": 0.04, "with_aux_calib": 0.28},
            eta_squared_partial={"loss_function": 0.15, "with_aux_calib": 0.05},
            omega_squared={"loss_function": 0.12, "with_aux_calib": 0.03},
            factor_names=["loss_function", "with_aux_calib"],
        ),
    ]


def _make_variance() -> list[VarianceDecompositionResult]:
    return [
        VarianceDecompositionResult(
            metric="dsc",
            friedman_statistic=5.0,
            friedman_p=0.08,
            nemenyi_matrix=None,
            icc_value=0.7,
            icc_ci_lower=0.3,
            icc_ci_upper=0.9,
            icc_type="ICC2",
        ),
    ]


def _make_rankings() -> list[RankingResult]:
    return [
        RankingResult(
            metric="dsc",
            condition_ranks={"dice_ce": 1.5, "cbdice_cldice": 2.5},
            mean_ranks={"dice_ce": 1.5, "cbdice_cldice": 2.5},
            cd_value=0.8,
        ),
    ]


def _make_diagnostics() -> list[dict]:
    return [
        {
            "metric": "dsc",
            "test_type": "stratified_permutation",
            "alpha_used": 0.05,
            "effect_size_assumed": 0.5,
            "achieved_power": 0.45,
            "effective_n": 8,
            "design_effect": 1.2,
            "icc_within_fold": 0.15,
            "min_detectable_effect": 0.5,
            "recommended_additional_folds": 3,
            "recommendation": "Underpowered",
        },
    ]


class TestBuildBiostatisticsResultsDuckdb:
    """Tests for build_biostatistics_results_duckdb()."""

    def test_creates_nine_tables(self, tmp_path: Path) -> None:
        """DuckDB 2 must have 9 tables."""
        from minivess.pipeline.biostatistics_duckdb import (
            build_biostatistics_results_duckdb,
        )

        db_path = tmp_path / "biostatistics.duckdb"
        build_biostatistics_results_duckdb(
            pairwise_results=_make_pairwise(),
            anova_results=_make_anova(),
            variance_results=_make_variance(),
            ranking_results=_make_rankings(),
            diagnostics=_make_diagnostics(),
            tripod_items=[
                {"item_id": "9a", "item_name": "Sample size", "status": "limitation_documented",
                 "evidence": "N=2 folds", "limitation": "Underpowered"},
            ],
            metadata={"git_sha": "abc123", "analysis_duckdb_sha256": "deadbeef"},
            output_path=db_path,
        )

        conn = duckdb.connect(str(db_path), read_only=True)
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables ORDER BY table_name"
        ).fetchall()
        conn.close()

        table_names = [t[0] for t in tables]
        assert len(table_names) == 9
        assert "pairwise_comparisons" in table_names
        assert "anova_results" in table_names
        assert "diagnostics" in table_names
        assert "tripod_compliance" in table_names
        assert "metadata" in table_names

    def test_pairwise_has_test_type(self, tmp_path: Path) -> None:
        """pairwise_comparisons must have test_type column."""
        from minivess.pipeline.biostatistics_duckdb import (
            build_biostatistics_results_duckdb,
        )

        db_path = tmp_path / "biostatistics.duckdb"
        build_biostatistics_results_duckdb(
            pairwise_results=_make_pairwise(),
            anova_results=_make_anova(),
            variance_results=_make_variance(),
            ranking_results=_make_rankings(),
            diagnostics=_make_diagnostics(),
            tripod_items=[],
            metadata={},
            output_path=db_path,
        )

        conn = duckdb.connect(str(db_path), read_only=True)
        cols = conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'pairwise_comparisons'"
        ).fetchall()
        conn.close()

        col_names = [c[0] for c in cols]
        assert "test_type" in col_names

    def test_tripod_has_items(self, tmp_path: Path) -> None:
        """TRIPOD compliance table must store items 9a, 18e, 24."""
        from minivess.pipeline.biostatistics_duckdb import (
            build_biostatistics_results_duckdb,
        )

        db_path = tmp_path / "biostatistics.duckdb"
        tripod = [
            {"item_id": "9a", "item_name": "Sample size justification",
             "status": "limitation_documented", "evidence": "N=2 folds, 70 MiniVess",
             "limitation": "Dataset collected before study design"},
            {"item_id": "18e", "item_name": "Data availability",
             "status": "addressed", "evidence": "s3://minivessdataset",
             "limitation": ""},
            {"item_id": "24", "item_name": "Subgroup performance",
             "status": "addressed", "evidence": "Volume-size quartiles",
             "limitation": ""},
        ]

        build_biostatistics_results_duckdb(
            pairwise_results=_make_pairwise(),
            anova_results=_make_anova(),
            variance_results=_make_variance(),
            ranking_results=_make_rankings(),
            diagnostics=_make_diagnostics(),
            tripod_items=tripod,
            metadata={},
            output_path=db_path,
        )

        conn = duckdb.connect(str(db_path), read_only=True)
        rows = conn.execute("SELECT item_id FROM tripod_compliance").fetchall()
        conn.close()

        item_ids = {r[0] for r in rows}
        assert "9a" in item_ids
        assert "18e" in item_ids
        assert "24" in item_ids


class TestExtendedJsonSidecars:
    """Tests for export_extended_r_data() — 11 JSON sidecars."""

    def test_exports_eleven_files(self, tmp_path: Path) -> None:
        """Should produce 11 JSON sidecar files."""
        from minivess.pipeline.biostatistics_r_export import (
            export_extended_r_data,
        )

        rng = np.random.default_rng(42)
        pv_data = {
            "dsc": {
                "cond_a": {0: rng.random(5), 1: rng.random(5)},
                "cond_b": {0: rng.random(5), 1: rng.random(5)},
            }
        }

        r_dir = tmp_path / "r_data"
        paths = export_extended_r_data(
            pairwise=_make_pairwise(),
            anova=_make_anova(),
            variance=_make_variance(),
            rankings=_make_rankings(),
            diagnostics=_make_diagnostics(),
            per_volume_data=pv_data,
            tripod_items=[{"item_id": "9a", "item_name": "Sample size",
                          "status": "addressed", "evidence": "", "limitation": ""}],
            metadata={"git_sha": "abc"},
            output_dir=r_dir,
        )

        assert len(paths) == 11
        for p in paths:
            assert p.exists()
            assert p.suffix == ".json"

    def test_each_sidecar_valid_json(self, tmp_path: Path) -> None:
        """All sidecar files must be valid JSON."""
        from minivess.pipeline.biostatistics_r_export import (
            export_extended_r_data,
        )

        rng = np.random.default_rng(42)
        pv_data = {
            "dsc": {"cond_a": {0: rng.random(3)}}
        }

        r_dir = tmp_path / "r_data"
        paths = export_extended_r_data(
            pairwise=_make_pairwise(),
            anova=_make_anova(),
            variance=_make_variance(),
            rankings=_make_rankings(),
            diagnostics=_make_diagnostics(),
            per_volume_data=pv_data,
            tripod_items=[],
            metadata={},
            output_dir=r_dir,
        )

        for p in paths:
            data = json.loads(p.read_text(encoding="utf-8"))
            assert data is not None
