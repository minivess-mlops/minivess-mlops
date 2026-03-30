"""Tests for biostatistics verification — Plan Task 5.1.

Verifies verify_artifact_chain() and validate_statistical_results().
"""

from __future__ import annotations

import json
from pathlib import Path

from minivess.pipeline.biostatistics_types import (
    FactorialAnovaResult,
    PairwiseResult,
    RankingResult,
    VarianceDecompositionResult,
)


def _build_both_duckdbs(tmp_path: Path) -> tuple[Path, Path]:
    """Build both DuckDB files for verification testing."""
    import numpy as np

    from minivess.pipeline.biostatistics_duckdb import (
        build_analysis_results_duckdb,
        build_biostatistics_results_duckdb,
    )
    from minivess.pipeline.biostatistics_types import SourceRun, SourceRunManifest

    # Analysis DuckDB
    runs = [
        SourceRun(
            run_id=f"run_{i}", experiment_id="exp1",
            experiment_name="test", loss_function="dice_ce",
            fold_id=i % 2, status="FINISHED",
        )
        for i in range(4)
    ]
    manifest = SourceRunManifest.from_runs(runs)
    rng = np.random.default_rng(42)
    pv_records = [
        {"run_id": r.run_id, "fold_id": r.fold_id, "split": "trainval",
         "dataset": "minivess", "volume_id": f"vol_{j}",
         "metric_name": "dsc", "metric_value": float(rng.random())}
        for r in runs for j in range(3)
    ]
    fold_records = [
        {"run_id": r.run_id, "fold_id": r.fold_id, "split": "trainval",
         "metric_name": "dsc", "metric_value": float(rng.random())}
        for r in runs
    ]
    analysis_path = tmp_path / "analysis_results.duckdb"
    build_analysis_results_duckdb(
        manifest=manifest, per_volume_records=pv_records,
        fold_metric_records=fold_records, output_path=analysis_path,
        metadata={"git_sha": "test123"},
    )

    # Biostatistics DuckDB
    biostats_path = tmp_path / "biostatistics.duckdb"
    build_biostatistics_results_duckdb(
        pairwise_results=[
            PairwiseResult(
                condition_a="a", condition_b="b", metric="dsc",
                p_value=0.05, p_adjusted=0.1, correction_method="holm",
                significant=False, cohens_d=0.3, cliffs_delta=0.2, vda=0.6,
            ),
        ],
        anova_results=[
            FactorialAnovaResult(
                metric="dsc", n_models=1, n_losses=2,
                f_values={"loss": 2.0}, p_values={"loss": 0.1},
                eta_squared_partial={"loss": 0.1},
                omega_squared={"loss": 0.08},
                factor_names=["loss"],
            ),
        ],
        variance_results=[
            VarianceDecompositionResult(
                metric="dsc", friedman_statistic=3.0, friedman_p=0.2,
                nemenyi_matrix=None, icc_value=0.5, icc_ci_lower=0.1,
                icc_ci_upper=0.8, icc_type="ICC2",
            ),
        ],
        ranking_results=[
            RankingResult(
                metric="dsc",
                condition_ranks={"a": 1.0, "b": 2.0},
                mean_ranks={"a": 1.0, "b": 2.0},
                cd_value=0.5,
            ),
        ],
        diagnostics=[{
            "metric": "dsc", "test_type": "stratified_permutation",
            "alpha_used": 0.05, "effect_size_assumed": 0.5,
            "achieved_power": 0.45, "effective_n": 6,
            "design_effect": 1.1, "icc_within_fold": 0.1,
            "min_detectable_effect": 0.5,
            "recommended_additional_folds": 3, "recommendation": "Underpowered",
        }],
        tripod_items=[
            {"item_id": "9a", "item_name": "Sample size",
             "status": "addressed", "evidence": "N=2", "limitation": ""},
        ],
        metadata={"git_sha": "test123"},
        output_path=biostats_path,
    )

    return analysis_path, biostats_path


class TestVerifyArtifactChain:
    """Tests for verify_artifact_chain()."""

    def test_passes_with_valid_duckdbs(self, tmp_path: Path) -> None:
        """Should pass when both DuckDBs have all expected tables."""
        from minivess.pipeline.biostatistics_verification import (
            verify_artifact_chain,
        )

        analysis_path, biostats_path = _build_both_duckdbs(tmp_path)
        result = verify_artifact_chain(
            analysis_duckdb_path=analysis_path,
            biostatistics_duckdb_path=biostats_path,
        )
        assert result["passed"] is True
        assert result["n_passed"] == result["n_checks"]

    def test_fails_on_missing_file(self, tmp_path: Path) -> None:
        """Should fail when DuckDB file doesn't exist."""
        from minivess.pipeline.biostatistics_verification import (
            verify_artifact_chain,
        )

        result = verify_artifact_chain(
            analysis_duckdb_path=tmp_path / "nonexistent.duckdb",
        )
        assert result["passed"] is False

    def test_checks_json_sidecars(self, tmp_path: Path) -> None:
        """Should check all 11 JSON sidecar files."""
        from minivess.pipeline.biostatistics_verification import (
            verify_artifact_chain,
        )

        r_data_dir = tmp_path / "r_data"
        r_data_dir.mkdir()
        # Create all 11 expected sidecars
        for name in [
            "pairwise_results.json", "per_volume_data.json",
            "variance_decomposition.json", "rankings.json",
            "anova_results.json", "diagnostics.json",
            "calibration_data.json", "tripod_compliance.json",
            "metadata.json", "bayesian_results.json",
            "specification_curve.json",
        ]:
            (r_data_dir / name).write_text(
                json.dumps({"test": True}), encoding="utf-8"
            )

        result = verify_artifact_chain(r_data_dir=r_data_dir)
        sidecar_checks = [
            c for c in result["checks"] if c["name"].startswith("sidecar_")
        ]
        assert len(sidecar_checks) == 11
        assert all(c["passed"] for c in sidecar_checks)


class TestValidateStatisticalResults:
    """Tests for validate_statistical_results()."""

    def test_passes_with_valid_results(self, tmp_path: Path) -> None:
        """Should pass with valid statistical results."""
        from minivess.pipeline.biostatistics_verification import (
            validate_statistical_results,
        )

        _, biostats_path = _build_both_duckdbs(tmp_path)
        result = validate_statistical_results(
            biostatistics_duckdb_path=biostats_path,
        )
        assert result["passed"] is True

    def test_catches_invalid_cliffs_delta(self, tmp_path: Path) -> None:
        """Should catch Cliff's delta outside [-1, 1]."""
        from minivess.pipeline.biostatistics_duckdb import (
            build_biostatistics_results_duckdb,
        )
        from minivess.pipeline.biostatistics_verification import (
            validate_statistical_results,
        )

        biostats_path = tmp_path / "bad.duckdb"
        build_biostatistics_results_duckdb(
            pairwise_results=[
                PairwiseResult(
                    condition_a="a", condition_b="b", metric="dsc",
                    p_value=0.05, p_adjusted=0.1, correction_method="holm",
                    significant=False, cohens_d=0.3,
                    cliffs_delta=1.5,  # INVALID: > 1
                    vda=0.6,
                ),
            ],
            anova_results=[], variance_results=[], ranking_results=[],
            diagnostics=[], tripod_items=[], metadata={},
            output_path=biostats_path,
        )

        result = validate_statistical_results(
            biostatistics_duckdb_path=biostats_path,
        )
        delta_check = next(
            c for c in result["checks"] if c["name"] == "cliffs_delta_range"
        )
        assert delta_check["passed"] is False
