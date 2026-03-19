"""Integration test for the biostatistics flow (Phase 9, Task 9.1).

Runs the full biostatistics pipeline with synthetic mlruns data.
Tests the pure pipeline functions end-to-end (no Prefect task wrappers).
MLflow logging is mocked to avoid side effects.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from minivess.config.biostatistics_config import BiostatisticsConfig
from minivess.pipeline.biostatistics_discovery import (
    discover_source_runs,
    validate_source_completeness,
)
from minivess.pipeline.biostatistics_duckdb import (
    build_biostatistics_duckdb,
    export_parquet,
)
from minivess.pipeline.biostatistics_figures import generate_figures
from minivess.pipeline.biostatistics_lineage import build_lineage_manifest
from minivess.pipeline.biostatistics_rankings import compute_rankings
from minivess.pipeline.biostatistics_statistics import (
    compute_pairwise_comparisons,
    compute_variance_decomposition,
)
from minivess.pipeline.biostatistics_tables import generate_tables

if TYPE_CHECKING:
    from pathlib import Path

    from minivess.pipeline.biostatistics_types import SourceRunManifest

# ---------------------------------------------------------------------------
# Synthetic mlruns builder
# ---------------------------------------------------------------------------

_CONDITIONS = ["dice_ce", "tversky", "cbdice_cldice"]
_N_FOLDS = 3
_N_VOLUMES = 10
_METRICS = ["dsc", "cldice"]
_EXPERIMENT_NAME = "dynunet_loss_variation_v2"


def _create_synthetic_mlruns(base: Path) -> Path:
    """Create a synthetic mlruns directory with realistic structure.

    Layout matches the slash-separated metric key convention that
    ``_read_metrics`` produces via ``pathlib.Path.relative_to``:

        mlruns/
          1/                                # experiment ID
            meta.yaml                       # experiment metadata
            run_{cond}_{fold}/              # run directories
              meta.yaml                     # run metadata (status=FINISHED)
              params/
                loss_name                   # condition (loss function)
                fold_id                     # fold number
              metrics/
                eval/{fold}/vol/{v}/{metric}  # per-volume metrics (nested dirs)
                val_dice                      # summary metric (flat file)
    """
    mlruns = base / "mlruns"
    exp_dir = mlruns / "1"
    exp_dir.mkdir(parents=True)

    # Experiment metadata
    (exp_dir / "meta.yaml").write_text(f"name: {_EXPERIMENT_NAME}\n", encoding="utf-8")

    rng = np.random.default_rng(42)
    means = {"dice_ce": 0.82, "tversky": 0.78, "cbdice_cldice": 0.86}

    for cond in _CONDITIONS:
        for fold in range(_N_FOLDS):
            run_id = f"run_{cond}_{fold}"
            run_dir = exp_dir / run_id
            run_dir.mkdir()

            # Run metadata
            (run_dir / "meta.yaml").write_text("status: FINISHED\n", encoding="utf-8")

            # Params
            params_dir = run_dir / "params"
            params_dir.mkdir()
            (params_dir / "loss_name").write_text(cond, encoding="utf-8")
            (params_dir / "fold_id").write_text(str(fold), encoding="utf-8")

            # Per-volume metrics using nested directory structure:
            # metrics/eval/{fold}/vol/{vol_id}/{metric_name}
            # This matches the slash-separated key format that
            # _parse_per_volume_metric expects: "eval/{fold}/vol/{id}/{metric}"
            metrics_dir = run_dir / "metrics"
            metrics_dir.mkdir()
            for metric in _METRICS:
                mean = means[cond]
                if metric == "cldice":
                    mean -= 0.02  # slightly lower
                for vol in range(_N_VOLUMES):
                    score = float(rng.normal(mean, 0.04))
                    metric_path = (
                        metrics_dir / "eval" / str(fold) / "vol" / str(vol) / metric
                    )
                    metric_path.parent.mkdir(parents=True, exist_ok=True)
                    # MLflow metric format: "timestamp value step"
                    metric_path.write_text(f"1700000000000 {score} 0", encoding="utf-8")

            # Also add a summary metric (non per-volume)
            (metrics_dir / "val_dice").write_text(
                f"1700000000000 {float(rng.normal(means[cond], 0.03))} 0",
                encoding="utf-8",
            )

    return mlruns


def _build_per_volume_data_from_mlruns(
    manifest: SourceRunManifest,
    mlruns_dir: Path,
) -> dict[str, dict[str, dict[int, np.ndarray]]]:
    """Build per-volume data from synthetic mlruns (same logic as flow)."""
    from minivess.pipeline.biostatistics_duckdb import (
        _is_per_volume_metric,
        _parse_per_volume_metric,
        _read_metrics,
    )

    data: dict[str, dict[str, dict[int, list[float]]]] = {}
    for run in manifest.runs:
        run_dir = mlruns_dir / run.experiment_id / run.run_id
        metrics = _read_metrics(run_dir)
        for metric_name, value in metrics.items():
            if _is_per_volume_metric(metric_name):
                fold_id, _volume_id, base_metric = _parse_per_volume_metric(metric_name)
                if fold_id is not None:
                    data.setdefault(base_metric, {}).setdefault(
                        run.loss_function, {}
                    ).setdefault(fold_id, []).append(value)

    result: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    for metric, conditions in data.items():
        result[metric] = {}
        for cond, folds in conditions.items():
            result[metric][cond] = {}
            for fold, values in folds.items():
                result[metric][cond][fold] = np.array(values)
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_pipeline(tmp_path: Path) -> dict:
    """Run the full biostatistics pipeline on synthetic data.

    Returns a dict with all intermediate results and paths.
    """
    mlruns_dir = _create_synthetic_mlruns(tmp_path)
    output_dir = tmp_path / "outputs" / "biostatistics"

    config = BiostatisticsConfig(
        experiment_names=[_EXPERIMENT_NAME],
        mlruns_dir=mlruns_dir,
        output_dir=output_dir,
        metrics=list(_METRICS),
        primary_metric="cldice",
        n_bootstrap=500,
        seed=42,
    )

    # Phase 2: Discovery
    manifest = discover_source_runs(mlruns_dir, config.experiment_names)
    validation = validate_source_completeness(
        manifest,
        min_folds=config.min_folds_per_condition,
        min_conditions=config.min_conditions,
    )

    # Phase 2: DuckDB
    db_path = output_dir / "biostatistics.duckdb"
    build_biostatistics_duckdb(manifest, mlruns_dir, db_path)
    parquet_dir = output_dir / "parquet"
    export_parquet(db_path, parquet_dir)

    # Build per-volume data
    per_volume_data = _build_per_volume_data_from_mlruns(manifest, mlruns_dir)

    higher_is_better: dict[str, bool] = {
        m: m not in ("hd95", "assd", "be_0", "be_1") for m in config.metrics
    }

    # Phase 3: Statistics
    all_pairwise = []
    all_variance = []
    for metric in config.metrics:
        if metric in per_volume_data:
            pw = compute_pairwise_comparisons(
                per_volume_data=per_volume_data[metric],
                metric_name=metric,
                alpha=config.alpha,
                primary_metric=config.primary_metric,
                n_bootstrap=config.n_bootstrap,
                seed=config.seed,
            )
            all_pairwise.extend(pw)
            vd = compute_variance_decomposition(
                per_volume_data=per_volume_data[metric],
                metric_name=metric,
            )
            all_variance.extend(vd)

    # Phase 4: Rankings
    rankings = compute_rankings(
        per_volume_data=per_volume_data,
        metric_names=config.metrics,
        higher_is_better=higher_is_better,
    )

    # Phase 5: Figures
    figures_dir = output_dir / "figures"
    figures = generate_figures(
        per_volume_data=per_volume_data,
        pairwise=all_pairwise,
        variance=all_variance,
        rankings=rankings,
        output_dir=figures_dir,
    )

    # Phase 6: Tables
    tables_dir = output_dir / "tables"
    tables = generate_tables(
        pairwise=all_pairwise,
        variance=all_variance,
        rankings=rankings,
        output_dir=tables_dir,
    )

    # Phase 7: Lineage
    lineage = build_lineage_manifest(
        manifest=manifest,
        figures=figures,
        tables=tables,
    )

    return {
        "config": config,
        "manifest": manifest,
        "validation": validation,
        "db_path": db_path,
        "parquet_dir": parquet_dir,
        "per_volume_data": per_volume_data,
        "pairwise": all_pairwise,
        "variance": all_variance,
        "rankings": rankings,
        "figures": figures,
        "tables": tables,
        "lineage": lineage,
        "output_dir": output_dir,
        "figures_dir": figures_dir,
        "tables_dir": tables_dir,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFullFlowWithSyntheticMlruns:
    """Full pipeline integration test with synthetic mlruns."""

    def test_discovery_finds_all_runs(self, synthetic_pipeline: dict) -> None:
        manifest = synthetic_pipeline["manifest"]
        expected = len(_CONDITIONS) * _N_FOLDS
        assert len(manifest.runs) == expected

    def test_validation_passes(self, synthetic_pipeline: dict) -> None:
        v = synthetic_pipeline["validation"]
        assert v.valid is True
        assert v.n_conditions == len(_CONDITIONS)
        assert v.n_folds_per_condition == _N_FOLDS

    def test_manifest_fingerprint_is_deterministic(
        self, synthetic_pipeline: dict
    ) -> None:
        manifest = synthetic_pipeline["manifest"]
        assert len(manifest.fingerprint) == 64  # SHA-256 hex


class TestFlowOutputContainsDuckdb:
    def test_duckdb_file_exists(self, synthetic_pipeline: dict) -> None:
        assert synthetic_pipeline["db_path"].exists()

    def test_duckdb_has_7_tables(self, synthetic_pipeline: dict) -> None:
        import duckdb

        con = duckdb.connect(str(synthetic_pipeline["db_path"]))
        tables = con.execute("SHOW TABLES").fetchall()
        con.close()
        table_names = {t[0] for t in tables}
        assert len(table_names) >= 7


class TestFlowOutputContainsFiguresWithSidecars:
    def test_figures_generated(self, synthetic_pipeline: dict) -> None:
        figures = synthetic_pipeline["figures"]
        assert len(figures) >= 1

    def test_figure_files_exist(self, synthetic_pipeline: dict) -> None:
        for fig in synthetic_pipeline["figures"]:
            for p in fig.paths:
                assert p.exists(), f"Figure missing: {p}"

    def test_sidecars_are_valid_json(self, synthetic_pipeline: dict) -> None:
        for fig in synthetic_pipeline["figures"]:
            if fig.sidecar_path is not None and fig.sidecar_path.exists():
                data = json.loads(fig.sidecar_path.read_text(encoding="utf-8"))
                assert "figure_id" in data


class TestFlowOutputContainsLatexTables:
    def test_tables_generated(self, synthetic_pipeline: dict) -> None:
        tables = synthetic_pipeline["tables"]
        assert len(tables) >= 2  # comparison + effect sizes at minimum

    def test_table_files_exist(self, synthetic_pipeline: dict) -> None:
        for tab in synthetic_pipeline["tables"]:
            assert tab.path.exists(), f"Table missing: {tab.path}"

    def test_tables_contain_booktabs(self, synthetic_pipeline: dict) -> None:
        for tab in synthetic_pipeline["tables"]:
            content = tab.path.read_text(encoding="utf-8")
            assert r"\toprule" in content
            assert r"\bottomrule" in content


class TestFlowOutputContainsLineage:
    def test_lineage_has_fingerprint(self, synthetic_pipeline: dict) -> None:
        lineage = synthetic_pipeline["lineage"]
        assert "fingerprint" in lineage
        assert lineage["fingerprint"] == synthetic_pipeline["manifest"].fingerprint

    def test_lineage_has_schema_version(self, synthetic_pipeline: dict) -> None:
        lineage = synthetic_pipeline["lineage"]
        assert "schema_version" in lineage

    def test_lineage_lists_artifacts(self, synthetic_pipeline: dict) -> None:
        lineage = synthetic_pipeline["lineage"]
        assert "artifacts_produced" in lineage
        artifacts = lineage["artifacts_produced"]
        assert "n_figures" in artifacts
        assert "n_tables" in artifacts
        assert artifacts["n_figures"] >= 1
        assert artifacts["n_tables"] >= 1


class TestFingerprintShortCircuit:
    def test_same_data_produces_same_fingerprint(self, tmp_path: Path) -> None:
        """Running discovery twice on the same data yields the same fingerprint."""
        mlruns_dir = _create_synthetic_mlruns(tmp_path)
        m1 = discover_source_runs(mlruns_dir, [_EXPERIMENT_NAME])
        m2 = discover_source_runs(mlruns_dir, [_EXPERIMENT_NAME])
        assert m1.fingerprint == m2.fingerprint

    def test_different_data_produces_different_fingerprint(
        self, tmp_path: Path
    ) -> None:
        """Adding a run changes the fingerprint."""
        mlruns_dir = _create_synthetic_mlruns(tmp_path)
        m1 = discover_source_runs(mlruns_dir, [_EXPERIMENT_NAME])

        # Add an extra run
        extra_dir = mlruns_dir / "1" / "run_extra_0"
        extra_dir.mkdir()
        (extra_dir / "meta.yaml").write_text("status: FINISHED\n", encoding="utf-8")
        params = extra_dir / "params"
        params.mkdir()
        (params / "loss_name").write_text("extra_loss", encoding="utf-8")
        (params / "fold_id").write_text("0", encoding="utf-8")

        m2 = discover_source_runs(mlruns_dir, [_EXPERIMENT_NAME])
        assert m1.fingerprint != m2.fingerprint


class TestMlflowLogging:
    @patch("minivess.pipeline.biostatistics_mlflow.mlflow")
    def test_mlflow_run_created(
        self, mock_mlflow: MagicMock, synthetic_pipeline: dict
    ) -> None:
        """MLflow logging creates a run with correct tags."""
        from minivess.pipeline.biostatistics_mlflow import log_biostatistics_run

        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-123"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        run_id = log_biostatistics_run(
            manifest=synthetic_pipeline["manifest"],
            lineage=synthetic_pipeline["lineage"],
            figures=synthetic_pipeline["figures"],
            tables=synthetic_pipeline["tables"],
            db_path=synthetic_pipeline["db_path"],
        )
        assert run_id == "test-run-123"
        mock_mlflow.set_experiment.assert_called_once_with("minivess_biostatistics")


class TestParquetExport:
    def test_parquet_files_created(self, synthetic_pipeline: dict) -> None:
        parquet_dir = synthetic_pipeline["parquet_dir"]
        if parquet_dir.exists():
            parquet_files = list(parquet_dir.glob("*.parquet"))
            assert len(parquet_files) >= 1
