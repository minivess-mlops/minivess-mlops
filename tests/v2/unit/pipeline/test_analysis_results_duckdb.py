"""Tests for build_analysis_results_duckdb() — Plan Task 1.2.

Verifies the analysis DuckDB (DuckDB 1) builder:
- Creates 4 tables (runs, per_volume_metrics, fold_metrics, metadata)
- Correct schema with metric_value column (matching existing convention)
- condition_key ONLY in runs table, NOT in metric tables
- Parquet export for all tables
- Round-trip: write → read → data matches
- Metadata includes biostatistics_config_json

Pure unit tests — no Docker, no Prefect, no DagsHub.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np

from minivess.pipeline.biostatistics_types import SourceRun, SourceRunManifest


def _make_test_runs() -> list[SourceRun]:
    """Create 8 test runs: 4 conditions × 2 folds."""
    runs = []
    configs = [
        ("dice_ce", False),
        ("dice_ce", True),
        ("cbdice_cldice", False),
        ("cbdice_cldice", True),
    ]
    for loss, aux in configs:
        for fold in range(2):
            runs.append(
                SourceRun(
                    run_id=f"run_{loss}_{aux}_{fold}",
                    experiment_id="exp1",
                    experiment_name="test_exp",
                    loss_function=loss,
                    fold_id=fold,
                    model_family="dynunet",
                    with_aux_calib=aux,
                    status="FINISHED",
                )
            )
    return runs


def _make_per_volume_data(
    runs: list[SourceRun],
) -> dict[str, dict[str, dict[str, float]]]:
    """Create per-volume metric data keyed by (run_id, fold_id, volume_id, metric)."""
    records: list[dict[str, object]] = []
    metrics = ["dsc", "hd95", "cldice", "cal_ece"]
    for run in runs:
        for vol_idx in range(5):
            vol_id = f"vol_{vol_idx:03d}"
            for metric in metrics:
                records.append({
                    "run_id": run.run_id,
                    "fold_id": run.fold_id,
                    "split": "trainval",
                    "dataset": "minivess",
                    "volume_id": vol_id,
                    "metric_name": metric,
                    "metric_value": float(np.random.default_rng(42).random()),
                })
    return records


def _make_fold_metrics(runs: list[SourceRun]) -> list[dict[str, object]]:
    """Create fold-level metric data."""
    records: list[dict[str, object]] = []
    rng = np.random.default_rng(42)
    for run in runs:
        for metric in ["dsc", "hd95", "cldice"]:
            records.append({
                "run_id": run.run_id,
                "fold_id": run.fold_id,
                "split": "trainval",
                "metric_name": metric,
                "metric_value": float(rng.random()),
            })
    return records


class TestBuildAnalysisResultsDuckdb:
    """Tests for build_analysis_results_duckdb()."""

    def test_creates_four_tables(self, tmp_path: Path) -> None:
        """DuckDB 1 must have exactly 4 tables."""
        from minivess.pipeline.biostatistics_duckdb import (
            build_analysis_results_duckdb,
        )

        runs = _make_test_runs()
        manifest = SourceRunManifest.from_runs(runs)
        db_path = tmp_path / "analysis_results.duckdb"

        build_analysis_results_duckdb(
            manifest=manifest,
            per_volume_records=_make_per_volume_data(runs),
            fold_metric_records=_make_fold_metrics(runs),
            output_path=db_path,
            metadata={"git_sha": "abc123"},
        )

        conn = duckdb.connect(str(db_path), read_only=True)
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables ORDER BY table_name"
        ).fetchall()
        conn.close()

        table_names = [t[0] for t in tables]
        assert "runs" in table_names
        assert "per_volume_metrics" in table_names
        assert "fold_metrics" in table_names
        assert "metadata" in table_names
        assert len(table_names) == 4

    def test_runs_table_has_condition_key(self, tmp_path: Path) -> None:
        """runs table must have condition_key column."""
        from minivess.pipeline.biostatistics_duckdb import (
            build_analysis_results_duckdb,
        )

        runs = _make_test_runs()
        manifest = SourceRunManifest.from_runs(runs)
        db_path = tmp_path / "analysis_results.duckdb"

        build_analysis_results_duckdb(
            manifest=manifest,
            per_volume_records=_make_per_volume_data(runs),
            fold_metric_records=_make_fold_metrics(runs),
            output_path=db_path,
            metadata={},
        )

        conn = duckdb.connect(str(db_path), read_only=True)
        cols = conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'runs'"
        ).fetchall()
        conn.close()

        col_names = [c[0] for c in cols]
        assert "condition_key" in col_names

    def test_per_volume_metrics_no_condition_key(self, tmp_path: Path) -> None:
        """per_volume_metrics must NOT have condition_key (join via runs)."""
        from minivess.pipeline.biostatistics_duckdb import (
            build_analysis_results_duckdb,
        )

        runs = _make_test_runs()
        manifest = SourceRunManifest.from_runs(runs)
        db_path = tmp_path / "analysis_results.duckdb"

        build_analysis_results_duckdb(
            manifest=manifest,
            per_volume_records=_make_per_volume_data(runs),
            fold_metric_records=_make_fold_metrics(runs),
            output_path=db_path,
            metadata={},
        )

        conn = duckdb.connect(str(db_path), read_only=True)
        cols = conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'per_volume_metrics'"
        ).fetchall()
        conn.close()

        col_names = [c[0] for c in cols]
        assert "condition_key" not in col_names

    def test_correct_row_counts(self, tmp_path: Path) -> None:
        """Verify expected row counts in all tables."""
        from minivess.pipeline.biostatistics_duckdb import (
            build_analysis_results_duckdb,
        )

        runs = _make_test_runs()
        manifest = SourceRunManifest.from_runs(runs)
        pv_data = _make_per_volume_data(runs)
        fold_data = _make_fold_metrics(runs)
        db_path = tmp_path / "analysis_results.duckdb"

        build_analysis_results_duckdb(
            manifest=manifest,
            per_volume_records=pv_data,
            fold_metric_records=fold_data,
            output_path=db_path,
            metadata={"git_sha": "abc123"},
        )

        conn = duckdb.connect(str(db_path), read_only=True)
        n_runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        n_pv = conn.execute("SELECT COUNT(*) FROM per_volume_metrics").fetchone()[0]
        n_fold = conn.execute("SELECT COUNT(*) FROM fold_metrics").fetchone()[0]
        n_meta = conn.execute("SELECT COUNT(*) FROM metadata").fetchone()[0]
        conn.close()

        assert n_runs == 8  # 4 conditions × 2 folds
        assert n_pv == len(pv_data)  # 8 runs × 5 volumes × 4 metrics = 160
        assert n_fold == len(fold_data)  # 8 runs × 3 metrics = 24
        assert n_meta >= 1  # At least git_sha

    def test_parquet_export(self, tmp_path: Path) -> None:
        """All 4 tables exported as Parquet files."""
        from minivess.pipeline.biostatistics_duckdb import (
            build_analysis_results_duckdb,
            export_analysis_parquet,
        )

        runs = _make_test_runs()
        manifest = SourceRunManifest.from_runs(runs)
        db_path = tmp_path / "analysis_results.duckdb"

        build_analysis_results_duckdb(
            manifest=manifest,
            per_volume_records=_make_per_volume_data(runs),
            fold_metric_records=_make_fold_metrics(runs),
            output_path=db_path,
            metadata={},
        )

        parquet_dir = tmp_path / "parquet"
        paths = export_analysis_parquet(db_path, parquet_dir)

        assert len(paths) == 4
        for p in paths:
            assert p.exists()
            assert p.suffix == ".parquet"

    def test_round_trip_per_volume(self, tmp_path: Path) -> None:
        """Write per-volume data → read back → values match."""
        from minivess.pipeline.biostatistics_duckdb import (
            build_analysis_results_duckdb,
        )

        runs = _make_test_runs()
        manifest = SourceRunManifest.from_runs(runs)
        pv_data = _make_per_volume_data(runs)
        db_path = tmp_path / "analysis_results.duckdb"

        build_analysis_results_duckdb(
            manifest=manifest,
            per_volume_records=pv_data,
            fold_metric_records=_make_fold_metrics(runs),
            output_path=db_path,
            metadata={},
        )

        conn = duckdb.connect(str(db_path), read_only=True)
        result = conn.execute(
            "SELECT run_id, fold_id, volume_id, metric_name, metric_value "
            "FROM per_volume_metrics WHERE run_id = ? AND metric_name = ?",
            ["run_dice_ce_False_0", "dsc"],
        ).fetchall()
        conn.close()

        assert len(result) == 5  # 5 volumes
        for row in result:
            assert row[0] == "run_dice_ce_False_0"
            assert row[3] == "dsc"
            assert isinstance(row[4], float)

    def test_metadata_has_config_json(self, tmp_path: Path) -> None:
        """metadata table must store biostatistics_config_json."""
        from minivess.pipeline.biostatistics_duckdb import (
            build_analysis_results_duckdb,
        )

        runs = _make_test_runs()
        manifest = SourceRunManifest.from_runs(runs)
        db_path = tmp_path / "analysis_results.duckdb"

        config_dict = {"alpha": 0.05, "n_bootstrap": 10000}

        build_analysis_results_duckdb(
            manifest=manifest,
            per_volume_records=_make_per_volume_data(runs),
            fold_metric_records=_make_fold_metrics(runs),
            output_path=db_path,
            metadata={
                "git_sha": "abc123",
                "biostatistics_config_json": json.dumps(config_dict),
            },
        )

        conn = duckdb.connect(str(db_path), read_only=True)
        row = conn.execute(
            "SELECT value FROM metadata WHERE key = 'biostatistics_config_json'"
        ).fetchone()
        conn.close()

        assert row is not None
        parsed = json.loads(row[0])
        assert parsed["alpha"] == 0.05

    def test_per_volume_has_split_and_dataset(self, tmp_path: Path) -> None:
        """per_volume_metrics table must have split and dataset columns."""
        from minivess.pipeline.biostatistics_duckdb import (
            build_analysis_results_duckdb,
        )

        runs = _make_test_runs()
        manifest = SourceRunManifest.from_runs(runs)
        db_path = tmp_path / "analysis_results.duckdb"

        build_analysis_results_duckdb(
            manifest=manifest,
            per_volume_records=_make_per_volume_data(runs),
            fold_metric_records=_make_fold_metrics(runs),
            output_path=db_path,
            metadata={},
        )

        conn = duckdb.connect(str(db_path), read_only=True)
        cols = conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'per_volume_metrics'"
        ).fetchall()
        conn.close()

        col_names = [c[0] for c in cols]
        assert "split" in col_names
        assert "dataset" in col_names
