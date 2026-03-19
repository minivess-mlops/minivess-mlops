"""Tests for biostatistics DuckDB schema and extraction (Phase 2, Task 2.1)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml

from minivess.pipeline.biostatistics_discovery import discover_source_runs
from minivess.pipeline.biostatistics_duckdb import (
    BIOSTATISTICS_TABLES,
    build_biostatistics_duckdb,
    export_parquet,
)

if TYPE_CHECKING:
    from pathlib import Path


def _create_mock_mlruns_with_metrics(
    tmp_path: Path,
    conditions: list[str],
    n_folds: int = 3,
    n_volumes: int = 5,
) -> Path:
    """Create mlruns with per-fold metrics and per-volume metrics."""
    mlruns = tmp_path / "mlruns"
    exp_dir = mlruns / "1"
    exp_dir.mkdir(parents=True)

    (exp_dir / "meta.yaml").write_text(
        yaml.dump({"name": "test_experiment"}), encoding="utf-8"
    )

    run_idx = 0
    for loss in conditions:
        for fold in range(n_folds):
            run_id = f"run_{run_idx:03d}"
            run_dir = exp_dir / run_id
            run_dir.mkdir()
            (run_dir / "meta.yaml").write_text(
                yaml.dump({"status": "FINISHED"}), encoding="utf-8"
            )

            # Params
            params_dir = run_dir / "params"
            params_dir.mkdir()
            (params_dir / "loss_name").write_text(loss, encoding="utf-8")
            (params_dir / "fold_id").write_text(str(fold), encoding="utf-8")
            (params_dir / "learning_rate").write_text("0.001", encoding="utf-8")

            # Tags
            tags_dir = run_dir / "tags"
            tags_dir.mkdir()
            (tags_dir / "loss_function").write_text(loss, encoding="utf-8")
            (tags_dir / "model_family").write_text("dynunet", encoding="utf-8")

            # Metrics (MLflow format: each metric is a file with lines "timestamp value step")
            metrics_dir = run_dir / "metrics"
            metrics_dir.mkdir()

            # Eval metrics per fold (slash-prefix, #790)
            eval_fold_dir = metrics_dir / "eval" / str(fold)
            eval_fold_dir.mkdir(parents=True, exist_ok=True)
            (eval_fold_dir / "val_dice").write_text(
                f"1000 {0.8 + run_idx * 0.01} 100", encoding="utf-8"
            )
            (eval_fold_dir / "val_cldice").write_text(
                f"1000 {0.7 + run_idx * 0.01} 100", encoding="utf-8"
            )

            # Training metrics (slash-prefix, #790)
            train_dir = metrics_dir / "train"
            train_dir.mkdir(parents=True, exist_ok=True)
            (train_dir / "loss").write_text(
                f"1000 {0.3 - run_idx * 0.01} 100", encoding="utf-8"
            )
            val_dir = metrics_dir / "val"
            val_dir.mkdir(parents=True, exist_ok=True)
            (val_dir / "dice").write_text(
                f"1000 {0.8 + run_idx * 0.01} 100", encoding="utf-8"
            )

            # Per-volume metrics (eval/{fold}/vol/{id}/{metric}, slash-prefix #790)
            for vol_idx in range(n_volumes):
                vol_id = f"vol_{vol_idx:03d}"
                vol_dir = metrics_dir / "eval" / str(fold) / "vol" / vol_id
                vol_dir.mkdir(parents=True, exist_ok=True)
                (vol_dir / "dice").write_text(
                    f"1000 {0.75 + vol_idx * 0.02 + run_idx * 0.005} 100",
                    encoding="utf-8",
                )

            run_idx += 1

    return mlruns


class TestBuildBiostatisticsDuckdb:
    def test_schema_has_7_tables(self, tmp_path: Path) -> None:
        mlruns = _create_mock_mlruns_with_metrics(
            tmp_path, ["dice_ce", "tversky"], n_folds=2
        )
        manifest = discover_source_runs(mlruns, ["test_experiment"])
        db_path = tmp_path / "biostatistics.duckdb"

        build_biostatistics_duckdb(manifest, mlruns, db_path)

        import duckdb

        conn = duckdb.connect(str(db_path), read_only=True)
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main'"
            ).fetchall()
        ]
        conn.close()

        assert len(tables) == 7
        for expected in BIOSTATISTICS_TABLES:
            assert expected in tables

    def test_runs_table_has_experiment_columns(self, tmp_path: Path) -> None:
        mlruns = _create_mock_mlruns_with_metrics(tmp_path, ["dice_ce"], n_folds=1)
        manifest = discover_source_runs(mlruns, ["test_experiment"])
        db_path = tmp_path / "biostatistics.duckdb"

        build_biostatistics_duckdb(manifest, mlruns, db_path)

        import duckdb

        conn = duckdb.connect(str(db_path), read_only=True)
        columns = [
            row[0]
            for row in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'runs'"
            ).fetchall()
        ]
        conn.close()

        assert "experiment_name" in columns
        assert "experiment_id" in columns

    def test_per_volume_metrics_table_exists(self, tmp_path: Path) -> None:
        mlruns = _create_mock_mlruns_with_metrics(
            tmp_path, ["dice_ce"], n_folds=1, n_volumes=3
        )
        manifest = discover_source_runs(mlruns, ["test_experiment"])
        db_path = tmp_path / "biostatistics.duckdb"

        build_biostatistics_duckdb(manifest, mlruns, db_path)

        import duckdb

        conn = duckdb.connect(str(db_path), read_only=True)
        rows = conn.execute("SELECT COUNT(*) FROM per_volume_metrics").fetchone()
        conn.close()

        assert rows is not None
        assert rows[0] > 0

    def test_extraction_from_mock_mlruns(self, tmp_path: Path) -> None:
        mlruns = _create_mock_mlruns_with_metrics(
            tmp_path, ["dice_ce", "tversky"], n_folds=2
        )
        manifest = discover_source_runs(mlruns, ["test_experiment"])
        db_path = tmp_path / "biostatistics.duckdb"

        build_biostatistics_duckdb(manifest, mlruns, db_path)

        import duckdb

        conn = duckdb.connect(str(db_path), read_only=True)
        run_count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()
        param_count = conn.execute("SELECT COUNT(*) FROM params").fetchone()
        conn.close()

        assert run_count is not None and run_count[0] == 4  # 2 conditions x 2 folds
        assert param_count is not None and param_count[0] > 0

    def test_idempotent_rebuild(self, tmp_path: Path) -> None:
        mlruns = _create_mock_mlruns_with_metrics(tmp_path, ["dice_ce"], n_folds=1)
        manifest = discover_source_runs(mlruns, ["test_experiment"])
        db_path = tmp_path / "biostatistics.duckdb"

        # Build twice — should not fail or duplicate data
        build_biostatistics_duckdb(manifest, mlruns, db_path)
        build_biostatistics_duckdb(manifest, mlruns, db_path)

        import duckdb

        conn = duckdb.connect(str(db_path), read_only=True)
        run_count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()
        conn.close()

        assert run_count is not None and run_count[0] == 1

    def test_parquet_export_alongside_duckdb(self, tmp_path: Path) -> None:
        mlruns = _create_mock_mlruns_with_metrics(tmp_path, ["dice_ce"], n_folds=1)
        manifest = discover_source_runs(mlruns, ["test_experiment"])
        db_path = tmp_path / "biostatistics.duckdb"

        build_biostatistics_duckdb(manifest, mlruns, db_path)

        parquet_dir = tmp_path / "parquet"
        export_parquet(db_path, parquet_dir)

        parquet_files = list(parquet_dir.glob("*.parquet"))
        assert len(parquet_files) == 7  # One per table
