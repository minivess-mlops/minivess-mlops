"""Tests for DuckDB contract (Phase 2).

2.1: build_per_volume_data_from_duckdb reads DuckDB, not mlruns
2.2: Fixture DuckDB with synthetic data and known outcomes
2.3: Schema contract tests (columns, types, non-null)
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pytest

from minivess.config.biostatistics_config import BiostatisticsConfig
from minivess.pipeline.biostatistics_duckdb import (
    _ALL_DDL,
    BIOSTATISTICS_TABLES,
)

_CFG = BiostatisticsConfig()


# ---------------------------------------------------------------------------
# Fixture: synthetic DuckDB with known statistical properties
# ---------------------------------------------------------------------------


@pytest.fixture()
def fixture_db(tmp_path: Path) -> Path:
    """Create a synthetic DuckDB for testing.

    4 conditions: dynunet__dice_ce, dynunet__cbdice_cldice (each × 2 post-training)
    3 folds, 23 volumes per fold.
    Known effect: cbdice_cldice is 0.03 DSC better than dice_ce.
    """
    db_path = tmp_path / "fixture.duckdb"
    conn = duckdb.connect(str(db_path))

    for ddl in _ALL_DDL:
        conn.execute(ddl)

    rng = np.random.default_rng(42)

    conditions = [
        ("run_dice_ce_none_f0", "dice_ce", 0, "none"),
        ("run_dice_ce_none_f1", "dice_ce", 1, "none"),
        ("run_dice_ce_none_f2", "dice_ce", 2, "none"),
        ("run_cbdice_none_f0", "cbdice_cldice", 0, "none"),
        ("run_cbdice_none_f1", "cbdice_cldice", 1, "none"),
        ("run_cbdice_none_f2", "cbdice_cldice", 2, "none"),
    ]

    for run_id, loss, fold_id, pt_method in conditions:
        conn.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                run_id,
                "exp_001",
                "smoke_mini_evaluation",
                loss,
                fold_id,
                "dynunet",
                False,
                "FINISHED",
                "2026-03-29T00:00:00",
                pt_method,
                "none",
                "per_loss_single_best",
                False,
            ],
        )

        # Per-volume metrics: 23 volumes per fold
        base_dsc = 0.80 if loss == "cbdice_cldice" else 0.77
        for vol_idx in range(23):
            vol_id = f"mv{vol_idx:02d}"
            dsc_val = float(rng.normal(base_dsc, 0.03))
            cldice_val = float(rng.normal(base_dsc + 0.05, 0.03))
            masd_val = float(rng.exponential(1.5))
            for metric, val in [("dsc", dsc_val), ("cldice", cldice_val), ("masd", masd_val)]:
                conn.execute(
                    "INSERT INTO per_volume_metrics VALUES (?, ?, ?, ?, ?)",
                    [run_id, fold_id, vol_id, metric, val],
                )

    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# Task 2.1: DuckDB-only data loading
# ---------------------------------------------------------------------------


class TestDuckDBOnlyDataLoading:
    """build_per_volume_data_from_duckdb must exist and return correct structure."""

    def test_function_exists(self) -> None:
        from minivess.pipeline.biostatistics_duckdb import (
            build_per_volume_data_from_duckdb,
        )

        assert callable(build_per_volume_data_from_duckdb)

    def test_returns_nested_dict(self, fixture_db: Path) -> None:
        from minivess.pipeline.biostatistics_duckdb import (
            build_per_volume_data_from_duckdb,
        )

        data = build_per_volume_data_from_duckdb(fixture_db, metrics=["dsc", "cldice"])
        assert isinstance(data, dict)
        assert "dsc" in data
        assert "cldice" in data

    def test_conditions_present(self, fixture_db: Path) -> None:
        from minivess.pipeline.biostatistics_duckdb import (
            build_per_volume_data_from_duckdb,
        )

        data = build_per_volume_data_from_duckdb(fixture_db, metrics=["dsc"])
        conditions = list(data["dsc"].keys())
        assert len(conditions) >= 2  # At least dice_ce and cbdice_cldice

    def test_fold_structure_preserved(self, fixture_db: Path) -> None:
        from minivess.pipeline.biostatistics_duckdb import (
            build_per_volume_data_from_duckdb,
        )

        data = build_per_volume_data_from_duckdb(fixture_db, metrics=["dsc"])
        for _cond, folds in data["dsc"].items():
            assert isinstance(folds, dict)
            for fold_id, scores in folds.items():
                assert isinstance(fold_id, int)
                assert isinstance(scores, np.ndarray)
                assert len(scores) == 23  # 23 volumes per fold

    def test_23_volumes_per_fold(self, fixture_db: Path) -> None:
        from minivess.pipeline.biostatistics_duckdb import (
            build_per_volume_data_from_duckdb,
        )

        data = build_per_volume_data_from_duckdb(fixture_db, metrics=["dsc"])
        for _cond, folds in data["dsc"].items():
            for _fold, scores in folds.items():
                assert len(scores) == 23


# ---------------------------------------------------------------------------
# Task 2.3: Schema contract tests
# ---------------------------------------------------------------------------


class TestDuckDBSchemaContract:
    """All 8 tables must exist with correct columns and types."""

    def test_all_tables_exist(self, fixture_db: Path) -> None:
        conn = duckdb.connect(str(fixture_db), read_only=True)
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = {row[0] for row in tables}
        conn.close()
        for expected in BIOSTATISTICS_TABLES:
            assert expected in table_names, f"Missing table: {expected}"

    def test_runs_table_columns(self, fixture_db: Path) -> None:
        conn = duckdb.connect(str(fixture_db), read_only=True)
        cols = conn.execute("DESCRIBE runs").fetchall()
        col_names = {row[0] for row in cols}
        conn.close()
        required = {
            "run_id", "experiment_id", "experiment_name", "loss_function",
            "fold_id", "model_family", "with_aux_calib", "status",
            "post_training_method", "recalibration", "ensemble_strategy", "is_zero_shot",
        }
        for col in required:
            assert col in col_names, f"Missing column in runs: {col}"

    def test_per_volume_metrics_columns(self, fixture_db: Path) -> None:
        conn = duckdb.connect(str(fixture_db), read_only=True)
        cols = conn.execute("DESCRIBE per_volume_metrics").fetchall()
        col_names = {row[0] for row in cols}
        conn.close()
        required = {"run_id", "fold_id", "volume_id", "metric_name", "metric_value"}
        for col in required:
            assert col in col_names, f"Missing column in per_volume_metrics: {col}"

    def test_per_volume_metrics_row_count(self, fixture_db: Path) -> None:
        conn = duckdb.connect(str(fixture_db), read_only=True)
        count = conn.execute("SELECT COUNT(*) FROM per_volume_metrics").fetchone()[0]
        conn.close()
        # 6 runs × 23 volumes × 3 metrics = 414
        assert count == 414

    def test_runs_not_empty(self, fixture_db: Path) -> None:
        conn = duckdb.connect(str(fixture_db), read_only=True)
        count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        conn.close()
        assert count == 6

    def test_no_null_run_ids(self, fixture_db: Path) -> None:
        conn = duckdb.connect(str(fixture_db), read_only=True)
        null_count = conn.execute(
            "SELECT COUNT(*) FROM per_volume_metrics WHERE run_id IS NULL"
        ).fetchone()[0]
        conn.close()
        assert null_count == 0


# ---------------------------------------------------------------------------
# Task 2.2: Simulation validation — pipeline recovers known effect
# ---------------------------------------------------------------------------


class TestSimulationValidation:
    """Verify statistical pipeline detects the known 0.03 DSC effect."""

    def test_known_effect_detected(self, fixture_db: Path) -> None:
        """cbdice_cldice is 0.03 DSC better than dice_ce — pipeline should detect."""
        from minivess.pipeline.biostatistics_duckdb import (
            build_per_volume_data_from_duckdb,
        )
        from minivess.pipeline.biostatistics_statistics import (
            stratified_permutation_test,
        )

        data = build_per_volume_data_from_duckdb(fixture_db, metrics=["dsc"])
        conditions = sorted(data["dsc"].keys())
        assert len(conditions) >= 2

        # Find dice_ce and cbdice_cldice conditions
        cond_dice = [c for c in conditions if "dice_ce" in c and "cbdice" not in c]
        cond_cbdice = [c for c in conditions if "cbdice" in c]

        if cond_dice and cond_cbdice:
            result = stratified_permutation_test(
                fold_data_a=data["dsc"][cond_dice[0]],
                fold_data_b=data["dsc"][cond_cbdice[0]],
                n_permutations=499,
                seed=_CFG.seed,
            )
            # With 23 volumes × 3 folds and effect=0.03, should detect
            assert result.p_value < _CFG.alpha, (
                f"Failed to detect known 0.03 DSC effect: p={result.p_value}"
            )
