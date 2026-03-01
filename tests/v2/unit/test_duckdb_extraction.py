"""Unit and integration tests for DuckDB MLflow extraction pipeline.

Unit tests use a synthetic mock mlruns filesystem built in ``tmp_path``.
Integration tests use the real ``mlruns/`` directory in the repository root
and are guarded with ``pytest.mark.integration`` + ``skipif`` so they are
skipped in CI when the mlruns data is absent.

MLflow filesystem layout assumed:
    mlruns/<experiment_id>/<run_id>/
        tags/<key>          — plain text
        metrics/<key>       — lines "<timestamp> <value> <step>"
        params/<key>        — plain text
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from minivess.pipeline.duckdb_extraction import (
    _parse_and_insert_eval_metric,
    extract_runs_to_duckdb,
    query_best_per_metric,
    query_champion_runs,
    query_cross_loss_means,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EXP_ID = "unit_test_exp"
_REAL_EXP_ID = "843896622863223169"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_REAL_MLRUNS = _REPO_ROOT / "mlruns"
_HAS_REAL_MLRUNS = (_REAL_MLRUNS / _REAL_EXP_ID).is_dir()


# ---------------------------------------------------------------------------
# Mock mlruns filesystem helpers
# ---------------------------------------------------------------------------


def _make_metric(
    run_dir: Path,
    name: str,
    value: float,
    step: int = 0,
    timestamp: int = 1_700_000_000,
) -> None:
    """Write a single-step metric file in the MLflow format."""
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / name).write_text(f"{timestamp} {value} {step}\n", encoding="utf-8")


def _make_tag(run_dir: Path, key: str, value: str) -> None:
    tags_dir = run_dir / "tags"
    tags_dir.mkdir(parents=True, exist_ok=True)
    (tags_dir / key).write_text(value, encoding="utf-8")


def _make_param(run_dir: Path, key: str, value: str) -> None:
    params_dir = run_dir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)
    (params_dir / key).write_text(value, encoding="utf-8")


def _make_production_run(
    mlruns_dir: Path,
    exp_id: str,
    run_id: str,
    loss_function: str = "dice_ce",
    model_family: str = "dynunet",
    num_folds: int = 3,
) -> Path:
    """Create a complete 3-fold production run with synthetic metrics."""
    run_dir = mlruns_dir / exp_id / run_id

    _make_tag(run_dir, "loss_function", loss_function)
    _make_tag(run_dir, "model_family", model_family)
    _make_tag(run_dir, "num_folds", str(num_folds))
    _make_tag(run_dir, "started_at", "2026-01-01T00:00:00+00:00")

    _make_param(run_dir, "batch_size", "2")
    _make_param(run_dir, "max_epochs", "100")
    _make_param(run_dir, "seed", "42")

    # 3 folds × 3 metrics + CI variants
    for fold in range(3):
        for metric_base, value in [
            ("dsc", 0.80),
            ("centreline_dsc", 0.82),
            ("measured_masd", 2.5),
        ]:
            _make_metric(run_dir, f"eval_fold{fold}_{metric_base}", value)
            _make_metric(
                run_dir, f"eval_fold{fold}_{metric_base}_ci_lower", value - 0.05
            )
            _make_metric(
                run_dir, f"eval_fold{fold}_{metric_base}_ci_upper", value + 0.05
            )
            _make_metric(run_dir, f"eval_fold{fold}_{metric_base}_ci_level", 0.95)

    # Training / validation metrics
    for name, val in [
        ("val_loss", 0.35),
        ("val_dice", 0.80),
        ("val_cldice", 0.82),
        ("val_masd", 2.5),
        ("val_compound_masd_cldice", 0.88),
    ]:
        _make_metric(run_dir, name, val)

    return run_dir


def _make_mock_mlruns(tmp_path: Path, n_runs: int = 4) -> tuple[Path, str]:
    """Create a mock mlruns directory with *n_runs* production runs."""
    mlruns_dir = tmp_path / "mlruns"
    mlruns_dir.mkdir()

    loss_functions = ["dice_ce", "cbdice_cldice", "dice_ce_cldice", "cldice"]
    for i in range(n_runs):
        lf = loss_functions[i % len(loss_functions)]
        _make_production_run(mlruns_dir, _EXP_ID, f"run_{i:02d}", loss_function=lf)

    return mlruns_dir, _EXP_ID


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_db(tmp_path: Path) -> Any:
    """Return a populated in-memory DuckDB from a mock mlruns tree."""
    mlruns_dir, exp_id = _make_mock_mlruns(tmp_path)
    return extract_runs_to_duckdb(mlruns_dir, exp_id)


@pytest.fixture()
def mock_db_two_runs(tmp_path: Path) -> Any:
    """Return DB with two runs having different loss functions."""
    mlruns_dir, exp_id = _make_mock_mlruns(tmp_path, n_runs=2)
    return extract_runs_to_duckdb(mlruns_dir, exp_id)


# ---------------------------------------------------------------------------
# Tests: table creation
# ---------------------------------------------------------------------------


class TestCreatesTables:
    """Verify the four expected tables are created."""

    def test_runs_table_exists(self, mock_db: Any) -> None:
        result = mock_db.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name = 'runs'"
        ).fetchone()
        assert result is not None

    def test_params_table_exists(self, mock_db: Any) -> None:
        result = mock_db.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_name = 'params'"
        ).fetchone()
        assert result is not None

    def test_eval_metrics_table_exists(self, mock_db: Any) -> None:
        result = mock_db.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_name = 'eval_metrics'"
        ).fetchone()
        assert result is not None

    def test_training_metrics_table_exists(self, mock_db: Any) -> None:
        result = mock_db.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_name = 'training_metrics'"
        ).fetchone()
        assert result is not None


# ---------------------------------------------------------------------------
# Tests: runs table population
# ---------------------------------------------------------------------------


class TestRunsTable:
    """Tests for the runs table contents."""

    def test_inserts_four_mock_runs(self, tmp_path: Path) -> None:
        mlruns_dir, exp_id = _make_mock_mlruns(tmp_path, n_runs=4)
        db = extract_runs_to_duckdb(mlruns_dir, exp_id)
        count = db.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        assert count == 4

    def test_loss_function_populated(self, mock_db: Any) -> None:
        rows = mock_db.execute(
            "SELECT DISTINCT loss_function FROM runs ORDER BY loss_function"
        ).fetchall()
        loss_functions = {r[0] for r in rows}
        assert "dice_ce" in loss_functions

    def test_model_family_populated(self, mock_db: Any) -> None:
        rows = mock_db.execute("SELECT DISTINCT model_family FROM runs").fetchall()
        assert len(rows) > 0
        assert rows[0][0] == "dynunet"

    def test_num_folds_is_integer(self, mock_db: Any) -> None:
        rows = mock_db.execute("SELECT DISTINCT num_folds FROM runs").fetchall()
        assert all(isinstance(r[0], int) for r in rows)
        assert rows[0][0] == 3

    def test_start_time_not_empty(self, mock_db: Any) -> None:
        rows = mock_db.execute("SELECT start_time FROM runs").fetchall()
        assert all(r[0] != "" for r in rows)

    def test_empty_experiment_gives_zero_rows(self, tmp_path: Path) -> None:
        mlruns_dir = tmp_path / "mlruns"
        mlruns_dir.mkdir()
        db = extract_runs_to_duckdb(mlruns_dir, "nonexistent_exp")
        count = db.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        assert count == 0


# ---------------------------------------------------------------------------
# Tests: params table
# ---------------------------------------------------------------------------


class TestParamsTable:
    """Tests for the params table contents."""

    def test_params_populated(self, mock_db: Any) -> None:
        count = mock_db.execute("SELECT COUNT(*) FROM params").fetchone()[0]
        assert count > 0

    def test_batch_size_param_present(self, mock_db: Any) -> None:
        rows = mock_db.execute(
            "SELECT param_value FROM params WHERE param_name = 'batch_size' LIMIT 1"
        ).fetchall()
        assert len(rows) > 0
        assert rows[0][0] == "2"

    def test_max_epochs_param_present(self, mock_db: Any) -> None:
        rows = mock_db.execute(
            "SELECT param_value FROM params WHERE param_name = 'max_epochs' LIMIT 1"
        ).fetchall()
        assert len(rows) > 0
        assert rows[0][0] == "100"

    def test_params_keyed_per_run(self, mock_db: Any) -> None:
        run_ids = mock_db.execute("SELECT DISTINCT run_id FROM params").fetchall()
        assert len(run_ids) == 4


# ---------------------------------------------------------------------------
# Tests: eval_metrics table
# ---------------------------------------------------------------------------


class TestEvalMetricsTable:
    """Tests for the eval_metrics table contents."""

    def test_eval_metrics_populated(self, mock_db: Any) -> None:
        count = mock_db.execute("SELECT COUNT(*) FROM eval_metrics").fetchone()[0]
        assert count > 0

    def test_three_folds_per_run(self, mock_db: Any) -> None:
        """Each run should have 3 distinct fold IDs."""
        rows = mock_db.execute(
            "SELECT run_id, COUNT(DISTINCT fold_id) as nf FROM eval_metrics "
            "GROUP BY run_id"
        ).fetchall()
        assert all(r[1] == 3 for r in rows)

    def test_dsc_metric_present(self, mock_db: Any) -> None:
        rows = mock_db.execute(
            "SELECT COUNT(*) FROM eval_metrics WHERE metric_name = 'dsc'"
        ).fetchone()
        assert rows[0] > 0

    def test_centreline_dsc_metric_present(self, mock_db: Any) -> None:
        rows = mock_db.execute(
            "SELECT COUNT(*) FROM eval_metrics WHERE metric_name = 'centreline_dsc'"
        ).fetchone()
        assert rows[0] > 0

    def test_measured_masd_metric_present(self, mock_db: Any) -> None:
        rows = mock_db.execute(
            "SELECT COUNT(*) FROM eval_metrics WHERE metric_name = 'measured_masd'"
        ).fetchone()
        assert rows[0] > 0

    def test_ci_lower_populated(self, mock_db: Any) -> None:
        rows = mock_db.execute(
            "SELECT ci_lower FROM eval_metrics WHERE metric_name = 'dsc' LIMIT 1"
        ).fetchone()
        assert rows is not None
        assert not math.isnan(rows[0])

    def test_ci_upper_populated(self, mock_db: Any) -> None:
        rows = mock_db.execute(
            "SELECT ci_upper FROM eval_metrics WHERE metric_name = 'dsc' LIMIT 1"
        ).fetchone()
        assert rows is not None
        assert not math.isnan(rows[0])

    def test_ci_level_populated(self, mock_db: Any) -> None:
        rows = mock_db.execute(
            "SELECT ci_level FROM eval_metrics WHERE metric_name = 'dsc' LIMIT 1"
        ).fetchone()
        assert rows is not None
        assert rows[0] == pytest.approx(0.95)

    def test_ci_variants_not_inserted_as_own_rows(self, mock_db: Any) -> None:
        """CI variant names (e.g. 'dsc_ci_lower') should NOT appear as metric_name."""
        rows = mock_db.execute(
            "SELECT metric_name FROM eval_metrics WHERE metric_name LIKE '%_ci_%'"
        ).fetchall()
        assert len(rows) == 0

    def test_point_estimate_within_bounds(self, mock_db: Any) -> None:
        rows = mock_db.execute(
            "SELECT point_estimate FROM eval_metrics WHERE metric_name = 'dsc'"
        ).fetchall()
        assert all(0.0 <= r[0] <= 1.0 for r in rows)


# ---------------------------------------------------------------------------
# Tests: training_metrics table
# ---------------------------------------------------------------------------


class TestTrainingMetricsTable:
    """Tests for the training_metrics table contents."""

    def test_training_metrics_populated(self, mock_db: Any) -> None:
        count = mock_db.execute("SELECT COUNT(*) FROM training_metrics").fetchone()[0]
        assert count > 0

    def test_val_loss_present(self, mock_db: Any) -> None:
        rows = mock_db.execute(
            "SELECT COUNT(*) FROM training_metrics WHERE metric_name = 'val_loss'"
        ).fetchone()
        assert rows[0] > 0

    def test_val_dice_present(self, mock_db: Any) -> None:
        rows = mock_db.execute(
            "SELECT COUNT(*) FROM training_metrics WHERE metric_name = 'val_dice'"
        ).fetchone()
        assert rows[0] > 0

    def test_val_compound_masd_cldice_present(self, mock_db: Any) -> None:
        rows = mock_db.execute(
            "SELECT COUNT(*) FROM training_metrics "
            "WHERE metric_name = 'val_compound_masd_cldice'"
        ).fetchone()
        assert rows[0] > 0

    def test_eval_metrics_not_in_training_table(self, mock_db: Any) -> None:
        """eval_fold* metrics must not appear in training_metrics."""
        rows = mock_db.execute(
            "SELECT COUNT(*) FROM training_metrics WHERE metric_name LIKE 'eval_%'"
        ).fetchone()
        assert rows[0] == 0


# ---------------------------------------------------------------------------
# Tests: query helpers
# ---------------------------------------------------------------------------


class TestQueryCrossLossMeans:
    """Tests for query_cross_loss_means."""

    def test_returns_results(self, mock_db: Any) -> None:
        results = query_cross_loss_means(mock_db)
        assert len(results) > 0

    def test_result_columns(self, mock_db: Any) -> None:
        """Each row has 5 columns: loss, metric, mean, std, n_folds."""
        results = query_cross_loss_means(mock_db)
        assert all(len(r) == 5 for r in results)

    def test_mean_values_in_valid_range(self, mock_db: Any) -> None:
        results = query_cross_loss_means(mock_db)
        for row in results:
            _loss, metric, mean, _std, _n = row
            if metric in ("dsc", "centreline_dsc"):
                assert 0.0 <= mean <= 1.0, f"DSC mean {mean} out of range"

    def test_n_folds_is_positive(self, mock_db: Any) -> None:
        results = query_cross_loss_means(mock_db)
        assert all(r[4] > 0 for r in results)


class TestQueryBestPerMetric:
    """Tests for query_best_per_metric."""

    def test_returns_results(self, mock_db: Any) -> None:
        results = query_best_per_metric(mock_db)
        assert len(results) > 0

    def test_result_columns(self, mock_db: Any) -> None:
        """Each row has 3 columns: loss, metric, mean_value."""
        results = query_best_per_metric(mock_db)
        assert all(len(r) == 3 for r in results)

    def test_one_winner_per_metric(self, tmp_path: Path) -> None:
        """Each metric should have exactly one best loss function."""
        mlruns_dir, exp_id = _make_mock_mlruns(tmp_path, n_runs=4)
        db = extract_runs_to_duckdb(mlruns_dir, exp_id)
        results = query_best_per_metric(db)
        metric_names = [r[1] for r in results]
        # No duplicates
        assert len(metric_names) == len(set(metric_names))

    def test_minimization_for_masd(self, tmp_path: Path) -> None:
        """The winner for measured_masd should be the run with the LOWEST mean."""
        mlruns_dir = tmp_path / "mlruns"
        mlruns_dir.mkdir()

        # run_low: masd = 1.0, run_high: masd = 5.0
        run_low = mlruns_dir / _EXP_ID / "run_low"
        _make_tag(run_low, "loss_function", "low_loss")
        _make_tag(run_low, "model_family", "dynunet")
        _make_tag(run_low, "num_folds", "3")
        _make_tag(run_low, "started_at", "")
        for fold in range(3):
            _make_metric(run_low, f"eval_fold{fold}_measured_masd", 1.0)
            _make_metric(run_low, f"eval_fold{fold}_dsc", 0.80)
            _make_metric(run_low, f"eval_fold{fold}_centreline_dsc", 0.80)

        run_high = mlruns_dir / _EXP_ID / "run_high"
        _make_tag(run_high, "loss_function", "high_loss")
        _make_tag(run_high, "model_family", "dynunet")
        _make_tag(run_high, "num_folds", "3")
        _make_tag(run_high, "started_at", "")
        for fold in range(3):
            _make_metric(run_high, f"eval_fold{fold}_measured_masd", 5.0)
            _make_metric(run_high, f"eval_fold{fold}_dsc", 0.75)
            _make_metric(run_high, f"eval_fold{fold}_centreline_dsc", 0.75)

        db = extract_runs_to_duckdb(mlruns_dir, _EXP_ID)
        results = query_best_per_metric(db)

        masd_winner = next(r for r in results if r[1] == "measured_masd")
        assert masd_winner[0] == "low_loss", (
            f"Expected 'low_loss' to win measured_masd but got {masd_winner[0]}"
        )

    def test_maximization_for_dsc(self, tmp_path: Path) -> None:
        """The winner for dsc should be the run with the HIGHEST mean."""
        mlruns_dir = tmp_path / "mlruns"
        mlruns_dir.mkdir()

        run_good = mlruns_dir / _EXP_ID / "run_good"
        _make_tag(run_good, "loss_function", "good_loss")
        _make_tag(run_good, "model_family", "dynunet")
        _make_tag(run_good, "num_folds", "3")
        _make_tag(run_good, "started_at", "")
        for fold in range(3):
            _make_metric(run_good, f"eval_fold{fold}_dsc", 0.90)
            _make_metric(run_good, f"eval_fold{fold}_centreline_dsc", 0.88)
            _make_metric(run_good, f"eval_fold{fold}_measured_masd", 2.0)

        run_bad = mlruns_dir / _EXP_ID / "run_bad"
        _make_tag(run_bad, "loss_function", "bad_loss")
        _make_tag(run_bad, "model_family", "dynunet")
        _make_tag(run_bad, "num_folds", "3")
        _make_tag(run_bad, "started_at", "")
        for fold in range(3):
            _make_metric(run_bad, f"eval_fold{fold}_dsc", 0.70)
            _make_metric(run_bad, f"eval_fold{fold}_centreline_dsc", 0.68)
            _make_metric(run_bad, f"eval_fold{fold}_measured_masd", 3.0)

        db = extract_runs_to_duckdb(mlruns_dir, _EXP_ID)
        results = query_best_per_metric(db)

        dsc_winner = next(r for r in results if r[1] == "dsc")
        assert dsc_winner[0] == "good_loss", (
            f"Expected 'good_loss' to win dsc but got {dsc_winner[0]}"
        )


# ---------------------------------------------------------------------------
# Tests: in-memory vs persistent DB
# ---------------------------------------------------------------------------


class TestDatabasePersistence:
    """Tests for in-memory and file-based database creation."""

    def test_in_memory_db(self, tmp_path: Path) -> None:
        """extract_runs_to_duckdb with no db_path creates a working in-memory DB."""
        mlruns_dir, exp_id = _make_mock_mlruns(tmp_path, n_runs=2)
        db = extract_runs_to_duckdb(mlruns_dir, exp_id, db_path=None)
        count = db.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        assert count == 2

    def test_persistent_db_creates_file(self, tmp_path: Path) -> None:
        """When db_path is given, a .duckdb file is created on disk."""
        mlruns_dir, exp_id = _make_mock_mlruns(tmp_path, n_runs=1)
        db_path = tmp_path / "test_analytics.duckdb"

        db = extract_runs_to_duckdb(mlruns_dir, exp_id, db_path=db_path)
        db.close()

        assert db_path.exists(), f"Expected DuckDB file at {db_path}"

    def test_persistent_db_is_queryable_after_reopen(self, tmp_path: Path) -> None:
        """Data written to a persistent DB can be re-read after reopening."""
        import duckdb

        mlruns_dir, exp_id = _make_mock_mlruns(tmp_path, n_runs=2)
        db_path = tmp_path / "reopen_test.duckdb"

        db = extract_runs_to_duckdb(mlruns_dir, exp_id, db_path=db_path)
        db.close()

        db2 = duckdb.connect(str(db_path))
        count = db2.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        db2.close()

        assert count == 2


# ---------------------------------------------------------------------------
# Tests: _parse_and_insert_eval_metric (unit)
# ---------------------------------------------------------------------------


class TestParseAndInsertEvalMetric:
    """Unit tests for the internal metric parser / inserter."""

    def test_skips_non_matching_names(self, tmp_path: Path) -> None:
        import duckdb

        from minivess.pipeline.duckdb_extraction import _DDL_EVAL_METRICS

        db = duckdb.connect(":memory:")
        db.execute(_DDL_EVAL_METRICS)

        mlruns_dir = tmp_path / "mlruns"
        mlruns_dir.mkdir()
        run_dir = mlruns_dir / _EXP_ID / "r0"
        (run_dir / "metrics").mkdir(parents=True)

        _parse_and_insert_eval_metric(db, mlruns_dir, _EXP_ID, "r0", "val_loss", set())
        count = db.execute("SELECT COUNT(*) FROM eval_metrics").fetchone()[0]
        assert count == 0, "val_loss should not be inserted into eval_metrics"

    def test_skips_ci_variant_metrics(self, tmp_path: Path) -> None:
        import duckdb

        from minivess.pipeline.duckdb_extraction import _DDL_EVAL_METRICS

        db = duckdb.connect(":memory:")
        db.execute(_DDL_EVAL_METRICS)

        mlruns_dir = tmp_path / "mlruns"
        mlruns_dir.mkdir()
        run_dir = mlruns_dir / _EXP_ID / "r0"
        (run_dir / "metrics").mkdir(parents=True)

        _parse_and_insert_eval_metric(
            db, mlruns_dir, _EXP_ID, "r0", "eval_fold0_dsc_ci_lower", set()
        )
        count = db.execute("SELECT COUNT(*) FROM eval_metrics").fetchone()[0]
        assert count == 0, "ci_lower variant should not be inserted"

    def test_inserts_valid_metric(self, tmp_path: Path) -> None:
        import duckdb

        from minivess.pipeline.duckdb_extraction import _DDL_EVAL_METRICS

        db = duckdb.connect(":memory:")
        db.execute(_DDL_EVAL_METRICS)

        mlruns_dir = tmp_path / "mlruns"
        run_dir = mlruns_dir / _EXP_ID / "r0"
        _make_metric(run_dir, "eval_fold1_dsc", 0.85)

        _parse_and_insert_eval_metric(
            db, mlruns_dir, _EXP_ID, "r0", "eval_fold1_dsc", set()
        )
        row = db.execute(
            "SELECT fold_id, metric_name, point_estimate FROM eval_metrics"
        ).fetchone()
        assert row is not None
        assert row[0] == 1
        assert row[1] == "dsc"
        assert row[2] == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# Tests: champion_tags table
# ---------------------------------------------------------------------------


class TestChampionTagsTable:
    """Tests for the champion_tags table extraction."""

    def test_champion_tags_table_exists(self, mock_db: Any) -> None:
        result = mock_db.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_name = 'champion_tags'"
        ).fetchone()
        assert result is not None

    def test_no_champion_tags_when_none_set(self, mock_db: Any) -> None:
        """Without champion tags, the table is empty."""
        count = mock_db.execute("SELECT COUNT(*) FROM champion_tags").fetchone()[0]
        assert count == 0

    def test_champion_tags_extracted(self, tmp_path: Path) -> None:
        """Champion tags are extracted into the champion_tags table."""
        mlruns_dir, exp_id = _make_mock_mlruns(tmp_path, n_runs=1)
        # Add champion tags to the run
        run_dir = mlruns_dir / exp_id / "run_00"
        _make_tag(run_dir, "champion_best_single_fold", "true")
        _make_tag(run_dir, "champion_metric_name", "dsc")
        _make_tag(run_dir, "champion_metric_value", "0.85")

        db = extract_runs_to_duckdb(mlruns_dir, exp_id)
        count = db.execute("SELECT COUNT(*) FROM champion_tags").fetchone()[0]
        assert count == 3

    def test_non_champion_tags_excluded(self, tmp_path: Path) -> None:
        """Only tags starting with 'champion_' end up in champion_tags."""
        mlruns_dir, exp_id = _make_mock_mlruns(tmp_path, n_runs=1)
        run_dir = mlruns_dir / exp_id / "run_00"
        _make_tag(run_dir, "champion_best_cv_mean", "true")

        db = extract_runs_to_duckdb(mlruns_dir, exp_id)
        # Should have 1 champion tag, no non-champion tags
        rows = db.execute(
            "SELECT tag_key FROM champion_tags WHERE tag_key NOT LIKE 'champion_%'"
        ).fetchall()
        assert len(rows) == 0

    def test_query_champion_runs_returns_results(self, tmp_path: Path) -> None:
        """query_champion_runs returns (run_id, loss_function, tag_key, tag_value)."""
        mlruns_dir, exp_id = _make_mock_mlruns(tmp_path, n_runs=2)
        run_dir = mlruns_dir / exp_id / "run_00"
        _make_tag(run_dir, "champion_best_single_fold", "true")
        _make_tag(run_dir, "champion_metric_value", "0.85")

        db = extract_runs_to_duckdb(mlruns_dir, exp_id)
        results = query_champion_runs(db)
        assert len(results) == 2
        # Each row: (run_id, loss_function, tag_key, tag_value)
        assert all(len(r) == 4 for r in results)

    def test_query_champion_runs_empty_when_no_tags(self, mock_db: Any) -> None:
        """query_champion_runs returns empty list when no champion tags exist."""
        results = query_champion_runs(mock_db)
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Integration tests against real mlruns/
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    not _HAS_REAL_MLRUNS,
    reason=f"Real mlruns not available at {_REAL_MLRUNS}",
)
class TestIntegrationRealMlruns:
    """Integration tests against the production mlruns directory."""

    def test_inserts_4_production_runs(self) -> None:
        db = extract_runs_to_duckdb(_REAL_MLRUNS, _REAL_EXP_ID)
        count = db.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        assert count == 4, f"Expected 4 production runs, got {count}"

    def test_params_populated_real(self) -> None:
        db = extract_runs_to_duckdb(_REAL_MLRUNS, _REAL_EXP_ID)
        count = db.execute("SELECT COUNT(*) FROM params").fetchone()[0]
        assert count > 0

    def test_eval_metrics_populated_real(self) -> None:
        db = extract_runs_to_duckdb(_REAL_MLRUNS, _REAL_EXP_ID)
        count = db.execute("SELECT COUNT(*) FROM eval_metrics").fetchone()[0]
        assert count > 0

    def test_training_metrics_populated_real(self) -> None:
        db = extract_runs_to_duckdb(_REAL_MLRUNS, _REAL_EXP_ID)
        count = db.execute("SELECT COUNT(*) FROM training_metrics").fetchone()[0]
        assert count > 0

    def test_query_cross_loss_means_real(self) -> None:
        db = extract_runs_to_duckdb(_REAL_MLRUNS, _REAL_EXP_ID)
        results = query_cross_loss_means(db)
        assert len(results) > 0
        # Each row: (loss_function, metric_name, mean_value, std, n_folds)
        for row in results:
            assert len(row) == 5

    def test_query_best_per_metric_real(self) -> None:
        db = extract_runs_to_duckdb(_REAL_MLRUNS, _REAL_EXP_ID)
        results = query_best_per_metric(db)
        assert len(results) > 0
        metric_names = [r[1] for r in results]
        # No duplicates — one winner per metric
        assert len(metric_names) == len(set(metric_names))

    def test_val_loss_in_training_metrics_real(self) -> None:
        db = extract_runs_to_duckdb(_REAL_MLRUNS, _REAL_EXP_ID)
        rows = db.execute(
            "SELECT COUNT(*) FROM training_metrics WHERE metric_name = 'val_loss'"
        ).fetchone()
        assert rows[0] == 4, "Expected one val_loss row per production run (4 total)"

    def test_four_distinct_loss_functions_real(self) -> None:
        db = extract_runs_to_duckdb(_REAL_MLRUNS, _REAL_EXP_ID)
        rows = db.execute(
            "SELECT DISTINCT loss_function FROM runs ORDER BY loss_function"
        ).fetchall()
        assert len(rows) == 4, f"Expected 4 distinct loss functions, got {rows}"

    def test_persistent_db_real(self, tmp_path: Path) -> None:
        db_path = tmp_path / "real_analytics.duckdb"
        db = extract_runs_to_duckdb(_REAL_MLRUNS, _REAL_EXP_ID, db_path=db_path)
        db.close()
        assert db_path.exists()
