"""DuckDB extraction from MLflow runs for fast SQL analytics.

Extracts training run metadata, metrics, and parameters into DuckDB
tables for efficient querying during academic report generation.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema creation helpers
# ---------------------------------------------------------------------------

_DDL_RUNS = """
    CREATE TABLE IF NOT EXISTS runs (
        run_id VARCHAR PRIMARY KEY,
        loss_function VARCHAR,
        model_family VARCHAR,
        num_folds INTEGER,
        start_time VARCHAR
    )
"""

_DDL_PARAMS = """
    CREATE TABLE IF NOT EXISTS params (
        run_id VARCHAR,
        param_name VARCHAR,
        param_value VARCHAR,
        PRIMARY KEY (run_id, param_name)
    )
"""

_DDL_EVAL_METRICS = """
    CREATE TABLE IF NOT EXISTS eval_metrics (
        run_id VARCHAR,
        fold_id INTEGER,
        metric_name VARCHAR,
        point_estimate DOUBLE,
        ci_lower DOUBLE,
        ci_upper DOUBLE,
        ci_level DOUBLE,
        PRIMARY KEY (run_id, fold_id, metric_name)
    )
"""

_DDL_TRAINING_METRICS = """
    CREATE TABLE IF NOT EXISTS training_metrics (
        run_id VARCHAR,
        metric_name VARCHAR,
        last_value DOUBLE,
        PRIMARY KEY (run_id, metric_name)
    )
"""

# Suffixes that indicate a CI variant of an eval metric (not a point estimate)
_CI_SUFFIXES = ("_ci_level", "_ci_lower", "_ci_upper")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_runs_to_duckdb(
    mlruns_dir: Path,
    experiment_id: str,
    db_path: Path | None = None,
) -> Any:
    """Extract MLflow runs into a DuckDB database.

    Creates four tables:

    - **runs**: run_id, loss_function, model_family, num_folds, start_time
    - **params**: run_id, param_name, param_value
    - **eval_metrics**: run_id, fold_id, metric_name, point_estimate,
      ci_lower, ci_upper, ci_level
    - **training_metrics**: run_id, metric_name, last_value

    Only *production* runs (those with all three evaluation folds complete)
    are extracted.  See
    :func:`~minivess.pipeline.mlruns_inspector.get_production_runs` for the
    production-run definition.

    Parameters
    ----------
    mlruns_dir:
        Path to the root ``mlruns/`` directory.
    experiment_id:
        MLflow experiment ID string (e.g. ``"843896622863223169"``).
    db_path:
        Optional path for a persistent DuckDB database file.  When ``None``
        an in-memory database is used.

    Returns
    -------
    duckdb.DuckDBPyConnection
        Open connection with all tables populated.
    """
    import duckdb

    from minivess.pipeline.mlruns_inspector import (
        get_production_runs,
        get_run_metrics_list,
        get_run_params,
        get_run_tags,
        read_metric_last_value,
    )

    conn_str = str(db_path) if db_path is not None else ":memory:"
    db = duckdb.connect(conn_str)

    # Create schema
    db.execute(_DDL_RUNS)
    db.execute(_DDL_PARAMS)
    db.execute(_DDL_EVAL_METRICS)
    db.execute(_DDL_TRAINING_METRICS)

    production_runs = get_production_runs(mlruns_dir, experiment_id)
    logger.info(
        "Extracting %d production run(s) from experiment %s",
        len(production_runs),
        experiment_id,
    )

    for run_id in production_runs:
        tags = get_run_tags(mlruns_dir, experiment_id, run_id)
        params = get_run_params(mlruns_dir, experiment_id, run_id)
        metrics_list = get_run_metrics_list(mlruns_dir, experiment_id, run_id)

        # --- runs row ---
        try:
            num_folds_raw = tags.get("num_folds", "3")
            num_folds = int(num_folds_raw)
        except ValueError:
            num_folds = 3

        db.execute(
            "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?)",
            [
                run_id,
                tags.get("loss_function", ""),
                tags.get("model_family", ""),
                num_folds,
                tags.get("started_at", ""),
            ],
        )

        # --- params rows ---
        for param_name, param_value in params.items():
            db.execute(
                "INSERT OR REPLACE INTO params VALUES (?, ?, ?)",
                [run_id, param_name, str(param_value)],
            )

        # --- metric rows ---
        metrics_set = set(metrics_list)
        for metric_name in metrics_list:
            if metric_name.startswith("eval_fold"):
                # Delegate to the fold-aware inserter
                _parse_and_insert_eval_metric(
                    db,
                    mlruns_dir,
                    experiment_id,
                    run_id,
                    metric_name,
                    metrics_set,
                )
            elif not metric_name.startswith("eval_"):
                # Non-eval metric: training / validation epoch curve last value
                try:
                    value = read_metric_last_value(
                        mlruns_dir, experiment_id, run_id, metric_name
                    )
                    db.execute(
                        "INSERT OR REPLACE INTO training_metrics VALUES (?, ?, ?)",
                        [run_id, metric_name, value],
                    )
                except (ValueError, FileNotFoundError, OSError):
                    logger.debug(
                        "Could not read training metric %s for run %s",
                        metric_name,
                        run_id,
                    )

    return db


def query_cross_loss_means(db: Any) -> list[tuple[Any, ...]]:
    """Query mean eval metrics aggregated per loss function.

    Returns rows of ``(loss_function, metric_name, mean_value, std_value,
    n_folds)``, ordered by loss function then metric name.

    Parameters
    ----------
    db:
        Open DuckDB connection returned by :func:`extract_runs_to_duckdb`.

    Returns
    -------
    list[tuple]
        List of ``(loss_function, metric_name, mean_value, std_value,
        n_folds)`` tuples.
    """
    result: list[tuple[Any, ...]] = db.execute("""
        SELECT
            r.loss_function,
            em.metric_name,
            AVG(em.point_estimate)      AS mean_value,
            STDDEV_SAMP(em.point_estimate) AS std_value,
            COUNT(*)                    AS n_folds
        FROM eval_metrics em
        JOIN runs r ON em.run_id = r.run_id
        GROUP BY r.loss_function, em.metric_name
        ORDER BY r.loss_function, em.metric_name
    """).fetchall()
    return result


def query_best_per_metric(db: Any) -> list[tuple[Any, ...]]:
    """Query the best-performing loss function for each eval metric.

    For metrics where lower is better (``measured_masd``), the loss with the
    lowest mean is returned.  For all other metrics the loss with the highest
    mean is returned.

    Parameters
    ----------
    db:
        Open DuckDB connection returned by :func:`extract_runs_to_duckdb`.

    Returns
    -------
    list[tuple]
        List of ``(loss_function, metric_name, mean_value)`` tuples, one per
        metric, ordered by metric name.
    """
    result: list[tuple[Any, ...]] = db.execute("""
        WITH aggregated AS (
            SELECT
                r.loss_function,
                em.metric_name,
                AVG(em.point_estimate) AS mean_value
            FROM eval_metrics em
            JOIN runs r ON em.run_id = r.run_id
            GROUP BY r.loss_function, em.metric_name
        ),
        ranked AS (
            SELECT
                loss_function,
                metric_name,
                mean_value,
                ROW_NUMBER() OVER (
                    PARTITION BY metric_name
                    ORDER BY CASE
                        WHEN metric_name = 'measured_masd' THEN  mean_value
                        ELSE                                     -mean_value
                    END
                ) AS rank
            FROM aggregated
        )
        SELECT loss_function, metric_name, mean_value
        FROM ranked
        WHERE rank = 1
        ORDER BY metric_name
    """).fetchall()
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_and_insert_eval_metric(
    db: Any,
    mlruns_dir: Path,
    experiment_id: str,
    run_id: str,
    metric_name: str,
    all_metrics: set[str],
) -> None:
    """Parse an ``eval_fold{i}_{name}`` metric file and insert into DuckDB.

    CI variant metrics (``_ci_lower``, ``_ci_upper``, ``_ci_level``) are
    skipped here — they are read as side-effects when the corresponding point
    estimate metric is processed.

    Parameters
    ----------
    db:
        Open DuckDB connection.
    mlruns_dir:
        Path to the root ``mlruns/`` directory.
    experiment_id:
        MLflow experiment ID string.
    run_id:
        32-character hexadecimal run ID.
    metric_name:
        Full metric name to process (e.g. ``"eval_fold0_dsc"``).
    all_metrics:
        Set of all metric names present for this run (used for CI lookup).
    """
    from minivess.pipeline.mlruns_inspector import read_metric_last_value

    # Pattern: eval_fold{i}_{base_metric}
    match = re.match(r"eval_fold(\d+)_(.+)", metric_name)
    if not match:
        return

    fold_id = int(match.group(1))
    base_metric = match.group(2)

    # Skip CI variant metrics — handled when their point estimate is processed
    if base_metric.endswith(_CI_SUFFIXES):
        return

    try:
        point_estimate = read_metric_last_value(
            mlruns_dir, experiment_id, run_id, metric_name
        )
    except (ValueError, FileNotFoundError, OSError):
        logger.debug("Could not read eval metric %s for run %s", metric_name, run_id)
        return

    # Attempt to read CI companions
    ci_lower = ci_upper = ci_level = float("nan")

    _ci_map = {
        "_ci_lower": "ci_lower",
        "_ci_upper": "ci_upper",
        "_ci_level": "ci_level",
    }
    for suffix, target_name in _ci_map.items():
        ci_metric = f"eval_fold{fold_id}_{base_metric}{suffix}"
        if ci_metric in all_metrics:
            try:
                val = read_metric_last_value(
                    mlruns_dir, experiment_id, run_id, ci_metric
                )
                if target_name == "ci_lower":
                    ci_lower = val
                elif target_name == "ci_upper":
                    ci_upper = val
                elif target_name == "ci_level":
                    ci_level = val
            except (ValueError, FileNotFoundError, OSError):
                pass

    db.execute(
        "INSERT OR REPLACE INTO eval_metrics VALUES (?, ?, ?, ?, ?, ?, ?)",
        [run_id, fold_id, base_metric, point_estimate, ci_lower, ci_upper, ci_level],
    )
