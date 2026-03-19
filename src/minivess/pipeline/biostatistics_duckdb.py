"""DuckDB schema and extraction for the biostatistics flow.

Builds a biostatistics-specific DuckDB database from a SourceRunManifest,
extending the base duckdb_extraction.py schema with per-volume metrics
and ensemble member tables. Also provides Parquet export.

Pure functions — no Prefect, no Docker dependency.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

BIOSTATISTICS_TABLES = [
    "runs",
    "params",
    "eval_metrics",
    "training_metrics",
    "champion_tags",
    "per_volume_metrics",
    "ensemble_members",
]

# ---------------------------------------------------------------------------
# DDL statements
# ---------------------------------------------------------------------------

_DDL_RUNS = """
    CREATE TABLE IF NOT EXISTS runs (
        run_id VARCHAR PRIMARY KEY,
        experiment_id VARCHAR,
        experiment_name VARCHAR,
        loss_function VARCHAR,
        fold_id INTEGER,
        model_family VARCHAR,
        status VARCHAR,
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

_DDL_CHAMPION_TAGS = """
    CREATE TABLE IF NOT EXISTS champion_tags (
        run_id VARCHAR,
        tag_key VARCHAR,
        tag_value VARCHAR,
        PRIMARY KEY (run_id, tag_key)
    )
"""

_DDL_PER_VOLUME_METRICS = """
    CREATE TABLE IF NOT EXISTS per_volume_metrics (
        run_id VARCHAR,
        fold_id INTEGER,
        volume_id VARCHAR,
        metric_name VARCHAR,
        metric_value DOUBLE,
        PRIMARY KEY (run_id, fold_id, volume_id, metric_name)
    )
"""

_DDL_ENSEMBLE_MEMBERS = """
    CREATE TABLE IF NOT EXISTS ensemble_members (
        ensemble_run_id VARCHAR,
        member_run_id VARCHAR,
        PRIMARY KEY (ensemble_run_id, member_run_id)
    )
"""

_ALL_DDL = [
    _DDL_RUNS,
    _DDL_PARAMS,
    _DDL_EVAL_METRICS,
    _DDL_TRAINING_METRICS,
    _DDL_CHAMPION_TAGS,
    _DDL_PER_VOLUME_METRICS,
    _DDL_ENSEMBLE_MEMBERS,
]

# CI variant suffixes — skip these as standalone metrics
_CI_SUFFIXES = ("_ci_level", "_ci95_lo", "_ci95_hi")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_biostatistics_duckdb(
    manifest: Any,
    mlruns_dir: Path,
    db_path: Path,
) -> Path:
    """Build a biostatistics DuckDB database from discovered source runs.

    Full rebuild every time (idempotent — drops and recreates tables).
    The dataset is small (~840 metric rows), so full rebuild is fast.

    Parameters
    ----------
    manifest:
        SourceRunManifest with discovered runs.
    mlruns_dir:
        Path to the mlruns directory.
    db_path:
        Output path for the DuckDB file.

    Returns
    -------
    Path to the created DuckDB file.
    """
    import duckdb

    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(db_path))

    # Drop all tables for idempotent rebuild
    for table in BIOSTATISTICS_TABLES:
        conn.execute(f"DROP TABLE IF EXISTS {table}")  # noqa: S608

    # Create schema
    for ddl in _ALL_DDL:
        conn.execute(ddl)

    for run in manifest.runs:
        _insert_run(conn, run, mlruns_dir)

    conn.close()

    logger.info(
        "Built biostatistics DuckDB at %s with %d runs",
        db_path,
        len(manifest.runs),
    )
    return db_path


def export_parquet(db_path: Path, output_dir: Path) -> list[Path]:
    """Export all DuckDB tables to Parquet files.

    Parameters
    ----------
    db_path:
        Path to the DuckDB file.
    output_dir:
        Directory for Parquet files.

    Returns
    -------
    List of created Parquet file paths.
    """
    import duckdb

    output_dir.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path), read_only=True)

    paths: list[Path] = []
    for table in BIOSTATISTICS_TABLES:
        parquet_path = output_dir / f"{table}.parquet"
        conn.execute(
            f"COPY {table} TO '{parquet_path}' (FORMAT PARQUET)"  # noqa: S608
        )
        paths.append(parquet_path)

    conn.close()
    logger.info("Exported %d Parquet files to %s", len(paths), output_dir)
    return paths


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _insert_run(conn: Any, run: Any, mlruns_dir: Path) -> None:
    """Insert a single run and its associated data into DuckDB."""
    # Find the run directory
    exp_dir = mlruns_dir / run.experiment_id
    run_dir = exp_dir / run.run_id

    # Read tags
    tags = _read_tags(run_dir)
    model_family = tags.get("model_family", "unknown")
    start_time = tags.get("started_at", "")

    # Insert into runs table
    conn.execute(
        "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            run.run_id,
            run.experiment_id,
            run.experiment_name,
            run.loss_function,
            run.fold_id,
            model_family,
            run.status,
            start_time,
        ],
    )

    # Insert params
    params = _read_params(run_dir)
    for param_name, param_value in params.items():
        conn.execute(
            "INSERT INTO params VALUES (?, ?, ?)",
            [run.run_id, param_name, str(param_value)],
        )

    # Insert metrics
    metrics = _read_metrics(run_dir)
    for metric_name, value in metrics.items():
        # Classify the metric
        if _is_per_volume_metric(metric_name):
            fold_id, volume_id, base_metric = _parse_per_volume_metric(metric_name)
            if fold_id is not None:
                conn.execute(
                    "INSERT INTO per_volume_metrics VALUES (?, ?, ?, ?, ?)",
                    [run.run_id, fold_id, volume_id, base_metric, value],
                )
        elif _is_eval_fold_metric(metric_name):
            parsed = _parse_eval_fold_metric(metric_name)
            if parsed is not None:
                fold_id_val, base_metric = parsed
                # Skip CI variant metrics
                if not any(base_metric.endswith(s) for s in _CI_SUFFIXES):
                    conn.execute(
                        "INSERT INTO eval_metrics VALUES (?, ?, ?, ?, ?, ?, ?)",
                        [run.run_id, fold_id_val, base_metric, value, None, None, None],
                    )
        else:
            # Training/validation metric
            conn.execute(
                "INSERT INTO training_metrics VALUES (?, ?, ?)",
                [run.run_id, metric_name, value],
            )

    # Insert champion tags
    for tag_key, tag_value in tags.items():
        if tag_key.startswith("champion_"):
            conn.execute(
                "INSERT INTO champion_tags VALUES (?, ?, ?)",
                [run.run_id, tag_key, tag_value],
            )


def _read_tags(run_dir: Path) -> dict[str, str]:
    """Read MLflow run tags from the tags directory."""
    tags_dir = run_dir / "tags"
    if not tags_dir.is_dir():
        return {}
    result: dict[str, str] = {}
    for tag_file in tags_dir.iterdir():
        if tag_file.is_file():
            result[tag_file.name] = tag_file.read_text(encoding="utf-8").strip()
    return result


def _read_params(run_dir: Path) -> dict[str, str]:
    """Read MLflow run parameters from the params directory."""
    params_dir = run_dir / "params"
    if not params_dir.is_dir():
        return {}
    result: dict[str, str] = {}
    for param_file in params_dir.iterdir():
        if param_file.is_file():
            result[param_file.name] = param_file.read_text(encoding="utf-8").strip()
    return result


def _read_metrics(run_dir: Path) -> dict[str, float]:
    """Read MLflow metrics (last value from each metric file).

    MLflow metric files contain lines of 'timestamp value step'.
    We take the last line's value.
    """
    metrics_dir = run_dir / "metrics"
    if not metrics_dir.is_dir():
        return {}
    result: dict[str, float] = {}
    for metric_file in metrics_dir.rglob("*"):
        if not metric_file.is_file():
            continue
        try:
            content = metric_file.read_text(encoding="utf-8").strip()
            if not content:
                continue
            last_line = content.splitlines()[-1]
            parts = last_line.split()
            if len(parts) >= 2:
                rel = metric_file.relative_to(metrics_dir)
                metric_key = "/".join(rel.parts)
                result[metric_key] = float(parts[1])
        except (ValueError, IndexError, OSError):
            logger.debug("Could not read metric %s", metric_file.name)
    return result


def _is_per_volume_metric(metric_name: str) -> bool:
    """Check if metric name matches eval/{fold}/vol/{id}/{metric} pattern."""
    parts = metric_name.split("/")
    return len(parts) >= 5 and parts[0] == "eval" and parts[2] == "vol"


def _is_eval_fold_metric(metric_name: str) -> bool:
    """Check if metric name matches eval/{fold}/{metric} pattern (not per-volume)."""
    parts = metric_name.split("/")
    if len(parts) < 3 or parts[0] != "eval":
        return False
    if not parts[1].isdigit():
        return False
    return not (len(parts) >= 4 and parts[2] == "vol")


def _parse_per_volume_metric(metric_name: str) -> tuple[int | None, str, str]:
    """Parse eval/{fold}/vol/{id}/{metric} into (fold_id, volume_id, base_metric)."""
    parts = metric_name.split("/")
    if len(parts) < 5 or parts[0] != "eval" or parts[2] != "vol":
        return None, "", ""
    if not parts[1].isdigit():
        return None, "", ""
    return int(parts[1]), parts[3], parts[4]


def _parse_eval_fold_metric(metric_name: str) -> tuple[int, str] | None:
    """Parse eval/{fold}/{metric} into (fold_id, base_metric)."""
    parts = metric_name.split("/")
    if len(parts) < 3 or parts[0] != "eval":
        return None
    if not parts[1].isdigit():
        return None
    return int(parts[1]), parts[2]
