"""DuckDB schema and extraction for the biostatistics flow.

Builds a biostatistics-specific DuckDB database from a SourceRunManifest,
extending the base duckdb_extraction.py schema with per-volume metrics
and ensemble member tables. Also provides Parquet export.

Pure functions — no Prefect, no Docker dependency.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

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
    "test_metrics",
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
        with_aux_calib BOOLEAN,
        status VARCHAR,
        start_time VARCHAR,
        post_training_method VARCHAR DEFAULT 'none',
        recalibration VARCHAR DEFAULT 'none',
        ensemble_strategy VARCHAR DEFAULT 'none',
        is_zero_shot BOOLEAN DEFAULT FALSE
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

_DDL_TEST_METRICS = """
    CREATE TABLE IF NOT EXISTS test_metrics (
        run_id VARCHAR,
        dataset_name VARCHAR,
        subset_name VARCHAR,
        metric_name VARCHAR,
        point_estimate DOUBLE,
        ci_lower DOUBLE,
        ci_upper DOUBLE,
        PRIMARY KEY (run_id, dataset_name, subset_name, metric_name)
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
    _DDL_TEST_METRICS,
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


def build_per_volume_data_from_duckdb(
    db_path: Path,
    metrics: list[str] | None = None,
    split: str = "trainval",
) -> dict[str, dict[str, dict[int, np.ndarray]]]:
    """Build per-volume data structure from DuckDB (not mlruns).

    This is the DuckDB-only replacement for _build_per_volume_data() which
    scans mlruns filesystem directly. The DuckDB is the single input contract.

    Parameters
    ----------
    db_path:
        Path to the biostatistics DuckDB file.
    metrics:
        List of metric names to extract. If None, extract all.
    split:
        ``"trainval"`` reads per_volume_metrics (eval/ prefix),
        ``"test"`` reads test_metrics (test/ prefix).

    Returns
    -------
    ``{metric_name: {condition_key: {fold_id: scores_array}}}``
    where condition_key is built from the runs table (loss_function + factors).
    """
    import duckdb

    from minivess.pipeline.biostatistics_statistics import encode_condition_key

    conn = duckdb.connect(str(db_path), read_only=True)

    if split == "trainval":
        # Join per_volume_metrics with runs to get factor tags
        query = """
            SELECT
                r.run_id,
                r.loss_function,
                r.model_family,
                r.with_aux_calib,
                r.post_training_method,
                r.recalibration,
                r.ensemble_strategy,
                pv.fold_id,
                pv.volume_id,
                pv.metric_name,
                pv.metric_value
            FROM per_volume_metrics pv
            JOIN runs r ON pv.run_id = r.run_id
        """
        if metrics:
            placeholders = ", ".join(f"'{m}'" for m in metrics)
            query += f" WHERE pv.metric_name IN ({placeholders})"
        query += " ORDER BY pv.metric_name, r.run_id, pv.fold_id, pv.volume_id"
    else:
        # Test split: read from test_metrics table (no fold structure)
        query = """
            SELECT
                r.run_id,
                r.loss_function,
                r.model_family,
                r.with_aux_calib,
                r.post_training_method,
                r.recalibration,
                r.ensemble_strategy,
                0 AS fold_id,
                tm.subset_name AS volume_id,
                tm.metric_name,
                tm.point_estimate AS metric_value
            FROM test_metrics tm
            JOIN runs r ON tm.run_id = r.run_id
        """
        if metrics:
            placeholders = ", ".join(f"'{m}'" for m in metrics)
            query += f" WHERE tm.metric_name IN ({placeholders})"

    rows = conn.execute(query).fetchall()
    conn.close()

    # Build nested dict: metric -> condition_key -> fold_id -> list[float]
    data: dict[str, dict[str, dict[int, list[float]]]] = {}

    for row in rows:
        (
            _run_id,
            loss_function,
            model_family,
            with_aux_calib,
            post_training_method,
            recalibration,
            ensemble_strategy,
            fold_id,
            _volume_id,
            metric_name,
            metric_value,
        ) = row

        condition_key = encode_condition_key({
            "ensemble_strategy": str(ensemble_strategy or "none"),
            "loss_function": str(loss_function or "unknown"),
            "model_family": str(model_family or "unknown"),
            "post_training_method": str(post_training_method or "none"),
            "recalibration": str(recalibration or "none"),
            "with_aux_calib": str(with_aux_calib).lower() if with_aux_calib is not None else "false",
        })

        data.setdefault(metric_name, {}).setdefault(condition_key, {}).setdefault(
            int(fold_id), []
        ).append(float(metric_value))

    # Convert lists to numpy arrays
    result: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    for metric, conditions in data.items():
        result[metric] = {}
        for cond, folds in conditions.items():
            result[metric][cond] = {
                fold_id: np.array(values, dtype=np.float64)
                for fold_id, values in sorted(folds.items())
            }

    return result


# ---------------------------------------------------------------------------
# Analysis Results DuckDB (DuckDB 1) — Plan Task 1.2
# ---------------------------------------------------------------------------
#
# NOTE: analysis_results.duckdb has its OWN schema (different DDL from the
# biostatistics DuckDB above). "Extend module" = both DuckDB builders live
# in the same .py file (Rule 26), NOT the same DDL. Two DuckDB files,
# two schemas, one Python module.

ANALYSIS_TABLES = [
    "runs",
    "per_volume_metrics",
    "fold_metrics",
    "metadata",
]

_ANALYSIS_DDL_RUNS = """
    CREATE TABLE IF NOT EXISTS runs (
        run_id VARCHAR PRIMARY KEY,
        experiment_id VARCHAR,
        experiment_name VARCHAR,
        loss_function VARCHAR,
        fold_id INTEGER,
        model_family VARCHAR,
        with_aux_calib BOOLEAN,
        post_training_method VARCHAR DEFAULT 'none',
        recalibration VARCHAR DEFAULT 'none',
        ensemble_strategy VARCHAR DEFAULT 'none',
        is_zero_shot BOOLEAN DEFAULT FALSE,
        status VARCHAR,
        condition_key VARCHAR
    )
"""

_ANALYSIS_DDL_PER_VOLUME_METRICS = """
    CREATE TABLE IF NOT EXISTS per_volume_metrics (
        run_id VARCHAR,
        fold_id INTEGER,
        split VARCHAR,
        dataset VARCHAR,
        volume_id VARCHAR,
        metric_name VARCHAR,
        metric_value DOUBLE,
        PRIMARY KEY (run_id, fold_id, volume_id, metric_name)
    )
"""

_ANALYSIS_DDL_FOLD_METRICS = """
    CREATE TABLE IF NOT EXISTS fold_metrics (
        run_id VARCHAR,
        fold_id INTEGER,
        split VARCHAR,
        metric_name VARCHAR,
        metric_value DOUBLE,
        PRIMARY KEY (run_id, fold_id, metric_name, split)
    )
"""

_ANALYSIS_DDL_METADATA = """
    CREATE TABLE IF NOT EXISTS metadata (
        key VARCHAR PRIMARY KEY,
        value VARCHAR
    )
"""

_ANALYSIS_ALL_DDL = [
    _ANALYSIS_DDL_RUNS,
    _ANALYSIS_DDL_PER_VOLUME_METRICS,
    _ANALYSIS_DDL_FOLD_METRICS,
    _ANALYSIS_DDL_METADATA,
]


def build_analysis_results_duckdb(
    *,
    manifest: Any,
    per_volume_records: list[dict[str, object]],
    fold_metric_records: list[dict[str, object]],
    output_path: Path,
    metadata: dict[str, str],
) -> Path:
    """Build analysis_results.duckdb (DuckDB 1) from evaluated run data.

    This is the self-contained dataset for external biostatisticians.
    Schema is DIFFERENT from the biostatistics DuckDB above.

    Parameters
    ----------
    manifest:
        SourceRunManifest with discovered runs.
    per_volume_records:
        List of dicts with keys: run_id, fold_id, split, dataset,
        volume_id, metric_name, metric_value.
    fold_metric_records:
        List of dicts with keys: run_id, fold_id, split,
        metric_name, metric_value.
    output_path:
        Output path for the DuckDB file.
    metadata:
        Key-value pairs for the metadata table (git_sha, config JSON, etc.)

    Returns
    -------
    Path to the created DuckDB file.
    """
    import duckdb

    from minivess.pipeline.biostatistics_statistics import encode_condition_key

    output_path.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(output_path))

    # Drop all tables for idempotent rebuild
    for table in ANALYSIS_TABLES:
        conn.execute(f"DROP TABLE IF EXISTS {table}")  # noqa: S608

    # Create schema
    for ddl in _ANALYSIS_ALL_DDL:
        conn.execute(ddl)

    # Insert runs with condition_key
    for run in manifest.runs:
        condition_key = encode_condition_key({
            "ensemble_strategy": str(getattr(run, "ensemble_strategy", "none")),
            "loss_function": run.loss_function,
            "model_family": run.model_family,
            "post_training_method": str(
                getattr(run, "post_training_method", "none")
            ),
            "recalibration": str(getattr(run, "recalibration", "none")),
            "with_aux_calib": str(run.with_aux_calib).lower(),
        })
        conn.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                run.run_id,
                run.experiment_id,
                run.experiment_name,
                run.loss_function,
                run.fold_id,
                run.model_family,
                run.with_aux_calib,
                getattr(run, "post_training_method", "none"),
                getattr(run, "recalibration", "none"),
                getattr(run, "ensemble_strategy", "none"),
                getattr(run, "is_zero_shot", False),
                run.status,
                condition_key,
            ],
        )

    # Insert per-volume metrics
    for rec in per_volume_records:
        conn.execute(
            "INSERT INTO per_volume_metrics VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                rec["run_id"],
                rec["fold_id"],
                rec["split"],
                rec["dataset"],
                rec["volume_id"],
                rec["metric_name"],
                rec["metric_value"],
            ],
        )

    # Insert fold-level metrics
    for rec in fold_metric_records:
        conn.execute(
            "INSERT INTO fold_metrics VALUES (?, ?, ?, ?, ?)",
            [
                rec["run_id"],
                rec["fold_id"],
                rec["split"],
                rec["metric_name"],
                rec["metric_value"],
            ],
        )

    # Insert metadata
    for key, value in metadata.items():
        conn.execute(
            "INSERT INTO metadata VALUES (?, ?)",
            [key, str(value)],
        )

    conn.close()

    logger.info(
        "Built analysis_results DuckDB at %s: %d runs, %d per-volume, %d fold metrics",
        output_path,
        len(manifest.runs),
        len(per_volume_records),
        len(fold_metric_records),
    )
    return output_path


def read_analysis_duckdb(
    db_path: Path,
) -> tuple[Any, dict[str, dict[str, dict[int, np.ndarray]]]]:
    """Read analysis_results.duckdb into SourceRunManifest + PerVolumeData.

    This is the primary input path for the Biostatistics Flow when
    analysis_duckdb_path is set in BiostatisticsConfig.

    Parameters
    ----------
    db_path:
        Path to the analysis_results.duckdb file.

    Returns
    -------
    Tuple of:
      - SourceRunManifest with all runs and factorial metadata
      - PerVolumeData: ``{metric: {condition_key: {fold_id: scores_array}}}``
    """
    import duckdb

    from minivess.pipeline.biostatistics_statistics import encode_condition_key
    from minivess.pipeline.biostatistics_types import SourceRun, SourceRunManifest

    conn = duckdb.connect(str(db_path), read_only=True)

    # --- Build SourceRunManifest from runs table ---
    run_rows = conn.execute(
        "SELECT run_id, experiment_id, experiment_name, loss_function, fold_id, "
        "model_family, with_aux_calib, post_training_method, recalibration, "
        "ensemble_strategy, is_zero_shot, status FROM runs"
    ).fetchall()

    runs: list[SourceRun] = []
    for row in run_rows:
        runs.append(
            SourceRun(
                run_id=row[0],
                experiment_id=row[1] or "",
                experiment_name=row[2] or "",
                loss_function=row[3] or "unknown",
                fold_id=int(row[4]),
                model_family=row[5] or "unknown",
                with_aux_calib=bool(row[6]),
                post_training_method=row[7] or "none",
                recalibration=row[8] or "none",
                ensemble_strategy=row[9] or "none",
                is_zero_shot=bool(row[10]),
                status=row[11] or "FINISHED",
            )
        )
    manifest = SourceRunManifest.from_runs(runs)

    # --- Build PerVolumeData from per_volume_metrics + runs ---
    pv_rows = conn.execute(
        """
        SELECT
            r.loss_function,
            r.model_family,
            r.with_aux_calib,
            r.post_training_method,
            r.recalibration,
            r.ensemble_strategy,
            pv.fold_id,
            pv.metric_name,
            pv.metric_value
        FROM per_volume_metrics pv
        JOIN runs r ON pv.run_id = r.run_id
        ORDER BY pv.metric_name, r.run_id, pv.fold_id
        """
    ).fetchall()

    conn.close()

    # Build nested dict: metric -> condition_key -> fold_id -> list[float]
    data: dict[str, dict[str, dict[int, list[float]]]] = {}

    for row in pv_rows:
        (
            loss_function,
            model_family,
            with_aux_calib,
            post_training_method,
            recalibration,
            ensemble_strategy,
            fold_id,
            metric_name,
            metric_value,
        ) = row

        condition_key = encode_condition_key({
            "ensemble_strategy": str(ensemble_strategy or "none"),
            "loss_function": str(loss_function or "unknown"),
            "model_family": str(model_family or "unknown"),
            "post_training_method": str(post_training_method or "none"),
            "recalibration": str(recalibration or "none"),
            "with_aux_calib": (
                str(with_aux_calib).lower()
                if with_aux_calib is not None
                else "false"
            ),
        })

        data.setdefault(metric_name, {}).setdefault(condition_key, {}).setdefault(
            int(fold_id), []
        ).append(float(metric_value))

    # Convert lists to numpy arrays
    pv_data: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    for metric, conditions in data.items():
        pv_data[metric] = {}
        for cond, folds in conditions.items():
            pv_data[metric][cond] = {
                fold_id: np.array(values, dtype=np.float64)
                for fold_id, values in sorted(folds.items())
            }

    logger.info(
        "Read analysis DuckDB: %d runs, %d metrics, %d conditions",
        len(runs),
        len(pv_data),
        len(next(iter(pv_data.values()), {})) if pv_data else 0,
    )

    return manifest, pv_data


def export_analysis_parquet(db_path: Path, output_dir: Path) -> list[Path]:
    """Export all analysis DuckDB tables to Parquet files.

    Parameters
    ----------
    db_path:
        Path to the analysis_results.duckdb file.
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
    for table in ANALYSIS_TABLES:
        parquet_path = output_dir / f"{table}.parquet"
        conn.execute(
            f"COPY {table} TO '{parquet_path}' (FORMAT PARQUET)"  # noqa: S608
        )
        paths.append(parquet_path)

    conn.close()
    logger.info("Exported %d analysis Parquet files to %s", len(paths), output_dir)
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
    # Layer B+C tags
    post_training_method = tags.get(
        "post_training_method", getattr(run, "post_training_method", "none")
    )
    recalibration = tags.get("recalibration", getattr(run, "recalibration", "none"))
    ensemble_strategy = tags.get(
        "ensemble_strategy", getattr(run, "ensemble_strategy", "none")
    )
    is_zero_shot_str = tags.get("is_zero_shot", "false")
    is_zero_shot = is_zero_shot_str.lower() in ("true", "1") or getattr(
        run, "is_zero_shot", False
    )

    # Insert into runs table
    conn.execute(
        "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            run.run_id,
            run.experiment_id,
            run.experiment_name,
            run.loss_function,
            run.fold_id,
            model_family,
            run.with_aux_calib,
            run.status,
            start_time,
            post_training_method,
            recalibration,
            ensemble_strategy,
            is_zero_shot,
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
        # Classify the metric — test/ prefix first (most specific)
        if _is_test_metric(metric_name):
            dataset_name, subset_name, base_metric = _parse_test_metric(metric_name)
            # Skip CI variant metrics
            if not any(base_metric.endswith(s) for s in _CI_SUFFIXES):
                conn.execute(
                    "INSERT INTO test_metrics VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [
                        run.run_id,
                        dataset_name,
                        subset_name,
                        base_metric,
                        value,
                        None,
                        None,
                    ],
                )
        elif _is_per_volume_metric(metric_name):
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


def _extract_fold_id(fold_str: str) -> int | None:
    """Extract integer fold ID from ``"0"`` or ``"fold_0"`` convention."""
    if fold_str.isdigit():
        return int(fold_str)
    if fold_str.startswith("fold_") and fold_str[5:].isdigit():
        return int(fold_str[5:])
    return None


def _parse_per_volume_metric(metric_name: str) -> tuple[int | None, str, str]:
    """Parse eval/{fold}/vol/{id}/{metric} into (fold_id, volume_id, base_metric).

    Handles both ``eval/0/vol/3/dsc`` (integer fold) and ``eval/fold_0/vol/3/dsc``
    (prefixed fold) conventions.
    """
    parts = metric_name.split("/")
    if len(parts) < 5 or parts[0] != "eval" or parts[2] != "vol":
        return None, "", ""
    fold_id = _extract_fold_id(parts[1])
    if fold_id is None:
        return None, "", ""
    return fold_id, parts[3], parts[4]


def _parse_eval_fold_metric(metric_name: str) -> tuple[int, str] | None:
    """Parse eval/{fold}/{metric} into (fold_id, base_metric).

    Handles both ``eval/0/dsc`` and ``eval/fold_0/dsc`` conventions.
    """
    parts = metric_name.split("/")
    if len(parts) < 3 or parts[0] != "eval":
        return None
    fold_id = _extract_fold_id(parts[1])
    if fold_id is None:
        return None
    return fold_id, parts[2]


def _is_test_metric(metric_name: str) -> bool:
    """Check if metric name matches test/{dataset}/{metric} or test/{dataset}/{subset}/{metric}.

    Uses str.split("/") — no regex (CLAUDE.md Rule 16).

    Parameters
    ----------
    metric_name:
        Slash-separated metric key.

    Returns
    -------
    True if the metric has a test/ prefix.
    """
    parts = metric_name.split("/")
    return len(parts) >= 3 and parts[0] == "test"


def _parse_test_metric(metric_name: str) -> tuple[str, str, str]:
    """Parse test/ prefix metrics into (dataset_name, subset_name, base_metric).

    Handles two formats:
    - ``test/{dataset}/{subset}/{metric}`` -> (dataset, subset, metric)
    - ``test/{dataset}/{metric}``          -> (dataset, "", metric)

    Uses str.split("/") — no regex (CLAUDE.md Rule 16).

    Parameters
    ----------
    metric_name:
        Slash-separated metric key starting with ``test/``.

    Returns
    -------
    Tuple of (dataset_name, subset_name, base_metric).
    """
    parts = metric_name.split("/")
    # parts[0] is always "test"
    if len(parts) >= 4:
        # test/deepvess/all/dsc -> ("deepvess", "all", "dsc")
        return parts[1], parts[2], parts[3]
    if len(parts) == 3:
        # test/aggregate/dsc -> ("aggregate", "", "dsc")
        return parts[1], "", parts[2]
    # Shouldn't happen if _is_test_metric was checked first
    return "", "", metric_name
