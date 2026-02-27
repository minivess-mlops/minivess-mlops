"""Filesystem-based MLflow run inspector.

Reads directly from the mlruns/ directory layout without requiring a live
MLflow tracking server or client. Designed for integration tests and
post-hoc artifact verification.

MLflow stores its local filesystem artifacts as:
    mlruns/<experiment_id>/<run_id>/
        tags/<key>          — plain text, one value per file
        metrics/<key>       — lines: "<timestamp> <value> <step>"
        params/<key>        — plain text, one value per file
        artifacts/          — arbitrary nested artifacts

All functions are pure-Python and operate on pathlib.Path objects only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_production_runs(mlruns_dir: Path, experiment_id: str) -> list[str]:
    """Return run IDs for production runs in *experiment_id*.

    A run is classified as *production* when it has evaluation metrics for
    **all three folds** (fold0, fold1, and fold2), indicated by the presence
    of at least one metric file whose name starts with ``eval_fold2``.

    Incomplete runs (e.g. those interrupted after fold0/fold1 only) are
    excluded.  This mirrors the real data: 4 production runs out of 9 total
    in the v2 experiment, where 4 non-production runs have only fold0+fold1.

    Args:
        mlruns_dir: Root mlruns directory (e.g. ``repo_root / "mlruns"``).
        experiment_id: MLflow experiment ID string.

    Returns:
        Sorted list of run ID strings that pass the production filter.
    """
    experiment_dir = mlruns_dir / experiment_id
    if not experiment_dir.is_dir():
        return []

    production: list[str] = []
    for run_dir in experiment_dir.iterdir():
        if not run_dir.is_dir():
            continue
        if run_dir.name == "meta.yaml":
            continue
        metrics_dir = run_dir / "metrics"
        if not metrics_dir.is_dir():
            continue
        # Production discriminator: all 3 folds complete (fold2 present)
        has_fold2 = any(
            metric_file.name.startswith("eval_fold2")
            for metric_file in metrics_dir.iterdir()
        )
        if has_fold2:
            production.append(run_dir.name)

    return sorted(production)


def get_run_tags(
    mlruns_dir: Path,
    experiment_id: str,
    run_id: str,
) -> dict[str, str]:
    """Return all tag key-value pairs for *run_id*.

    Each tag is stored as a plain-text file under ``tags/``.  The filename is
    the tag key and the file content is the tag value (stripped of whitespace).

    Args:
        mlruns_dir: Root mlruns directory.
        experiment_id: MLflow experiment ID string.
        run_id: 32-character hexadecimal run ID.

    Returns:
        Dict mapping tag name to tag value.  Empty dict if tags dir absent.
    """
    tags_dir = mlruns_dir / experiment_id / run_id / "tags"
    if not tags_dir.is_dir():
        return {}

    tags: dict[str, str] = {}
    for tag_file in tags_dir.iterdir():
        if tag_file.is_file():
            tags[tag_file.name] = tag_file.read_text(encoding="utf-8").strip()
    return tags


def get_run_metrics_list(
    mlruns_dir: Path,
    experiment_id: str,
    run_id: str,
) -> list[str]:
    """Return sorted list of metric names recorded for *run_id*.

    Each metric is stored as a plain-text file under ``metrics/`` whose name
    is the metric key.  This function returns the names only; use
    ``read_metric_value`` to read the actual numeric value.

    Args:
        mlruns_dir: Root mlruns directory.
        experiment_id: MLflow experiment ID string.
        run_id: 32-character hexadecimal run ID.

    Returns:
        Sorted list of metric name strings.  Empty list if metrics dir absent.
    """
    metrics_dir = mlruns_dir / experiment_id / run_id / "metrics"
    if not metrics_dir.is_dir():
        return []

    return sorted(
        metric_file.name
        for metric_file in metrics_dir.iterdir()
        if metric_file.is_file()
    )


def get_run_params(
    mlruns_dir: Path,
    experiment_id: str,
    run_id: str,
) -> dict[str, str]:
    """Return all hyperparameter key-value pairs for *run_id*.

    Each param is stored as a plain-text file under ``params/``.  The filename
    is the param key and the file content is the param value (stripped).

    Args:
        mlruns_dir: Root mlruns directory.
        experiment_id: MLflow experiment ID string.
        run_id: 32-character hexadecimal run ID.

    Returns:
        Dict mapping param name to param value string.  Empty if absent.
    """
    params_dir = mlruns_dir / experiment_id / run_id / "params"
    if not params_dir.is_dir():
        return {}

    params: dict[str, str] = {}
    for param_file in params_dir.iterdir():
        if param_file.is_file():
            params[param_file.name] = param_file.read_text(encoding="utf-8").strip()
    return params


def read_metric_last_value(
    mlruns_dir: Path,
    experiment_id: str,
    run_id: str,
    metric_name: str,
) -> float:
    """Read the most recent (last line) float value of *metric_name*.

    MLflow metric files use the format ``<timestamp> <value> <step>`` with one
    entry per line. The last line holds the most recent logged value.

    Args:
        mlruns_dir: Root mlruns directory.
        experiment_id: MLflow experiment ID string.
        run_id: 32-character hexadecimal run ID.
        metric_name: Name of the metric to read.

    Returns:
        Most recent float value.

    Raises:
        FileNotFoundError: If the metric file does not exist.
        ValueError: If the file cannot be parsed as expected.
    """
    metric_file = mlruns_dir / experiment_id / run_id / "metrics" / metric_name
    content = metric_file.read_text(encoding="utf-8").strip()
    last_line = content.splitlines()[-1]
    # Format: "<timestamp> <value> <step>"
    parts = last_line.split()
    return float(parts[1])
