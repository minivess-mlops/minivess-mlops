"""Source run discovery for the biostatistics flow.

Scans MLflow mlruns directory for completed training runs and builds
a SourceRunManifest with SHA-256 fingerprint. Also provides completeness
validation to ensure sufficient data for statistical analysis.

Pure functions — no Prefect, no Docker dependency.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import yaml

from minivess.pipeline.biostatistics_types import (
    SourceRun,
    SourceRunManifest,
    ValidationResult,
)

logger = logging.getLogger(__name__)


def discover_source_runs(
    mlruns_dir: Path,
    experiment_names: list[str],
) -> SourceRunManifest:
    """Discover FINISHED training runs from MLflow mlruns directory.

    Parameters
    ----------
    mlruns_dir:
        Path to the mlruns directory.
    experiment_names:
        MLflow experiment names to include.

    Returns
    -------
    SourceRunManifest with all discovered runs and SHA-256 fingerprint.
    """
    runs: list[SourceRun] = []

    # MLflow stores experiments as numbered directories
    # with a meta.yaml containing the experiment name
    if not mlruns_dir.exists():
        logger.warning("mlruns_dir does not exist: %s", mlruns_dir)
        return SourceRunManifest.from_runs([])

    experiment_map = _build_experiment_map(mlruns_dir, experiment_names)

    for exp_id, exp_name in experiment_map.items():
        exp_dir = mlruns_dir / exp_id
        for run_dir in sorted(exp_dir.iterdir()):
            if not run_dir.is_dir() or run_dir.name.startswith("."):
                continue
            run = _parse_run_dir(run_dir, exp_id, exp_name)
            if run is not None and run.status == "FINISHED":
                runs.append(run)

    logger.info(
        "Discovered %d FINISHED runs across %d experiments",
        len(runs),
        len(experiment_map),
    )
    return SourceRunManifest.from_runs(runs)


def validate_source_completeness(
    manifest: SourceRunManifest,
    min_folds: int = 3,
    min_conditions: int = 2,
) -> ValidationResult:
    """Validate that source runs provide sufficient data for analysis.

    Parameters
    ----------
    manifest:
        Discovered source runs.
    min_folds:
        Minimum folds per condition.
    min_conditions:
        Minimum number of distinct conditions (loss functions).

    Returns
    -------
    ValidationResult indicating pass/fail with details.

    Raises
    ------
    BiostatisticsValidationError
        When critical validation failures are found (not silent).
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Group runs by condition (loss function)
    conditions: dict[str, list[SourceRun]] = {}
    for run in manifest.runs:
        conditions.setdefault(run.loss_function, []).append(run)

    n_conditions = len(conditions)

    if n_conditions < min_conditions:
        errors.append(
            f"Only {n_conditions} condition(s) found, need >= {min_conditions}"
        )

    # Check folds per condition
    min_folds_found = float("inf")
    for loss, loss_runs in conditions.items():
        folds = {r.fold_id for r in loss_runs}
        n_folds = len(folds)
        if n_folds < min_folds:
            errors.append(
                f"Condition '{loss}' has {n_folds} fold(s), need >= {min_folds}"
            )
        min_folds_found = min(min_folds_found, n_folds)

    n_folds_per_condition = (
        int(min_folds_found) if min_folds_found != float("inf") else 0
    )

    valid = len(errors) == 0

    if not valid:
        msg = "Biostatistics validation failed:\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        raise BiostatisticsValidationError(msg)

    return ValidationResult(
        valid=valid,
        warnings=warnings,
        errors=errors,
        n_conditions=n_conditions,
        n_folds_per_condition=n_folds_per_condition,
    )


class BiostatisticsValidationError(Exception):
    """Raised when source data validation fails."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_experiment_map(
    mlruns_dir: Path,
    experiment_names: list[str],
) -> dict[str, str]:
    """Map experiment IDs to names for requested experiments."""
    result: dict[str, str] = {}
    for entry in sorted(mlruns_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        meta_file = entry / "meta.yaml"
        if not meta_file.exists():
            continue
        meta = yaml.safe_load(meta_file.read_text(encoding="utf-8"))
        name = meta.get("name", "")
        if name in experiment_names:
            result[entry.name] = name
    return result


def _parse_run_dir(
    run_dir: Path,
    experiment_id: str,
    experiment_name: str,
) -> SourceRun | None:
    """Parse a single run directory into a SourceRun."""
    meta_file = run_dir / "meta.yaml"
    if not meta_file.exists():
        return None

    meta = yaml.safe_load(meta_file.read_text(encoding="utf-8"))
    status = meta.get("status", "UNKNOWN")

    # Extract loss_function and fold_id from params
    params = _read_params(run_dir)
    loss_function = params.get("loss_name", params.get("loss_function", "unknown"))
    fold_str = params.get("fold_id", params.get("fold", "0"))
    try:
        fold_id = int(fold_str)
    except (ValueError, TypeError):
        fold_id = 0

    return SourceRun(
        run_id=run_dir.name,
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        loss_function=loss_function,
        fold_id=fold_id,
        status=status,
    )


def _read_params(run_dir: Path) -> dict[str, Any]:
    """Read MLflow run parameters from the params directory."""
    params_dir = run_dir / "params"
    if not params_dir.is_dir():
        return {}
    result: dict[str, Any] = {}
    for param_file in params_dir.iterdir():
        if param_file.is_file():
            result[param_file.name] = param_file.read_text(encoding="utf-8").strip()
    return result
