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


def discover_source_runs_from_api(
    experiment_names: list[str],
    tracking_uri: str | None = None,
) -> SourceRunManifest:
    """Discover FINISHED training runs via MLflow API (remote backend).

    Use this instead of discover_source_runs() when MLflow tracking uses
    a remote backend (DagsHub, Cloud Run, etc.) where local mlruns/ is empty.

    Parameters
    ----------
    experiment_names:
        MLflow experiment names to include.
    tracking_uri:
        MLflow tracking URI. If None, reads from MLFLOW_TRACKING_URI env var.

    Returns
    -------
    SourceRunManifest with all discovered runs and SHA-256 fingerprint.
    """
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.MlflowClient()
    runs: list[SourceRun] = []

    for exp_name in experiment_names:
        exp = client.get_experiment_by_name(exp_name)
        if exp is None:
            logger.warning("Experiment %r not found on MLflow server", exp_name)
            continue

        api_runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="status = 'FINISHED'",
            max_results=500,
        )

        for r in api_runs:
            tags = dict(r.data.tags)
            run = SourceRun(
                run_id=r.info.run_id,
                experiment_id=exp.experiment_id,
                experiment_name=exp_name,
                loss_function=tags.get("loss_function", "unknown"),
                fold_id=int(tags.get("fold_id", -1)),
                model_family=tags.get("model_family", "unknown"),
                with_aux_calib=tags.get("with_aux_calib", "false").lower() == "true",
                status="FINISHED",
                post_training_method=tags.get("post_training_method", "none"),
                recalibration=tags.get("recalibration", "none"),
                ensemble_strategy=tags.get("ensemble_strategy", "none"),
                is_zero_shot=tags.get("is_zero_shot", "false").lower() == "true",
            )
            # Skip parent/aggregate runs that have no fold_id
            if run.fold_id >= 0:
                runs.append(run)

    logger.info(
        "Discovered %d FINISHED runs via MLflow API across %d experiments",
        len(runs),
        len(experiment_names),
    )
    return SourceRunManifest.from_runs(runs)


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
    raw_status = meta.get("status", "UNKNOWN")
    # MLflow file-based tracking stores status as integer:
    # 1=RUNNING, 2=SCHEDULED, 3=FINISHED, 4=FAILED, 5=KILLED
    _STATUS_MAP = {
        1: "RUNNING",
        2: "SCHEDULED",
        3: "FINISHED",
        4: "FAILED",
        5: "KILLED",
    }
    status = (
        _STATUS_MAP.get(raw_status, str(raw_status))
        if isinstance(raw_status, int)
        else str(raw_status)
    )

    # Extract loss_function and fold_id from params
    params = _read_params(run_dir)
    loss_function = params.get("loss_name", params.get("loss_function", "unknown"))
    fold_str = params.get("fold_id", params.get("fold", "0"))
    try:
        fold_id = int(fold_str)
    except (ValueError, TypeError):
        fold_id = 0

    model_family = params.get("model_family", params.get("model/family", "unknown"))
    aux_calib_str = params.get("with_aux_calib", params.get("aux_calib", "false"))
    with_aux_calib = aux_calib_str.lower() in ("true", "1", "yes")

    # Read tags directory for Layer B+C fields (tags override params)
    tags = _read_tags(run_dir)

    # Layer A overrides from tags (tags may be more up-to-date than params)
    if "model_family" in tags:
        model_family = tags["model_family"]
    if "loss_function" in tags:
        loss_function = tags["loss_function"]
    if "fold_id" in tags:
        import contextlib

        with contextlib.suppress(ValueError, TypeError):
            fold_id = int(tags["fold_id"])
    if "with_aux_calib" in tags:
        with_aux_calib = tags["with_aux_calib"].lower() in ("true", "1", "yes")

    # Layer B+C fields (from tags only — not in params)
    post_training_method = tags.get("post_training_method", "none")
    recalibration = tags.get("recalibration", tags.get("calibration_method", "none"))
    ensemble_strategy = tags.get("ensemble_strategy", "none")
    is_zero_shot = tags.get("is_zero_shot", "false").lower() in ("true", "1")

    return SourceRun(
        run_id=run_dir.name,
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        loss_function=loss_function,
        fold_id=fold_id,
        status=status,
        model_family=model_family,
        with_aux_calib=with_aux_calib,
        post_training_method=post_training_method,
        recalibration=recalibration,
        ensemble_strategy=ensemble_strategy,
        is_zero_shot=is_zero_shot,
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


def _read_tags(run_dir: Path) -> dict[str, str]:
    """Read MLflow run tags from the tags directory.

    Layer B+C fields (post_training_method, recalibration, ensemble_strategy,
    is_zero_shot) are stored as tags, not params. This function reads them.
    """
    tags_dir = run_dir / "tags"
    if not tags_dir.is_dir():
        return {}
    result: dict[str, str] = {}
    for tag_file in tags_dir.iterdir():
        if tag_file.is_file() and not tag_file.name.startswith("mlflow."):
            result[tag_file.name] = tag_file.read_text(encoding="utf-8").strip()
    return result
