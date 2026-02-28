"""Retroactive MLflow metadata backfill.

Safely adds new params to existing MLflow runs without overwriting
existing values or changing run status.

Safety rules:
- Skip params that already exist with a different value
- Preserve original run status (FINISHED/FAILED/KILLED)
- Skip RUNNING runs (likely crashed mid-training)
- Add sys_backfill_note for provenance
- Use absolute path for tracking URI

Pattern reference: foundation-PLR ``src/log_helpers/mlflow_utils.py``
"""

from __future__ import annotations

import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
from mlflow.tracking import MlflowClient

if TYPE_CHECKING:
    from minivess.data.splits import FoldSplit

logger = logging.getLogger(__name__)


def backfill_run(
    run_id: str,
    *,
    new_params: dict[str, str],
    tracking_uri: str,
) -> dict[str, int | str]:
    """Add new params to an existing MLflow run.

    Parameters
    ----------
    run_id:
        MLflow run ID to update.
    new_params:
        Dict of param name -> value to add.
    tracking_uri:
        MLflow tracking URI (should be absolute path).

    Returns
    -------
    Dict with keys: added, skipped, skipped_reason (if applicable).
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # Check run status
    run = client.get_run(run_id)
    original_status = run.info.status

    if original_status == "RUNNING":
        logger.warning("Skipping RUNNING run %s (likely crashed)", run_id)
        return {"added": 0, "skipped": 0, "skipped_reason": "RUNNING"}

    # Determine which params are new vs existing
    existing_params = run.data.params
    params_to_add: dict[str, str] = {}
    skipped = 0

    for key, value in new_params.items():
        if key in existing_params:
            if existing_params[key] == str(value):
                # Same value — idempotent, skip silently
                skipped += 1
            else:
                # Different value — skip with warning
                logger.warning(
                    "Skipping param %s: existing=%s, new=%s",
                    key,
                    existing_params[key],
                    value,
                )
                skipped += 1
        else:
            params_to_add[key] = str(value)

    # Add backfill note
    if "sys_backfill_note" not in existing_params:
        params_to_add["sys_backfill_note"] = (
            f"Backfilled {datetime.now(UTC).strftime('%Y-%m-%d')}, same machine"
        )

    if not params_to_add:
        return {"added": 0, "skipped": skipped}

    # Write new params
    with mlflow.start_run(run_id=run_id):
        mlflow.log_params(params_to_add)

    # Restore original status if not FINISHED (start_run silently sets FINISHED)
    if original_status != "FINISHED":
        client.set_terminated(run_id, status=original_status)

    logger.info(
        "Backfilled run %s: added=%d, skipped=%d",
        run_id[:8],
        len(params_to_add),
        skipped,
    )
    return {"added": len(params_to_add), "skipped": skipped}


def backfill_experiment(
    *,
    experiment_name: str,
    new_params: dict[str, str],
    tracking_uri: str,
) -> dict[str, int]:
    """Backfill all runs in an experiment.

    Parameters
    ----------
    experiment_name:
        MLflow experiment name.
    new_params:
        Dict of param name -> value to add to each run.
    tracking_uri:
        MLflow tracking URI.

    Returns
    -------
    Dict with keys: total, updated, skipped.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.warning("Experiment %s not found", experiment_name)
        return {"total": 0, "updated": 0, "skipped": 0}

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
    )

    total = len(runs)
    updated = 0
    skipped = 0

    for run in runs:
        result = backfill_run(
            run.info.run_id,
            new_params=new_params,
            tracking_uri=tracking_uri,
        )
        if result.get("skipped_reason") == "RUNNING":
            skipped += 1
        elif int(result["added"]) > 0:
            updated += 1
        else:
            skipped += 1

    return {"total": total, "updated": updated, "skipped": skipped}


def backfill_fold_tags(
    *,
    experiment_name: str,
    splits: list[FoldSplit],
    splits_file: Path,
    tracking_uri: str,
) -> dict[str, int]:
    """Backfill per-fold volume tags and splits artifact to all runs.

    For each run in the experiment, writes:
    - ``fold_N_train`` / ``fold_N_val`` tags with comma-separated volume IDs
    - ``splits_file`` tag pointing to the canonical splits JSON
    - ``split_mode`` param (set to ``"file"``)
    - Copies the splits JSON into the run's artifacts/splits/ directory

    Parameters
    ----------
    experiment_name:
        MLflow experiment name.
    splits:
        Pre-loaded FoldSplit objects from the canonical splits file.
    splits_file:
        Path to the canonical splits JSON file.
    tracking_uri:
        MLflow tracking URI (should be absolute path for filesystem backend).

    Returns
    -------
    Dict with keys: total, updated, skipped.
    """
    from minivess.observability.tracking import extract_volume_ids

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.warning("Experiment %s not found", experiment_name)
        return {"total": 0, "updated": 0, "skipped": 0}

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    total = len(runs)
    updated = 0
    skipped = 0

    # Pre-compute fold tags
    fold_tags: dict[str, str] = {}
    for fold_id, fold in enumerate(splits):
        train_ids = extract_volume_ids(fold.train)
        val_ids = extract_volume_ids(fold.val)
        fold_tags[f"fold_{fold_id}_train"] = ",".join(train_ids)
        fold_tags[f"fold_{fold_id}_val"] = ",".join(val_ids)
    fold_tags["splits_file"] = str(splits_file)
    fold_tags["backfill_fold_tags_at"] = datetime.now(UTC).isoformat()

    for run in runs:
        run_id = run.info.run_id
        if run.info.status == "RUNNING":
            logger.warning("Skipping RUNNING run %s", run_id[:8])
            skipped += 1
            continue

        existing_tags = run.data.tags
        if "fold_0_train" in existing_tags:
            logger.info("Run %s already has fold tags, skipping", run_id[:8])
            skipped += 1
            continue

        # Write fold tags
        for tag_key, tag_value in fold_tags.items():
            client.set_tag(run_id, tag_key, tag_value)

        # Write split_mode as param (skip if exists)
        if "split_mode" not in run.data.params:
            try:
                client.log_param(run_id, "split_mode", "file")
            except Exception:
                logger.debug("split_mode param already exists for %s", run_id[:8])

        # Copy splits file as artifact
        artifact_uri = run.info.artifact_uri
        if artifact_uri and artifact_uri.startswith("file://"):
            artifact_dir = Path(artifact_uri.replace("file://", ""))
        elif artifact_uri and not artifact_uri.startswith("http"):
            artifact_dir = Path(artifact_uri)
        else:
            artifact_dir = None

        if artifact_dir is not None and splits_file.exists():
            splits_dest = artifact_dir / "splits"
            splits_dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(splits_file), str(splits_dest / splits_file.name))
            logger.info("Copied splits file to %s", splits_dest)

        updated += 1
        logger.info("Backfilled fold tags for run %s", run_id[:8])

    return {"total": total, "updated": updated, "skipped": skipped}
