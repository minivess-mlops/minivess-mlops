"""Maintenance Prefect flow — periodic cleanup of stale MLflow runs.

Flow 6: Runs as a scheduled maintenance task. Finds and cleans up
ghost runs (RUNNING runs that are likely orphaned from crashed containers
or preempted spot instances).

Issue: #683
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from mlflow.tracking import MlflowClient
from prefect import flow, task

from minivess.observability.flow_observability import flow_observability_context
from minivess.observability.ghost_cleanup import cleanup_ghost_runs, find_ghost_runs
from minivess.observability.prefect_hooks import create_task_timing_hooks
from minivess.observability.tracking import resolve_tracking_uri
from minivess.orchestration.constants import (
    EXPERIMENT_TRAINING,
    FLOW_NAME_MAINTENANCE,
)
from minivess.orchestration.docker_guard import require_docker_context

logger = logging.getLogger(__name__)

_on_complete, _on_fail = create_task_timing_hooks()


@task(name="cleanup-stale-runs", on_completion=[_on_complete], on_failure=[_on_fail])
def cleanup_stale_runs_task(
    *,
    experiment_name: str = EXPERIMENT_TRAINING,
    max_age_hours: float = 24.0,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Find and clean up stale RUNNING MLflow runs.

    Parameters
    ----------
    experiment_name:
        MLflow experiment to search for ghost runs.
    max_age_hours:
        Maximum age for a run to be considered potentially active.
    dry_run:
        If True, only report what would be cleaned.

    Returns
    -------
    Dict with cleanup results (cleaned/would_clean/errors counts).
    """
    tracking_uri = resolve_tracking_uri()
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.warning("Experiment '%s' not found — skipping cleanup", experiment_name)
        return {"would_clean": 0, "cleaned": 0, "errors": 0, "skipped": True}

    ghost_runs = find_ghost_runs(
        client,
        experiment_ids=[experiment.experiment_id],
        max_age_hours=max_age_hours,
    )

    if not ghost_runs:
        logger.info("No ghost runs found in experiment '%s'", experiment_name)
        return {"would_clean": 0, "cleaned": 0, "errors": 0}

    result = cleanup_ghost_runs(
        client,
        ghost_runs=ghost_runs,
        dry_run=dry_run,
    )

    logger.info(
        "Cleanup result for '%s': %s (dry_run=%s)",
        experiment_name,
        result,
        dry_run,
    )
    return result


@flow(name=FLOW_NAME_MAINTENANCE)
def maintenance_flow(
    *,
    experiment_name: str = EXPERIMENT_TRAINING,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Maintenance flow — periodic MLflow cleanup.

    Designed for Prefect scheduling (e.g., daily at 03:00 UTC).
    """
    require_docker_context("maintenance")

    logs_dir = Path(os.environ.get("LOGS_DIR", "/app/logs"))
    with flow_observability_context("maintenance", logs_dir=logs_dir):
        logger.info("Maintenance flow started (dry_run=%s)", dry_run)

        result = cleanup_stale_runs_task(
            experiment_name=experiment_name,
            dry_run=dry_run,
        )

        logger.info("Maintenance flow complete: %s", result)
        return result


if __name__ == "__main__":
    maintenance_flow(dry_run=True)
