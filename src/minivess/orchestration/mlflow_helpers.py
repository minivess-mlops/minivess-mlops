"""MLflow helper utilities for Prefect orchestration flows.

Eliminates the 3-liner duplication (set_tracking_uri + set_experiment + start_run)
and FlowContract boilerplate repeated across 6+ flow files.

Functions
---------
find_upstream_safely:
    Wrapper around FlowContract.find_upstream_run() that catches all exceptions
    and returns None instead of raising. Upstream discovery is best-effort —
    flows must handle None gracefully.

log_completion_safe:
    Wrapper around FlowContract.log_flow_completion() that catches all exceptions.
    Completion logging is observability — it must never block the flow.

emit_lineage_safe:
    Wrapper around OpenLineage emission that catches all exceptions.
    Lineage emission is observability — it must never block the flow.

start_mlflow_run_safe:
    Unified MLflow run creation: set URI + set experiment + start run + log metrics.
    Handles all exceptions so MLflow errors never block a flow.
"""

from __future__ import annotations

import logging
from typing import Any

from minivess.orchestration.flow_contract import FlowContract

logger = logging.getLogger(__name__)


def find_upstream_safely(
    *,
    tracking_uri: str,
    experiment_name: str,
    upstream_flow: str,
    tags: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    """Find the most recent upstream run, returning None on any failure.

    Wraps FlowContract.find_upstream_run() with exception isolation so
    upstream discovery failures never block the consumer flow.

    Parameters
    ----------
    tracking_uri:
        MLflow tracking URI.
    experiment_name:
        MLflow experiment name to search.
    upstream_flow:
        Prefect flow name of the upstream flow (FLOW_NAME_* constant).
    tags:
        Additional tag filters.

    Returns
    -------
    dict with ``run_id``, ``status``, ``tags`` if found, else None.
    """
    try:
        contract = FlowContract(tracking_uri=tracking_uri)
        return contract.find_upstream_run(
            experiment_name=experiment_name,
            upstream_flow=upstream_flow,
            tags=tags,
        )
    except Exception:
        logger.warning(
            "Failed to discover upstream flow %r in experiment %r — proceeding without it",
            upstream_flow,
            experiment_name,
            exc_info=True,
        )
        return None


def log_completion_safe(
    *,
    flow_name: str,
    tracking_uri: str,
    run_id: str | None,
) -> None:
    """Log flow completion via FlowContract, ignoring all failures.

    Wraps FlowContract.log_flow_completion() so that MLflow observability
    failures never block the flow from completing.

    Parameters
    ----------
    flow_name:
        Prefect @flow(name=...) value — use FLOW_NAME_* constants.
    tracking_uri:
        MLflow tracking URI.
    run_id:
        MLflow run ID to tag. If None, no action is taken.
    """
    if run_id is None:
        return
    try:
        contract = FlowContract(tracking_uri=tracking_uri)
        contract.log_flow_completion(flow_name=flow_name, run_id=run_id)
    except Exception:
        logger.warning(
            "Failed to log flow completion for %r (run_id=%r) — non-fatal",
            flow_name,
            run_id,
            exc_info=True,
        )


def emit_lineage_safe(
    *,
    job_name: str,
    inputs: list[dict[str, str]] | None = None,
    outputs: list[dict[str, str]] | None = None,
    namespace: str = "minivess",
) -> None:
    """Emit OpenLineage flow lineage, ignoring all failures.

    Wraps LineageEmitter + emit_flow_lineage() with exception isolation.
    Lineage emission is observability — it must never block the flow.

    Parameters
    ----------
    job_name:
        OpenLineage job name (e.g., "biostatistics-flow").
    inputs:
        Input dataset facets.
    outputs:
        Output dataset facets.
    namespace:
        OpenLineage namespace (default "minivess").
    """
    try:
        from minivess.observability.lineage import LineageEmitter, emit_flow_lineage

        _emitter = LineageEmitter(namespace=namespace)
        emit_flow_lineage(
            emitter=_emitter,
            job_name=job_name,
            inputs=inputs or [],
            outputs=outputs or [],
        )
    except Exception:
        logger.warning(
            "OpenLineage emission failed for %r — non-blocking",
            job_name,
            exc_info=True,
        )


def start_mlflow_run_safe(
    *,
    experiment_name: str,
    tracking_uri: str | None = None,
    run_tags: dict[str, str] | None = None,
    metrics: dict[str, float] | None = None,
    artifacts: dict[str, str] | None = None,
) -> str | None:
    """Create an MLflow run, log metrics/artifacts, and return the run ID.

    Unified MLflow run setup that replaces the 15-line boilerplate
    duplicated across 4+ flow files. Returns None on any failure.

    Parameters
    ----------
    experiment_name:
        MLflow experiment name (use ``resolve_experiment_name()``).
    tracking_uri:
        MLflow tracking URI. If None, auto-resolved.
    run_tags:
        Tags to set on the MLflow run.
    metrics:
        Metrics to log (key → value).
    artifacts:
        Artifacts to log (artifact_path → local_path).

    Returns
    -------
    MLflow run ID if successful, None otherwise.
    """
    try:
        import mlflow

        from minivess.observability.tracking import resolve_tracking_uri

        uri = tracking_uri or resolve_tracking_uri()
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(tags=run_tags or {}) as active_run:
            run_id = active_run.info.run_id
            if metrics:
                mlflow.log_metrics(metrics)
            if artifacts:
                for artifact_path, local_path in artifacts.items():
                    mlflow.log_artifact(local_path, artifact_path)
            return run_id
    except Exception:
        logger.warning(
            "Failed to create MLflow run for experiment %r — non-fatal",
            experiment_name,
            exc_info=True,
        )
        return None
