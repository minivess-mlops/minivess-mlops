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
