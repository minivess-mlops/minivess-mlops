"""Ghost run detection and cleanup for MLflow.

Provides signal handler registration to mark runs as FAILED on
SIGTERM/SIGINT, and cleanup functions for orphaned RUNNING runs.
"""

from __future__ import annotations

import logging
import signal
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import FrameType

    from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

# Store previous handlers to restore on cleanup
_previous_handlers: dict[int, Any] = {}


def _create_shutdown_handler(
    client: MlflowClient,
    *,
    run_id: str,
) -> Any:
    """Create a signal handler that marks the run as FAILED and exits.

    Parameters
    ----------
    client:
        MLflow client for setting run status.
    run_id:
        The active MLflow run ID to terminate.

    Returns
    -------
    Signal handler function.
    """

    def _handler(signum: int, frame: FrameType | None) -> None:
        sig_name = signal.Signals(signum).name
        logger.warning(
            "Received %s — marking MLflow run %s as FAILED and exiting",
            sig_name,
            run_id[:8],
        )
        try:
            client.set_terminated(run_id, status="FAILED")
            client.set_tag(run_id, "termination_signal", sig_name)
        except Exception:
            logger.warning("Failed to mark run as FAILED", exc_info=True)
        sys.exit(128 + signum)

    return _handler


def register_graceful_shutdown(
    client: MlflowClient,
    *,
    run_id: str,
) -> None:
    """Register SIGTERM and SIGINT handlers to gracefully terminate MLflow runs.

    On signal receipt, marks the active run as FAILED with a termination
    tag, then exits. Previous handlers are stored for potential restoration.

    Parameters
    ----------
    client:
        MLflow client instance.
    run_id:
        Active MLflow run ID to protect.
    """
    handler = _create_shutdown_handler(client, run_id=run_id)

    for sig in (signal.SIGTERM, signal.SIGINT):
        prev = signal.getsignal(sig)
        _previous_handlers[sig] = prev
        signal.signal(sig, handler)

    logger.info(
        "Registered graceful shutdown handlers for run %s (SIGTERM, SIGINT)",
        run_id[:8],
    )


def unregister_graceful_shutdown() -> None:
    """Restore previous signal handlers."""
    for sig, handler in _previous_handlers.items():
        if handler is not None:
            signal.signal(sig, handler)
    _previous_handlers.clear()


def find_ghost_runs(
    client: MlflowClient,
    *,
    experiment_ids: list[str],
    max_age_hours: float = 24.0,
) -> list[Any]:
    """Find RUNNING runs that are likely orphaned (ghost runs).

    Parameters
    ----------
    client:
        MLflow client instance.
    experiment_ids:
        Experiment IDs to search.
    max_age_hours:
        Maximum age in hours for a run to be considered potentially
        active. Runs older than this that are still RUNNING are
        likely ghosts.

    Returns
    -------
    List of MLflow Run objects that are RUNNING and likely orphaned.
    """
    ghost_runs: list[Any] = []
    for exp_id in experiment_ids:
        runs = client.search_runs(
            experiment_ids=[exp_id],
            filter_string="attributes.status = 'RUNNING'",
        )
        ghost_runs.extend(runs)

    logger.info(
        "Found %d RUNNING runs across %d experiments",
        len(ghost_runs),
        len(experiment_ids),
    )
    return ghost_runs


def cleanup_ghost_runs(
    client: MlflowClient,
    *,
    ghost_runs: list[Any],
    dry_run: bool = True,
) -> dict[str, int]:
    """Clean up ghost runs by marking them as FAILED.

    Parameters
    ----------
    client:
        MLflow client instance.
    ghost_runs:
        List of Run objects to clean up.
    dry_run:
        If True, only report what would be cleaned without modifying.

    Returns
    -------
    Dict with counts: ``cleaned``, ``would_clean``, ``errors``.
    """
    result = {"cleaned": 0, "would_clean": 0, "errors": 0}

    for run in ghost_runs:
        run_id = run.info.run_id
        if dry_run:
            result["would_clean"] += 1
            logger.info("[DRY RUN] Would mark run %s as FAILED", run_id[:8])
        else:
            try:
                client.set_terminated(run_id, status="FAILED")
                client.set_tag(run_id, "ghost_cleanup", "true")
                result["cleaned"] += 1
                logger.info("Marked ghost run %s as FAILED", run_id[:8])
            except Exception:
                result["errors"] += 1
                logger.warning("Failed to clean up run %s", run_id[:8], exc_info=True)

    return result
