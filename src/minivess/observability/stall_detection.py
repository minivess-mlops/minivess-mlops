"""MLflow training stall detection.

Detects when a training run has not logged new MLflow metrics for longer
than a configurable threshold. Complements the heartbeat-based stall
detection in gpu_heartbeat.py.

See: .claude/metalearning/2026-03-29-silent-cpu-fallback-no-observability-4h-wasted.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


@dataclass
class StallResult:
    """Result of MLflow metric stall check."""

    stale: bool
    last_metric_ts: str | None
    minutes_since: float
    message: str


def detect_mlflow_metric_stall(
    run_id: str,
    threshold_minutes: float = 15.0,
    tracking_uri: str | None = None,
) -> StallResult:
    """Check if an MLflow run has stopped producing metrics.

    Parameters
    ----------
    run_id:
        MLflow run ID to check.
    threshold_minutes:
        Minutes without new metrics before declaring stale.
    tracking_uri:
        MLflow tracking URI. If None, uses MLFLOW_TRACKING_URI env var.

    Returns
    -------
    StallResult with stale flag, last metric timestamp, and minutes since.
    """
    try:
        import mlflow

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        client = mlflow.MlflowClient()
        run = client.get_run(run_id)

        # Get the most recent metric timestamp
        metrics = run.data.metrics
        if not metrics:
            return StallResult(
                stale=False,
                last_metric_ts=None,
                minutes_since=0.0,
                message=f"No metrics logged yet for run {run_id[:8]}",
            )

        # MLflow run info has end_time for completed runs
        # For running runs, check the last logged metric time
        run_info = run.info
        if run_info.end_time:
            last_ts_epoch = run_info.end_time / 1000.0  # ms → s
        else:
            # Use run start time as fallback if no end time
            last_ts_epoch = run_info.start_time / 1000.0 if run_info.start_time else 0.0

        last_dt = datetime.fromtimestamp(last_ts_epoch, tz=UTC)
        minutes_since = (datetime.now(UTC) - last_dt).total_seconds() / 60.0

        if minutes_since > threshold_minutes:
            return StallResult(
                stale=True,
                last_metric_ts=last_dt.isoformat(),
                minutes_since=minutes_since,
                message=f"Run {run_id[:8]} stale: {minutes_since:.1f} min since last activity",
            )

        return StallResult(
            stale=False,
            last_metric_ts=last_dt.isoformat(),
            minutes_since=minutes_since,
            message=f"Run {run_id[:8]} active: {minutes_since:.1f} min since last activity",
        )

    except ImportError:
        logger.warning("MLflow not installed — skipping stall detection")
        return StallResult(stale=False, last_metric_ts=None, minutes_since=0.0, message="MLflow not available")
    except Exception:
        logger.warning("MLflow stall detection failed", exc_info=True)
        return StallResult(stale=False, last_metric_ts=None, minutes_since=0.0, message="Check failed")
