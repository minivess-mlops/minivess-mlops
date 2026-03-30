"""Prefect task timing hooks for observability.

Provides on_completion and on_failure hook factories that emit structured
events with task name, duration, and status. Compatible with Prefect 3.x
task hook signature.

Usage:
    from minivess.observability.prefect_hooks import create_task_timing_hooks

    on_complete, on_fail = create_task_timing_hooks()

    @task(on_completion=[on_complete], on_failure=[on_fail])
    def my_task():
        ...
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def create_task_timing_hooks() -> tuple[Any, Any]:
    """Create Prefect task hook callables for completion and failure events.

    Returns (on_completion, on_failure) callables compatible with Prefect 3.x.
    Each hook logs structured timing information via the Python logger.
    """

    def on_completion(task: Any, task_run: Any, state: Any) -> None:
        """Log task completion with duration."""
        task_name = getattr(task, "name", "unknown")
        duration_s = _compute_duration(task_run)
        logger.info(
            "Task completed: %s (%.1fs)",
            task_name,
            duration_s,
            extra={
                "event_type": "task_end",
                "task_name": task_name,
                "status": "completed",
                "duration_s": duration_s,
            },
        )

    def on_failure(task: Any, task_run: Any, state: Any) -> None:
        """Log task failure with duration and error type."""
        task_name = getattr(task, "name", "unknown")
        duration_s = _compute_duration(task_run)
        error_type = ""
        if state is not None and hasattr(state, "result"):
            try:
                result = state.result(raise_on_failure=False)
                if isinstance(result, BaseException):
                    error_type = type(result).__name__
            except Exception:
                error_type = "unknown"

        logger.error(
            "Task failed: %s (%.1fs) — %s",
            task_name,
            duration_s,
            error_type or "unknown error",
            extra={
                "event_type": "task_end",
                "task_name": task_name,
                "status": "failed",
                "duration_s": duration_s,
                "error_type": error_type,
            },
        )

    return on_completion, on_failure


def _compute_duration(task_run: Any) -> float:
    """Compute task duration from Prefect task run timestamps."""
    try:
        if hasattr(task_run, "start_time") and task_run.start_time:
            start = task_run.start_time
            end = task_run.end_time or time.time()
            if hasattr(start, "timestamp"):
                return float(end.timestamp() - start.timestamp()) if hasattr(end, "timestamp") else 0.0
    except Exception:
        pass
    return 0.0
