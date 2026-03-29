"""Reusable flow observability wrappers.

Provides single-line context managers that wire up ALL observability for a flow:
- Structured JSONL event logging (flow_start/flow_end)
- GPU heartbeat monitoring (GPU flows only)
- CUDA availability guard (GPU flows only)
- Force-flush stdout/stderr

Usage in ANY flow:
    # CPU flow (biostatistics, dashboard, deploy, etc.)
    with flow_observability_context("biostatistics", logs_dir=logs_dir):
        run_biostatistics_pipeline()

    # GPU flow (train, hpo, post_training, analysis)
    with gpu_flow_observability_context("train", logs_dir=logs_dir):
        train_model()

This eliminates per-flow boilerplate. Every flow gets full observability
by adding ONE line.
"""

from __future__ import annotations

import contextlib
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from minivess.observability.structured_logging import StructuredEventLogger

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def flow_observability_context(
    flow_name: str,
    *,
    logs_dir: Path | None = None,
) -> Generator[StructuredEventLogger, None, None]:
    """Context manager providing structured logging for ANY Prefect flow.

    Emits flow_start on enter, flow_end on exit. JSONL events written to
    logs_dir/events.jsonl. Force-flushes stdout/stderr on exit.

    Parameters
    ----------
    flow_name:
        Human-readable flow name (e.g., "biostatistics", "train").
    logs_dir:
        Directory for events.jsonl and heartbeat.json. If None, events go
        to Python logger only (no file output).
    """
    if logs_dir is not None:
        logs_dir.mkdir(parents=True, exist_ok=True)
    event_logger = StructuredEventLogger(output_dir=logs_dir)
    start_time = time.monotonic()

    event_logger.log_event("flow_start", {"flow_name": flow_name})
    logger.info("Flow started: %s", flow_name)
    sys.stdout.flush()

    try:
        yield event_logger
    except Exception:
        elapsed = time.monotonic() - start_time
        event_logger.log_event("flow_end", {
            "flow_name": flow_name,
            "status": "failed",
            "duration_s": round(elapsed, 1),
        })
        logger.error("Flow failed: %s (%.1fs)", flow_name, elapsed)
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    else:
        elapsed = time.monotonic() - start_time
        event_logger.log_event("flow_end", {
            "flow_name": flow_name,
            "status": "completed",
            "duration_s": round(elapsed, 1),
        })
        logger.info("Flow completed: %s (%.1fs)", flow_name, elapsed)
        sys.stdout.flush()
        sys.stderr.flush()


@contextlib.contextmanager
def gpu_flow_observability_context(
    flow_name: str,
    *,
    logs_dir: Path | None = None,
    heartbeat_interval_s: float = 30.0,
    low_util_threshold_pct: int = 5,
    alert_after_s: float = 120.0,
) -> Generator[StructuredEventLogger, None, None]:
    """Context manager for GPU Prefect flows with CUDA guard + heartbeat.

    Extends flow_observability_context with:
    1. require_cuda_context() — fail-fast if CUDA unavailable
    2. GpuHeartbeatMonitor — background GPU utilization monitoring

    Parameters
    ----------
    flow_name:
        Flow name for CUDA guard and logging.
    logs_dir:
        Directory for events.jsonl and heartbeat.json.
    heartbeat_interval_s:
        GPU heartbeat check interval (from GPU_HEARTBEAT_INTERVAL_S env).
    low_util_threshold_pct:
        Low GPU util threshold (from GPU_HEARTBEAT_LOW_UTIL_THRESHOLD_PCT env).
    alert_after_s:
        Alert delay (from GPU_HEARTBEAT_ALERT_AFTER_S env).
    """
    from minivess.observability.gpu_heartbeat import GpuHeartbeatMonitor
    from minivess.orchestration.cuda_guard import require_cuda_context

    # Fail-fast: CUDA must be available for GPU flows
    require_cuda_context(flow_name)

    # Start GPU heartbeat + structured logging
    heartbeat = GpuHeartbeatMonitor(
        output_dir=logs_dir or Path("/tmp/heartbeat"),  # noqa: S108
        check_interval_s=heartbeat_interval_s,
        low_util_threshold_pct=low_util_threshold_pct,
        alert_after_s=alert_after_s,
    )

    with heartbeat, flow_observability_context(flow_name, logs_dir=logs_dir) as event_logger:
        yield event_logger
