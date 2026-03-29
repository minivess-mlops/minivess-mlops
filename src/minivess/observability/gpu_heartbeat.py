"""GPU heartbeat monitor — background thread for training observability.

Periodically checks GPU utilization and writes heartbeat.json.
Logs ERROR when GPU utilization stays below threshold for too long.

Usage as context manager:
    with GpuHeartbeatMonitor(output_dir=Path("/app/logs")):
        train_model(...)  # heartbeat updates in background

Graceful degradation: no-op when pynvml/GPU unavailable.

See: .claude/metalearning/2026-03-29-silent-cpu-fallback-no-observability-4h-wasted.md
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _get_gpu_snapshot() -> dict[str, Any]:
    """Get current GPU metrics via pynvml.

    Returns dict with gpu_util_pct, gpu_memory_used_mb, gpu_temp_c, status.
    Returns no_gpu status if pynvml or GPU is unavailable.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except pynvml.NVMLError:
            temp = 0

        return {
            "gpu_util_pct": int(util.gpu),
            "gpu_memory_used_mb": int(mem_info.used / (1024 * 1024)),
            "gpu_temp_c": int(temp),
            "status": "healthy",
        }
    except Exception:
        return {
            "gpu_util_pct": 0,
            "gpu_memory_used_mb": 0,
            "gpu_temp_c": 0,
            "status": "no_gpu",
        }


class GpuHeartbeatMonitor:
    """Background GPU monitoring thread with heartbeat file output.

    Parameters
    ----------
    output_dir:
        Directory for heartbeat.json output.
    check_interval_s:
        Seconds between GPU checks (default 30).
    low_util_threshold_pct:
        GPU utilization below this triggers alert tracking (default 5%).
    alert_after_s:
        Seconds of sustained low utilization before ERROR log (default 120).
    """

    def __init__(
        self,
        output_dir: Path,
        check_interval_s: float = 30.0,
        low_util_threshold_pct: int = 5,
        alert_after_s: float = 120.0,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._check_interval_s = check_interval_s
        self._low_util_threshold_pct = low_util_threshold_pct
        self._alert_after_s = alert_after_s
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._low_util_start: float | None = None
        self._mlflow_run_id: str | None = None

    def set_mlflow_run_id(self, run_id: str) -> None:
        """Set MLflow run ID for stall detection cross-check.

        Called after the MLflow run is created inside the flow body.
        Enables combined stall detection: low GPU util AND stale MLflow metrics.
        """
        self._mlflow_run_id = run_id

    def __enter__(self) -> GpuHeartbeatMonitor:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="gpu-heartbeat",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def is_alive(self) -> bool:
        """Check if the monitor thread is running."""
        return self._thread is not None and self._thread.is_alive()

    def _monitor_loop(self) -> None:
        """Main monitoring loop — runs in background thread."""
        while not self._stop_event.is_set():
            snapshot = _get_gpu_snapshot()
            self._write_heartbeat(snapshot)
            self._check_low_utilization(snapshot)
            self._check_mlflow_stall()
            self._stop_event.wait(self._check_interval_s)

    def _check_mlflow_stall(self) -> None:
        """Check MLflow metric staleness if run_id is set."""
        if self._mlflow_run_id is None:
            return
        try:
            from minivess.observability.stall_detection import detect_mlflow_metric_stall

            result = detect_mlflow_metric_stall(self._mlflow_run_id)
            if result.stale:
                logger.error(
                    "MLflow stall detected: %s (run %s)",
                    result.message,
                    self._mlflow_run_id[:8],
                )
        except Exception:
            pass  # Stall detection is best-effort

    def _write_heartbeat(self, snapshot: dict[str, Any]) -> None:
        """Write heartbeat.json with current GPU metrics."""
        heartbeat = {
            "timestamp": datetime.now(UTC).isoformat(),
            **snapshot,
        }
        hb_path = self._output_dir / "heartbeat.json"
        try:
            with hb_path.open("w", encoding="utf-8") as f:
                json.dump(heartbeat, f, indent=2)
        except OSError:
            logger.warning("Failed to write heartbeat.json to %s", hb_path)

    def _check_low_utilization(self, snapshot: dict[str, Any]) -> None:
        """Alert when GPU utilization is below threshold for too long."""
        if snapshot["status"] == "no_gpu":
            return

        util = snapshot["gpu_util_pct"]
        now = time.monotonic()

        if util < self._low_util_threshold_pct:
            if self._low_util_start is None:
                self._low_util_start = now
            elif now - self._low_util_start > self._alert_after_s:
                duration = now - self._low_util_start
                logger.error(
                    "GPU utilization below %d%% for %.0f seconds "
                    "(current: %d%%). Training may be stuck or running on CPU.",
                    self._low_util_threshold_pct,
                    duration,
                    util,
                )
        else:
            self._low_util_start = None
