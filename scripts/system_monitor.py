#!/usr/bin/env python
"""Real-time system resource monitor for training sessions.

Logs CPU, RAM, swap, GPU, and disk metrics at configurable intervals.
Designed to run as a background thread or standalone process to capture
resource usage leading up to OOM crashes.

Usage (standalone)::

    uv run python scripts/system_monitor.py --interval 5 --log-dir logs/monitor

Usage (as library)::

    from scripts.system_monitor import SystemMonitor
    monitor = SystemMonitor(log_dir=Path("logs/run1"), interval_sec=10)
    monitor.start()
    # ... training ...
    monitor.stop()
"""

from __future__ import annotations

import csv
import json
import logging
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ResourceSnapshot:
    """Single point-in-time resource measurement."""

    timestamp: str
    elapsed_sec: float
    # RAM
    ram_total_gb: float
    ram_used_gb: float
    ram_available_gb: float
    ram_percent: float
    # Swap
    swap_total_gb: float
    swap_used_gb: float
    swap_percent: float
    # CPU
    cpu_percent: float
    load_avg_1m: float
    load_avg_5m: float
    load_avg_15m: float
    # GPU (optional)
    gpu_name: str = ""
    gpu_memory_total_mb: int = 0
    gpu_memory_used_mb: int = 0
    gpu_memory_free_mb: int = 0
    gpu_utilization_percent: int = 0
    gpu_temperature_c: int = 0
    # Process-specific (optional)
    process_rss_gb: float = 0.0
    process_vms_gb: float = 0.0


@dataclass
class MonitorConfig:
    """Configuration for the system monitor."""

    interval_sec: float = 5.0
    log_dir: Path = field(default_factory=lambda: Path("logs/monitor"))
    memory_warn_gb: float = 50.0
    memory_abort_gb: float = 58.0
    enable_gpu: bool = True
    track_pid: int | None = None


def _read_meminfo() -> dict[str, int]:
    """Read /proc/meminfo and return values in kB."""
    result = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    result[key] = int(parts[1])
    except OSError:
        pass
    return result


def _read_loadavg() -> tuple[float, float, float]:
    """Read /proc/loadavg."""
    try:
        with open("/proc/loadavg", encoding="utf-8") as f:
            parts = f.read().split()
            return float(parts[0]), float(parts[1]), float(parts[2])
    except (OSError, IndexError, ValueError):
        return 0.0, 0.0, 0.0


def _read_cpu_percent() -> float:
    """Estimate CPU usage from /proc/stat (simple snapshot)."""
    try:
        with open("/proc/stat", encoding="utf-8") as f:
            line = f.readline()
        parts = line.split()
        if parts[0] == "cpu":
            values = [int(x) for x in parts[1:]]
            idle = values[3] if len(values) > 3 else 0
            total = sum(values)
            if total > 0:
                return (1.0 - idle / total) * 100.0
    except (OSError, ValueError):
        pass
    return 0.0


def _read_gpu_stats() -> dict:
    """Query nvidia-smi for GPU stats."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) >= 6:
                return {
                    "gpu_name": parts[0],
                    "gpu_memory_total_mb": int(parts[1]),
                    "gpu_memory_used_mb": int(parts[2]),
                    "gpu_memory_free_mb": int(parts[3]),
                    "gpu_utilization_percent": int(parts[4]),
                    "gpu_temperature_c": int(parts[5]),
                }
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return {}


def _read_process_memory(pid: int) -> dict[str, float]:
    """Read process memory from /proc/[pid]/status."""
    result = {"process_rss_gb": 0.0, "process_vms_gb": 0.0}
    try:
        status_path = Path(f"/proc/{pid}/status")
        with open(status_path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    result["process_rss_gb"] = kb / (1024 * 1024)
                elif line.startswith("VmSize:"):
                    kb = int(line.split()[1])
                    result["process_vms_gb"] = kb / (1024 * 1024)
    except (OSError, ValueError):
        pass
    return result


def take_snapshot(
    start_time: float,
    enable_gpu: bool = True,
    track_pid: int | None = None,
) -> ResourceSnapshot:
    """Capture a single resource snapshot."""
    now = datetime.now(UTC)
    elapsed = time.monotonic() - start_time

    meminfo = _read_meminfo()
    ram_total_kb = meminfo.get("MemTotal", 0)
    ram_available_kb = meminfo.get("MemAvailable", 0)
    ram_used_kb = ram_total_kb - ram_available_kb
    swap_total_kb = meminfo.get("SwapTotal", 0)
    swap_free_kb = meminfo.get("SwapFree", 0)
    swap_used_kb = swap_total_kb - swap_free_kb

    load1, load5, load15 = _read_loadavg()
    cpu_pct = _read_cpu_percent()

    snapshot = ResourceSnapshot(
        timestamp=now.isoformat(),
        elapsed_sec=round(elapsed, 1),
        ram_total_gb=round(ram_total_kb / (1024 * 1024), 2),
        ram_used_gb=round(ram_used_kb / (1024 * 1024), 2),
        ram_available_gb=round(ram_available_kb / (1024 * 1024), 2),
        ram_percent=round(ram_used_kb / ram_total_kb * 100, 1) if ram_total_kb else 0.0,
        swap_total_gb=round(swap_total_kb / (1024 * 1024), 2),
        swap_used_gb=round(swap_used_kb / (1024 * 1024), 2),
        swap_percent=round(swap_used_kb / swap_total_kb * 100, 1) if swap_total_kb else 0.0,
        cpu_percent=round(cpu_pct, 1),
        load_avg_1m=load1,
        load_avg_5m=load5,
        load_avg_15m=load15,
    )

    if enable_gpu:
        gpu = _read_gpu_stats()
        snapshot.gpu_name = gpu.get("gpu_name", "")
        snapshot.gpu_memory_total_mb = gpu.get("gpu_memory_total_mb", 0)
        snapshot.gpu_memory_used_mb = gpu.get("gpu_memory_used_mb", 0)
        snapshot.gpu_memory_free_mb = gpu.get("gpu_memory_free_mb", 0)
        snapshot.gpu_utilization_percent = gpu.get("gpu_utilization_percent", 0)
        snapshot.gpu_temperature_c = gpu.get("gpu_temperature_c", 0)

    if track_pid is not None:
        proc_mem = _read_process_memory(track_pid)
        snapshot.process_rss_gb = proc_mem["process_rss_gb"]
        snapshot.process_vms_gb = proc_mem["process_vms_gb"]

    return snapshot


class SystemMonitor:
    """Background system resource monitor with CSV logging.

    Parameters
    ----------
    config:
        Monitor configuration.
    on_memory_warn:
        Optional callback when memory exceeds warn threshold.
        Receives the ResourceSnapshot.
    on_memory_abort:
        Optional callback when memory exceeds abort threshold.
        Receives the ResourceSnapshot. Should trigger graceful shutdown.
    """

    def __init__(
        self,
        config: MonitorConfig | None = None,
        on_memory_warn: callable | None = None,
        on_memory_abort: callable | None = None,
    ):
        self.config = config or MonitorConfig()
        self.on_memory_warn = on_memory_warn
        self.on_memory_abort = on_memory_abort
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time = 0.0
        self._csv_path: Path | None = None
        self._peak_ram_gb = 0.0
        self._peak_gpu_mb = 0
        self._snapshot_count = 0
        self._warn_count = 0

    @property
    def peak_ram_gb(self) -> float:
        """Peak RAM usage observed during monitoring."""
        return self._peak_ram_gb

    @property
    def peak_gpu_mb(self) -> int:
        """Peak GPU memory usage observed during monitoring."""
        return self._peak_gpu_mb

    def start(self) -> None:
        """Start the monitoring background thread."""
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self.config.log_dir / "system_metrics.csv"
        self._start_time = time.monotonic()
        self._stop_event.clear()

        # Write CSV header
        sample = take_snapshot(self._start_time, enable_gpu=False)
        fieldnames = list(asdict(sample).keys())
        with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

        # Write initial system info
        info_path = self.config.log_dir / "system_info.json"
        initial = take_snapshot(
            self._start_time,
            enable_gpu=self.config.enable_gpu,
        )
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "start_time": initial.timestamp,
                    "ram_total_gb": initial.ram_total_gb,
                    "swap_total_gb": initial.swap_total_gb,
                    "gpu_name": initial.gpu_name,
                    "gpu_memory_total_mb": initial.gpu_memory_total_mb,
                    "config": {
                        "interval_sec": self.config.interval_sec,
                        "memory_warn_gb": self.config.memory_warn_gb,
                        "memory_abort_gb": self.config.memory_abort_gb,
                        "track_pid": self.config.track_pid,
                    },
                },
                f,
                indent=2,
            )

        self._thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="system-monitor",
        )
        self._thread.start()
        logger.info(
            "System monitor started (interval=%.1fs, log=%s)",
            self.config.interval_sec,
            self._csv_path,
        )

    def stop(self) -> dict:
        """Stop the monitor and return summary statistics."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None

        summary = {
            "snapshots": self._snapshot_count,
            "peak_ram_gb": self._peak_ram_gb,
            "peak_gpu_mb": self._peak_gpu_mb,
            "warn_count": self._warn_count,
            "log_file": str(self._csv_path),
        }

        # Write summary
        if self.config.log_dir.exists():
            summary_path = self.config.log_dir / "monitor_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

        logger.info(
            "System monitor stopped: %d snapshots, peak_ram=%.1f GB, peak_gpu=%d MB",
            self._snapshot_count,
            self._peak_ram_gb,
            self._peak_gpu_mb,
        )
        return summary

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop_event.is_set():
            try:
                snapshot = take_snapshot(
                    self._start_time,
                    enable_gpu=self.config.enable_gpu,
                    track_pid=self.config.track_pid,
                )
                self._snapshot_count += 1

                # Track peaks
                if snapshot.ram_used_gb > self._peak_ram_gb:
                    self._peak_ram_gb = snapshot.ram_used_gb
                if snapshot.gpu_memory_used_mb > self._peak_gpu_mb:
                    self._peak_gpu_mb = snapshot.gpu_memory_used_mb

                # Write to CSV
                if self._csv_path is not None:
                    with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=list(asdict(snapshot).keys()))
                        writer.writerow(asdict(snapshot))

                # Memory threshold checks
                if snapshot.ram_used_gb >= self.config.memory_abort_gb:
                    logger.critical(
                        "MEMORY ABORT THRESHOLD: %.1f GB used (limit: %.1f GB)",
                        snapshot.ram_used_gb,
                        self.config.memory_abort_gb,
                    )
                    if self.on_memory_abort:
                        self.on_memory_abort(snapshot)
                elif snapshot.ram_used_gb >= self.config.memory_warn_gb:
                    self._warn_count += 1
                    logger.warning(
                        "MEMORY WARNING: %.1f GB used (warn: %.1f GB) [#%d]",
                        snapshot.ram_used_gb,
                        self.config.memory_warn_gb,
                        self._warn_count,
                    )
                    if self.on_memory_warn:
                        self.on_memory_warn(snapshot)

                # Console status every 12 snapshots (~60 sec at 5s interval)
                if self._snapshot_count % 12 == 0:
                    gpu_info = ""
                    if snapshot.gpu_memory_used_mb > 0:
                        gpu_info = f" | GPU: {snapshot.gpu_memory_used_mb}/{snapshot.gpu_memory_total_mb} MB"
                    proc_info = ""
                    if snapshot.process_rss_gb > 0:
                        proc_info = f" | Process RSS: {snapshot.process_rss_gb:.1f} GB"
                    logger.info(
                        "[MONITOR] RAM: %.1f/%.1f GB (%.0f%%) | Swap: %.1f/%.1f GB%s%s",
                        snapshot.ram_used_gb,
                        snapshot.ram_total_gb,
                        snapshot.ram_percent,
                        snapshot.swap_used_gb,
                        snapshot.swap_total_gb,
                        gpu_info,
                        proc_info,
                    )

            except Exception:
                logger.exception("Monitor snapshot failed")

            self._stop_event.wait(self.config.interval_sec)


def main() -> None:
    """Standalone monitoring entry point."""
    import argparse
    import os
    import signal

    parser = argparse.ArgumentParser(description="System resource monitor")
    parser.add_argument("--interval", type=float, default=5.0, help="Sampling interval (seconds)")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/monitor"), help="Log output directory")
    parser.add_argument("--memory-warn-gb", type=float, default=50.0, help="RAM warning threshold (GB)")
    parser.add_argument("--memory-abort-gb", type=float, default=58.0, help="RAM abort threshold (GB)")
    parser.add_argument("--track-pid", type=int, default=None, help="PID to track (default: self)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    config = MonitorConfig(
        interval_sec=args.interval,
        log_dir=args.log_dir,
        memory_warn_gb=args.memory_warn_gb,
        memory_abort_gb=args.memory_abort_gb,
        track_pid=args.track_pid or os.getpid(),
    )

    monitor = SystemMonitor(config)
    monitor.start()

    # Handle graceful shutdown
    stop = threading.Event()

    def _signal_handler(signum, frame):
        logger.info("Signal %d received, stopping monitor...", signum)
        stop.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    logger.info("Monitor running. Press Ctrl+C to stop.")
    stop.wait()
    summary = monitor.stop()
    logger.info("Final summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
