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
    gpu_mem_bw_util_pct: int = 0  # memory interface busy % (utilization.memory)
    gpu_power_w: float = 0.0  # board power draw in Watts (power.draw)
    gpu_sm_clock_mhz: int = 0  # SM clock — drop vs base = thermal throttle
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
    # GPU efficiency thresholds for STARVE / THROTTLE warnings
    gpu_starve_util_pct: int = (
        60  # GPU util below this AND cpu above cpu threshold → STARVE
    )
    gpu_starve_cpu_pct: float = (
        85.0  # CPU % above this (combined with low GPU util) → STARVE
    )
    gpu_throttle_temp_c: int = 87  # Die temp above this → THROTTLE
    gpu_throttle_clock_ratio: float = 0.92  # SM clock below base * ratio → THROTTLE


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


def _parse_int_or_zero(value: str) -> int:
    """Parse nvidia-smi integer field; return 0 for [N/A] or unparseable."""
    v = value.strip()
    if v in ("[N/A]", "N/A", ""):
        return 0
    try:
        return int(float(v))
    except ValueError:
        return 0


def _parse_float_or_zero(value: str) -> float:
    """Parse nvidia-smi float field; return 0.0 for [N/A] or unparseable."""
    v = value.strip()
    if v in ("[N/A]", "N/A", ""):
        return 0.0
    try:
        return float(v)
    except ValueError:
        return 0.0


def _read_gpu_stats() -> dict:
    """Query nvidia-smi for GPU stats including efficiency metrics.

    Extended query adds: utilization.memory (Mem-BW%), power.draw (Watts),
    clocks.sm (SM clock MHz for throttle detection).
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,"
                "utilization.gpu,temperature.gpu,"
                "utilization.memory,power.draw,clocks.sm",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) >= 6:
                stats: dict = {
                    "gpu_name": parts[0],
                    "gpu_memory_total_mb": _parse_int_or_zero(parts[1]),
                    "gpu_memory_used_mb": _parse_int_or_zero(parts[2]),
                    "gpu_memory_free_mb": _parse_int_or_zero(parts[3]),
                    "gpu_utilization_percent": _parse_int_or_zero(parts[4]),
                    "gpu_temperature_c": _parse_int_or_zero(parts[5]),
                }
                if len(parts) >= 9:
                    stats["gpu_mem_bw_util_pct"] = _parse_int_or_zero(parts[6])
                    stats["gpu_power_w"] = _parse_float_or_zero(parts[7])
                    stats["gpu_sm_clock_mhz"] = _parse_int_or_zero(parts[8])
                return stats
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return {}


def _query_gpu_max_sm_clock() -> int:
    """Query the GPU's maximum (base) SM clock for throttle detection.

    Returns 0 if nvidia-smi is unavailable or the query fails.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=clocks.max.sm",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            val = result.stdout.strip()
            if val and val not in ("[N/A]", "N/A"):
                return int(float(val))
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return 0


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
        swap_percent=round(swap_used_kb / swap_total_kb * 100, 1)
        if swap_total_kb
        else 0.0,
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
        snapshot.gpu_mem_bw_util_pct = gpu.get("gpu_mem_bw_util_pct", 0)
        snapshot.gpu_power_w = gpu.get("gpu_power_w", 0.0)
        snapshot.gpu_sm_clock_mhz = gpu.get("gpu_sm_clock_mhz", 0)

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
        self._jsonl_path: Path | None = None
        self._peak_ram_gb = 0.0
        self._peak_gpu_mb = 0
        self._snapshot_count = 0
        self._warn_count = 0
        # Ring buffer for epoch_summary() (drained each epoch)
        self._snapshot_buffer: list[ResourceSnapshot] = []
        self._buffer_lock = threading.Lock()
        self._latest_snapshot: ResourceSnapshot | None = None
        # GPU base (max) SM clock for throttle detection; populated in start()
        self._gpu_base_sm_clock_mhz: int = 0
        # One-shot warning flags (each warning fires at most once per session)
        self._starve_warned = False
        self._throttle_warned = False

    @property
    def peak_ram_gb(self) -> float:
        """Peak RAM usage observed during monitoring."""
        return self._peak_ram_gb

    @property
    def peak_gpu_mb(self) -> int:
        """Peak GPU memory usage observed during monitoring."""
        return self._peak_gpu_mb

    @property
    def csv_path(self) -> Path | None:
        """Path to the CSV metrics file (set after start())."""
        return self._csv_path

    @property
    def jsonl_path(self) -> Path | None:
        """Path to the JSONL metrics file (set after start())."""
        return self._jsonl_path

    def start(self) -> None:
        """Start the monitoring background thread."""
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self.config.log_dir / "system_metrics.csv"
        self._jsonl_path = self.config.log_dir / "system_metrics.jsonl"
        self._start_time = time.monotonic()
        self._stop_event.clear()
        # Query GPU max SM clock once (baseline for throttle detection)
        if self.config.enable_gpu:
            self._gpu_base_sm_clock_mhz = _query_gpu_max_sm_clock()

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

                # Update ring buffer and latest snapshot (thread-safe)
                with self._buffer_lock:
                    self._snapshot_buffer.append(snapshot)
                    self._latest_snapshot = snapshot
                    # Bootstrap base SM clock from first real GPU reading
                    if (
                        self._gpu_base_sm_clock_mhz == 0
                        and snapshot.gpu_sm_clock_mhz > 0
                    ):
                        self._gpu_base_sm_clock_mhz = snapshot.gpu_sm_clock_mhz

                # Write to CSV
                if self._csv_path is not None:
                    with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(
                            f, fieldnames=list(asdict(snapshot).keys())
                        )
                        writer.writerow(asdict(snapshot))

                # Write to JSONL (machine-parseable; no regex needed to consume)
                if self._jsonl_path is not None:
                    with open(self._jsonl_path, "a", encoding="utf-8") as jf:
                        jf.write(json.dumps(asdict(snapshot)) + "\n")

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

            except Exception:
                logger.exception("Monitor snapshot failed")

            self._stop_event.wait(self.config.interval_sec)

    def get_latest_snapshot(self) -> ResourceSnapshot | None:
        """Return the most recent snapshot without draining the ring buffer."""
        with self._buffer_lock:
            return self._latest_snapshot

    def gpu_status(self, snap: ResourceSnapshot) -> str:
        """Classify a snapshot as [OK], [STARVE], or [THROTTLE].

        Parameters
        ----------
        snap:
            The snapshot to classify.

        Returns
        -------
        str
            ``[OK]`` — GPU running efficiently.
            ``[STARVE]`` — DataLoader bottleneck (low GPU util + high CPU).
            ``[THROTTLE]`` — Thermal or power throttle (high temp or SM clock drop).
        """
        low_util = snap.gpu_utilization_percent < self.config.gpu_starve_util_pct
        high_cpu = snap.cpu_percent > self.config.gpu_starve_cpu_pct
        if low_util and high_cpu:
            return "[STARVE]"

        over_temp = snap.gpu_temperature_c > self.config.gpu_throttle_temp_c
        base = self._gpu_base_sm_clock_mhz
        clock_dropped = (
            base > 0
            and snap.gpu_sm_clock_mhz > 0
            and snap.gpu_sm_clock_mhz < base * self.config.gpu_throttle_clock_ratio
        )
        if over_temp or clock_dropped:
            return "[THROTTLE]"

        return "[OK]"

    def epoch_summary(self) -> dict[str, float]:
        """Drain the ring buffer, aggregate GPU efficiency metrics, and return.

        Emits [STARVE] or [THROTTLE] warnings (once per condition per session)
        when thresholds are exceeded.

        Returns
        -------
        dict[str, float]
            Aggregated metrics keyed for MLflow logging::

                (
                    gpu_util_pct_mean,
                    gpu_util_pct_min,
                )
                (
                    gpu_temp_c_max,
                    gpu_sm_clock_mhz_min,
                )
                gpu_power_w_mean, cpu_pct_mean
        """
        with self._buffer_lock:
            snapshots = self._snapshot_buffer[:]
            self._snapshot_buffer.clear()

        if not snapshots:
            return {}

        # Filter out zero-util snapshots (GPU unavailable / monitor just started)
        gpu_snaps = [s for s in snapshots if s.gpu_utilization_percent > 0]

        gpu_utils = [s.gpu_utilization_percent for s in gpu_snaps]
        gpu_temps = [s.gpu_temperature_c for s in gpu_snaps if s.gpu_temperature_c > 0]
        gpu_clocks = [s.gpu_sm_clock_mhz for s in gpu_snaps if s.gpu_sm_clock_mhz > 0]
        gpu_powers = [s.gpu_power_w for s in gpu_snaps if s.gpu_power_w > 0.0]
        cpu_pcts = [s.cpu_percent for s in snapshots]

        metrics: dict[str, float] = {}
        if gpu_utils:
            metrics["gpu_util_pct_mean"] = sum(gpu_utils) / len(gpu_utils)
            metrics["gpu_util_pct_min"] = float(min(gpu_utils))
        if gpu_temps:
            metrics["gpu_temp_c_max"] = float(max(gpu_temps))
        if gpu_clocks:
            metrics["gpu_sm_clock_mhz_min"] = float(min(gpu_clocks))
        if gpu_powers:
            metrics["gpu_power_w_mean"] = sum(gpu_powers) / len(gpu_powers)
        if cpu_pcts:
            metrics["cpu_pct_mean"] = sum(cpu_pcts) / len(cpu_pcts)

        # T6: [STARVE] warning — DataLoader is the bottleneck (once per session)
        if gpu_utils and cpu_pcts and not self._starve_warned:
            avg_util = metrics.get("gpu_util_pct_mean", 100.0)
            avg_cpu = metrics.get("cpu_pct_mean", 0.0)
            if (
                avg_util < self.config.gpu_starve_util_pct
                and avg_cpu > self.config.gpu_starve_cpu_pct
            ):
                self._starve_warned = True
                logger.warning(
                    "[STARVE] GPU-util=%.0f%% cpu=%.0f%% — DataLoader bottleneck. "
                    "Try: num_workers up, pin_memory=True, persistent_workers=True, "
                    "cache_rate up",
                    avg_util,
                    avg_cpu,
                )

        # T7: [THROTTLE] warning — thermal or power throttle (once per session)
        if (gpu_temps or gpu_clocks) and not self._throttle_warned:
            max_temp = metrics.get("gpu_temp_c_max", 0.0)
            min_clock = metrics.get("gpu_sm_clock_mhz_min", 0.0)
            base = float(self._gpu_base_sm_clock_mhz)
            throttle_by_temp = max_temp > self.config.gpu_throttle_temp_c
            throttle_by_clock = (
                base > 0
                and min_clock > 0
                and min_clock < base * self.config.gpu_throttle_clock_ratio
            )
            if throttle_by_temp or throttle_by_clock:
                self._throttle_warned = True
                clock_pct = (min_clock / base * 100.0) if base > 0 else 0.0
                logger.warning(
                    "[THROTTLE] temp=%.0fC sm_clock=%.0fMHz (base=%.0fMHz, %.0f%%) — "
                    "GPU FLOPS silently degraded. Check cooling.",
                    max_temp,
                    min_clock,
                    base,
                    clock_pct,
                )

        return metrics


def main() -> None:
    """Standalone monitoring entry point."""
    import argparse
    import os
    import signal

    parser = argparse.ArgumentParser(description="System resource monitor")
    parser.add_argument(
        "--interval", type=float, default=5.0, help="Sampling interval (seconds)"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/monitor"),
        help="Log output directory",
    )
    parser.add_argument(
        "--memory-warn-gb", type=float, default=50.0, help="RAM warning threshold (GB)"
    )
    parser.add_argument(
        "--memory-abort-gb", type=float, default=58.0, help="RAM abort threshold (GB)"
    )
    parser.add_argument(
        "--track-pid", type=int, default=None, help="PID to track (default: self)"
    )
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
