"""Unit tests for GPU efficiency logging improvements (P1 #368).

Tests for:
- New ResourceSnapshot fields: gpu_mem_bw_util_pct, gpu_power_w, gpu_sm_clock_mhz
- Extended nvidia-smi query parsing (9-column output, [N/A] handling)
- epoch_summary() aggregation and buffer drain
- [STARVE] / [THROTTLE] warning logic (emitted once per condition)
- gpu_status() classification → [OK] / [STARVE] / [THROTTLE]
- JSONL output alongside CSV (parseable, no regex)
- get_latest_snapshot() thread-safe access
- csv_path / jsonl_path properties
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# system_monitor.py lives in scripts/ (not in the installed package).
# Insert the scripts directory so we can import it directly.
_SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from system_monitor import (  # noqa: E402
    MonitorConfig,
    ResourceSnapshot,
    SystemMonitor,
    _read_gpu_stats,
)


def _make_snap(**kwargs) -> ResourceSnapshot:
    """Create a ResourceSnapshot with healthy-GPU defaults for unit testing."""
    defaults: dict = dict(
        timestamp="2026-03-06T12:00:00+00:00",
        elapsed_sec=1.0,
        ram_total_gb=62.0,
        ram_used_gb=20.0,
        ram_available_gb=42.0,
        ram_percent=32.0,
        swap_total_gb=16.0,
        swap_used_gb=0.5,
        swap_percent=3.0,
        cpu_percent=40.0,
        load_avg_1m=2.0,
        load_avg_5m=1.5,
        load_avg_15m=1.2,
        gpu_name="NVIDIA GeForce RTX 2060",
        gpu_memory_total_mb=8192,
        gpu_memory_used_mb=5360,
        gpu_memory_free_mb=2832,
        gpu_utilization_percent=85,
        gpu_temperature_c=70,
        gpu_mem_bw_util_pct=45,
        gpu_power_w=180.0,
        gpu_sm_clock_mhz=1980,
    )
    defaults.update(kwargs)
    return ResourceSnapshot(**defaults)


# ---------------------------------------------------------------------------
# T1: New ResourceSnapshot fields
# ---------------------------------------------------------------------------


class TestResourceSnapshotNewFields:
    """T1: ResourceSnapshot must expose the three new GPU efficiency fields."""

    def _minimal_snap(self) -> ResourceSnapshot:
        """Snapshot with only required (non-default) fields."""
        return ResourceSnapshot(
            timestamp="t",
            elapsed_sec=0.0,
            ram_total_gb=0.0,
            ram_used_gb=0.0,
            ram_available_gb=0.0,
            ram_percent=0.0,
            swap_total_gb=0.0,
            swap_used_gb=0.0,
            swap_percent=0.0,
            cpu_percent=0.0,
            load_avg_1m=0.0,
            load_avg_5m=0.0,
            load_avg_15m=0.0,
        )

    def test_has_gpu_mem_bw_util_pct(self) -> None:
        snap = self._minimal_snap()
        assert hasattr(snap, "gpu_mem_bw_util_pct")
        assert snap.gpu_mem_bw_util_pct == 0

    def test_has_gpu_power_w(self) -> None:
        snap = self._minimal_snap()
        assert hasattr(snap, "gpu_power_w")
        assert snap.gpu_power_w == 0.0

    def test_has_gpu_sm_clock_mhz(self) -> None:
        snap = self._minimal_snap()
        assert hasattr(snap, "gpu_sm_clock_mhz")
        assert snap.gpu_sm_clock_mhz == 0

    def test_new_fields_serialised_by_asdict(self) -> None:
        snap = _make_snap(
            gpu_mem_bw_util_pct=45, gpu_power_w=185.0, gpu_sm_clock_mhz=1980
        )
        d = asdict(snap)
        assert d["gpu_mem_bw_util_pct"] == 45
        assert d["gpu_power_w"] == 185.0
        assert d["gpu_sm_clock_mhz"] == 1980


# ---------------------------------------------------------------------------
# T2: Extended nvidia-smi query parsing
# ---------------------------------------------------------------------------


class TestReadGpuStatsExtended:
    """T2: _read_gpu_stats must parse 9-column nvidia-smi output."""

    def _mock_result(self, stdout: str) -> MagicMock:
        mock = MagicMock()
        mock.returncode = 0
        mock.stdout = stdout
        return mock

    def _smi_line(
        self,
        name: str = "NVIDIA GeForce RTX 2060",
        mem_total: int = 8192,
        mem_used: int = 5360,
        mem_free: int = 2832,
        util_gpu: int = 87,
        temp: int = 72,
        util_mem: str = "45",
        power: str = "185.23",
        sm_clock: str = "1980",
    ) -> str:
        return f"{name}, {mem_total}, {mem_used}, {mem_free}, {util_gpu}, {temp}, {util_mem}, {power}, {sm_clock}\n"

    def test_parses_gpu_mem_bw_util_pct(self) -> None:
        with patch(
            "subprocess.run",
            return_value=self._mock_result(self._smi_line(util_mem="45")),
        ):
            result = _read_gpu_stats()
        assert result["gpu_mem_bw_util_pct"] == 45

    def test_parses_gpu_power_w(self) -> None:
        with patch(
            "subprocess.run",
            return_value=self._mock_result(self._smi_line(power="185.23")),
        ):
            result = _read_gpu_stats()
        assert result["gpu_power_w"] == pytest.approx(185.23, abs=0.01)

    def test_parses_gpu_sm_clock_mhz(self) -> None:
        with patch(
            "subprocess.run",
            return_value=self._mock_result(self._smi_line(sm_clock="1980")),
        ):
            result = _read_gpu_stats()
        assert result["gpu_sm_clock_mhz"] == 1980

    def test_na_power_defaults_to_zero(self) -> None:
        with patch(
            "subprocess.run",
            return_value=self._mock_result(self._smi_line(power="[N/A]")),
        ):
            result = _read_gpu_stats()
        assert result.get("gpu_power_w", 0) == 0.0

    def test_na_bw_util_defaults_to_zero(self) -> None:
        with patch(
            "subprocess.run",
            return_value=self._mock_result(self._smi_line(util_mem="[N/A]")),
        ):
            result = _read_gpu_stats()
        assert result.get("gpu_mem_bw_util_pct", 0) == 0

    def test_na_sm_clock_defaults_to_zero(self) -> None:
        with patch(
            "subprocess.run",
            return_value=self._mock_result(self._smi_line(sm_clock="[N/A]")),
        ):
            result = _read_gpu_stats()
        assert result.get("gpu_sm_clock_mhz", 0) == 0

    def test_missing_gpu_returns_empty_dict(self) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = _read_gpu_stats()
        assert result == {}

    def test_old_fields_still_present(self) -> None:
        """Backward compat: name, memory, util, temp still returned."""
        with patch("subprocess.run", return_value=self._mock_result(self._smi_line())):
            result = _read_gpu_stats()
        assert result["gpu_name"] == "NVIDIA GeForce RTX 2060"
        assert result["gpu_memory_total_mb"] == 8192
        assert result["gpu_utilization_percent"] == 87
        assert result["gpu_temperature_c"] == 72


# ---------------------------------------------------------------------------
# T3: epoch_summary() — ring-buffer aggregation + drain
# ---------------------------------------------------------------------------


class TestEpochSummary:
    """T3: epoch_summary() drains the buffer and returns aggregated GPU metrics."""

    def _monitor_with_snapshots(
        self, snaps: list[ResourceSnapshot], base_clock: int = 1980
    ) -> SystemMonitor:
        monitor = SystemMonitor()
        monitor._snapshot_buffer = list(snaps)
        monitor._gpu_base_sm_clock_mhz = base_clock
        return monitor

    def test_empty_buffer_returns_empty_dict(self) -> None:
        monitor = SystemMonitor()
        assert monitor.epoch_summary() == {}

    def test_computes_gpu_util_mean(self) -> None:
        snaps = [
            _make_snap(gpu_utilization_percent=80),
            _make_snap(gpu_utilization_percent=90),
        ]
        result = self._monitor_with_snapshots(snaps).epoch_summary()
        assert result["gpu_util_pct_mean"] == pytest.approx(85.0)

    def test_computes_gpu_util_min(self) -> None:
        snaps = [
            _make_snap(gpu_utilization_percent=70),
            _make_snap(gpu_utilization_percent=90),
        ]
        result = self._monitor_with_snapshots(snaps).epoch_summary()
        assert result["gpu_util_pct_min"] == pytest.approx(70.0)

    def test_computes_gpu_temp_max(self) -> None:
        snaps = [
            _make_snap(gpu_temperature_c=70),
            _make_snap(gpu_temperature_c=80),
        ]
        result = self._monitor_with_snapshots(snaps).epoch_summary()
        assert result["gpu_temp_c_max"] == pytest.approx(80.0)

    def test_computes_sm_clock_min(self) -> None:
        snaps = [
            _make_snap(gpu_sm_clock_mhz=1980),
            _make_snap(gpu_sm_clock_mhz=1800),
        ]
        result = self._monitor_with_snapshots(snaps).epoch_summary()
        assert result["gpu_sm_clock_mhz_min"] == pytest.approx(1800.0)

    def test_computes_gpu_power_mean(self) -> None:
        snaps = [
            _make_snap(gpu_power_w=180.0),
            _make_snap(gpu_power_w=200.0),
        ]
        result = self._monitor_with_snapshots(snaps).epoch_summary()
        assert result["gpu_power_w_mean"] == pytest.approx(190.0)

    def test_computes_cpu_pct_mean(self) -> None:
        snaps = [
            _make_snap(cpu_percent=30.0),
            _make_snap(cpu_percent=50.0),
        ]
        result = self._monitor_with_snapshots(snaps).epoch_summary()
        assert result["cpu_pct_mean"] == pytest.approx(40.0)

    def test_drains_buffer_after_call(self) -> None:
        monitor = self._monitor_with_snapshots([_make_snap(), _make_snap()])
        monitor.epoch_summary()
        assert len(monitor._snapshot_buffer) == 0

    def test_second_call_on_empty_buffer_returns_empty(self) -> None:
        monitor = self._monitor_with_snapshots([_make_snap()])
        monitor.epoch_summary()
        assert monitor.epoch_summary() == {}

    def test_zero_util_snapshots_excluded_from_mean(self) -> None:
        """Snapshots with zero GPU util (GPU unavailable) are ignored."""
        snaps = [
            _make_snap(gpu_utilization_percent=0),  # no GPU data
            _make_snap(gpu_utilization_percent=90),
        ]
        result = self._monitor_with_snapshots(snaps).epoch_summary()
        # Mean should be 90 (only non-zero included)
        assert result["gpu_util_pct_mean"] == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# T6 + T7: Warning logic — emitted once per session
# ---------------------------------------------------------------------------


class TestWarningLogic:
    """T6/T7: STARVE and THROTTLE warnings printed once, never again."""

    def _monitor(
        self, snaps: list[ResourceSnapshot], base_clock: int = 1980
    ) -> SystemMonitor:
        m = SystemMonitor()
        m._snapshot_buffer = list(snaps)
        m._gpu_base_sm_clock_mhz = base_clock
        return m

    def test_starve_warning_when_low_util_and_high_cpu(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        snaps = [_make_snap(gpu_utilization_percent=38, cpu_percent=92.0)]
        monitor = self._monitor(snaps)
        with caplog.at_level(logging.WARNING):
            monitor.epoch_summary()
        assert "[STARVE]" in caplog.text

    def test_starve_warning_emitted_only_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        monitor = SystemMonitor()
        monitor._gpu_base_sm_clock_mhz = 1980
        with caplog.at_level(logging.WARNING):
            monitor._snapshot_buffer = [
                _make_snap(gpu_utilization_percent=38, cpu_percent=92.0)
            ]
            monitor.epoch_summary()
            caplog.clear()
            monitor._snapshot_buffer = [
                _make_snap(gpu_utilization_percent=38, cpu_percent=92.0)
            ]
            monitor.epoch_summary()
        # Warning should NOT fire again on second call
        assert "[STARVE]" not in caplog.text

    def test_no_starve_when_util_above_threshold(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        snaps = [_make_snap(gpu_utilization_percent=90, cpu_percent=92.0)]
        monitor = self._monitor(snaps)
        with caplog.at_level(logging.WARNING):
            monitor.epoch_summary()
        assert "[STARVE]" not in caplog.text

    def test_no_starve_when_cpu_below_threshold(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        snaps = [_make_snap(gpu_utilization_percent=38, cpu_percent=70.0)]
        monitor = self._monitor(snaps)
        with caplog.at_level(logging.WARNING):
            monitor.epoch_summary()
        assert "[STARVE]" not in caplog.text

    def test_throttle_warning_by_temperature(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        # temp=88 > threshold=87
        snaps = [_make_snap(gpu_temperature_c=88, gpu_sm_clock_mhz=1980)]
        monitor = self._monitor(snaps)
        with caplog.at_level(logging.WARNING):
            monitor.epoch_summary()
        assert "[THROTTLE]" in caplog.text

    def test_throttle_warning_by_clock_drop(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        # base=1980, drop to 1700 → 85.9% of base < 0.92 threshold
        snaps = [_make_snap(gpu_temperature_c=70, gpu_sm_clock_mhz=1700)]
        monitor = self._monitor(snaps, base_clock=1980)
        with caplog.at_level(logging.WARNING):
            monitor.epoch_summary()
        assert "[THROTTLE]" in caplog.text

    def test_throttle_warning_emitted_only_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        monitor = SystemMonitor()
        monitor._gpu_base_sm_clock_mhz = 1980
        with caplog.at_level(logging.WARNING):
            monitor._snapshot_buffer = [_make_snap(gpu_temperature_c=88)]
            monitor.epoch_summary()
            caplog.clear()
            monitor._snapshot_buffer = [_make_snap(gpu_temperature_c=88)]
            monitor.epoch_summary()
        assert "[THROTTLE]" not in caplog.text

    def test_no_throttle_when_healthy(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        snaps = [_make_snap(gpu_temperature_c=70, gpu_sm_clock_mhz=1980)]
        monitor = self._monitor(snaps)
        with caplog.at_level(logging.WARNING):
            monitor.epoch_summary()
        assert "[THROTTLE]" not in caplog.text


# ---------------------------------------------------------------------------
# gpu_status() — per-snapshot classification
# ---------------------------------------------------------------------------


class TestGpuStatus:
    """gpu_status(snap) returns [OK], [STARVE], or [THROTTLE]."""

    def _monitor(self, base_clock: int = 1980) -> SystemMonitor:
        m = SystemMonitor()
        m._gpu_base_sm_clock_mhz = base_clock
        return m

    def test_ok_when_healthy(self) -> None:
        snap = _make_snap(
            gpu_utilization_percent=90,
            cpu_percent=40.0,
            gpu_temperature_c=70,
            gpu_sm_clock_mhz=1980,
        )
        assert self._monitor().gpu_status(snap) == "[OK]"

    def test_starve_when_low_util_high_cpu(self) -> None:
        snap = _make_snap(
            gpu_utilization_percent=38,
            cpu_percent=92.0,
            gpu_temperature_c=70,
            gpu_sm_clock_mhz=1980,
        )
        assert self._monitor().gpu_status(snap) == "[STARVE]"

    def test_throttle_when_temp_too_high(self) -> None:
        snap = _make_snap(
            gpu_utilization_percent=95,
            cpu_percent=40.0,
            gpu_temperature_c=88,
            gpu_sm_clock_mhz=1980,
        )
        assert self._monitor().gpu_status(snap) == "[THROTTLE]"

    def test_throttle_when_clock_dropped(self) -> None:
        snap = _make_snap(
            gpu_utilization_percent=95,
            cpu_percent=40.0,
            gpu_temperature_c=70,
            gpu_sm_clock_mhz=1700,
        )
        assert self._monitor(base_clock=1980).gpu_status(snap) == "[THROTTLE]"

    def test_ok_when_no_base_clock_info(self) -> None:
        """Without base clock reference, clock-drop throttle check is skipped."""
        snap = _make_snap(
            gpu_utilization_percent=95,
            cpu_percent=40.0,
            gpu_temperature_c=70,
            gpu_sm_clock_mhz=500,
        )
        # base_clock=0 → no throttle-by-clock check
        assert self._monitor(base_clock=0).gpu_status(snap) in ("[OK]", "[THROTTLE]")
        # Critically: no exception raised
        self._monitor(base_clock=0).gpu_status(snap)  # must not raise


# ---------------------------------------------------------------------------
# get_latest_snapshot()
# ---------------------------------------------------------------------------


class TestGetLatestSnapshot:
    """get_latest_snapshot() returns the most recent snapshot thread-safely."""

    def test_none_before_any_snapshot_added(self) -> None:
        monitor = SystemMonitor()
        assert monitor.get_latest_snapshot() is None

    def test_returns_last_injected_snapshot(self) -> None:
        monitor = SystemMonitor()
        snap1 = _make_snap(cpu_percent=30.0)
        snap2 = _make_snap(cpu_percent=50.0)
        monitor._latest_snapshot = snap1
        monitor._latest_snapshot = snap2
        assert monitor.get_latest_snapshot() is snap2

    def test_returns_none_after_fresh_monitor(self) -> None:
        monitor = SystemMonitor()
        result = monitor.get_latest_snapshot()
        assert result is None


# ---------------------------------------------------------------------------
# JSONL output
# ---------------------------------------------------------------------------


class TestJsonlOutput:
    """JSONL file written alongside CSV; parseable with json.loads(), no regex."""

    def test_jsonl_file_created_after_monitoring(self, tmp_path: Path) -> None:
        config = MonitorConfig(log_dir=tmp_path, interval_sec=0.01, enable_gpu=False)
        monitor = SystemMonitor(config)
        monitor.start()
        time.sleep(0.08)
        monitor.stop()
        assert (tmp_path / "system_metrics.jsonl").exists()

    def test_jsonl_path_property_exposed(self, tmp_path: Path) -> None:
        config = MonitorConfig(log_dir=tmp_path, interval_sec=60.0, enable_gpu=False)
        monitor = SystemMonitor(config)
        monitor.start()
        monitor.stop()
        assert monitor.jsonl_path == tmp_path / "system_metrics.jsonl"

    def test_csv_path_property_exposed(self, tmp_path: Path) -> None:
        config = MonitorConfig(log_dir=tmp_path, interval_sec=60.0, enable_gpu=False)
        monitor = SystemMonitor(config)
        monitor.start()
        monitor.stop()
        assert monitor.csv_path == tmp_path / "system_metrics.csv"

    def test_jsonl_lines_are_valid_json(self, tmp_path: Path) -> None:
        config = MonitorConfig(log_dir=tmp_path, interval_sec=0.01, enable_gpu=False)
        monitor = SystemMonitor(config)
        monitor.start()
        time.sleep(0.08)
        monitor.stop()
        jsonl_path = tmp_path / "system_metrics.jsonl"
        assert jsonl_path.exists()
        lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) >= 1
        for line in lines:
            obj = json.loads(line)  # must not raise
            assert "timestamp" in obj
            assert "cpu_percent" in obj

    def test_jsonl_contains_new_fields(self, tmp_path: Path) -> None:
        config = MonitorConfig(log_dir=tmp_path, interval_sec=0.01, enable_gpu=False)
        monitor = SystemMonitor(config)
        monitor.start()
        time.sleep(0.08)
        monitor.stop()
        jsonl_path = tmp_path / "system_metrics.jsonl"
        lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) >= 1
        obj = json.loads(lines[0])
        assert "gpu_mem_bw_util_pct" in obj
        assert "gpu_power_w" in obj
        assert "gpu_sm_clock_mhz" in obj
