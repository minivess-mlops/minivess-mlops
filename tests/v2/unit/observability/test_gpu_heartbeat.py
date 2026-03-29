"""Tests for GPU heartbeat monitor (Phase 2, Task 2.1).

Background thread that periodically checks GPU utilization and writes
heartbeat.json. Logs ERROR when GPU util below threshold for > alert_after_s.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestGpuHeartbeatMonitorImportable:
    def test_class_exists(self) -> None:
        from minivess.observability.gpu_heartbeat import GpuHeartbeatMonitor

        assert GpuHeartbeatMonitor is not None


class TestGpuHeartbeatMonitorContextManager:
    """Must start/stop a daemon thread as context manager."""

    def test_context_manager_protocol(self, tmp_path: Path) -> None:
        from minivess.observability.gpu_heartbeat import GpuHeartbeatMonitor

        monitor = GpuHeartbeatMonitor(
            output_dir=tmp_path,
            check_interval_s=0.1,
            low_util_threshold_pct=5,
            alert_after_s=1,
        )
        with monitor:
            assert monitor.is_alive()
            time.sleep(0.3)
        # After exit, thread should be stopped
        time.sleep(0.2)
        assert not monitor.is_alive()

    def test_daemon_thread(self, tmp_path: Path) -> None:
        from minivess.observability.gpu_heartbeat import GpuHeartbeatMonitor

        monitor = GpuHeartbeatMonitor(
            output_dir=tmp_path,
            check_interval_s=0.1,
        )
        with monitor:
            assert monitor._thread.daemon is True


class TestGpuHeartbeatWritesFile:
    """Must write heartbeat.json to output_dir."""

    def test_heartbeat_json_created(self, tmp_path: Path) -> None:
        from minivess.observability.gpu_heartbeat import GpuHeartbeatMonitor

        monitor = GpuHeartbeatMonitor(
            output_dir=tmp_path,
            check_interval_s=0.1,
        )
        with monitor:
            time.sleep(0.3)

        hb_path = tmp_path / "heartbeat.json"
        assert hb_path.exists(), "heartbeat.json not created"

    def test_heartbeat_json_has_required_fields(self, tmp_path: Path) -> None:
        from minivess.observability.gpu_heartbeat import GpuHeartbeatMonitor

        monitor = GpuHeartbeatMonitor(
            output_dir=tmp_path,
            check_interval_s=0.1,
        )
        with monitor:
            time.sleep(0.3)

        hb = json.loads((tmp_path / "heartbeat.json").read_text(encoding="utf-8"))
        assert "timestamp" in hb
        assert "gpu_util_pct" in hb
        assert "gpu_memory_used_mb" in hb
        assert "status" in hb


class TestGpuHeartbeatGracefulWithoutGpu:
    """No-op when pynvml/GPU unavailable (CPU-only environments)."""

    @patch("minivess.observability.gpu_heartbeat._get_gpu_snapshot")
    def test_graceful_without_gpu(self, mock_snapshot: MagicMock, tmp_path: Path) -> None:
        from minivess.observability.gpu_heartbeat import GpuHeartbeatMonitor

        mock_snapshot.return_value = {
            "gpu_util_pct": 0,
            "gpu_memory_used_mb": 0,
            "gpu_temp_c": 0,
            "status": "no_gpu",
        }
        monitor = GpuHeartbeatMonitor(
            output_dir=tmp_path,
            check_interval_s=0.1,
        )
        with monitor:
            time.sleep(0.3)
        # Should not crash — graceful degradation
        hb = json.loads((tmp_path / "heartbeat.json").read_text(encoding="utf-8"))
        assert hb["status"] == "no_gpu"
