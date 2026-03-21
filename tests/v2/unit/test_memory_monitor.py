"""Tests for MemoryMonitor — process-level memory guardrails.

Ported from foundation-PLR (streaming_duckdb_export.py:217-284).
See: .claude/metalearning/2026-03-21-ram-crash-biostatistics-test.md
"""

from __future__ import annotations

from unittest.mock import patch


class TestMemoryMonitorCheck:
    """MemoryMonitor.check() returns (usage_gb, status)."""

    def test_check_returns_tuple(self) -> None:
        from minivess.observability.memory_monitor import MemoryMonitor

        monitor = MemoryMonitor()
        usage_gb, status = monitor.check()
        assert isinstance(usage_gb, float)
        assert status in ("ok", "warning", "critical", "unknown")

    def test_check_reports_positive_usage(self) -> None:
        from minivess.observability.memory_monitor import MemoryMonitor

        monitor = MemoryMonitor()
        usage_gb, _status = monitor.check()
        # A running Python process always uses some memory
        assert usage_gb > 0.0

    def test_warning_threshold(self) -> None:
        from minivess.observability.memory_monitor import MemoryMonitor

        # Set threshold absurdly low so current usage triggers warning
        monitor = MemoryMonitor(warning_threshold_gb=0.001, critical_threshold_gb=999.0)
        _usage, status = monitor.check()
        assert status == "warning"

    def test_critical_threshold(self) -> None:
        from minivess.observability.memory_monitor import MemoryMonitor

        # Set threshold absurdly low so current usage triggers critical
        monitor = MemoryMonitor(
            warning_threshold_gb=0.0001, critical_threshold_gb=0.001
        )
        _usage, status = monitor.check()
        assert status == "critical"

    def test_ok_with_high_thresholds(self) -> None:
        from minivess.observability.memory_monitor import MemoryMonitor

        monitor = MemoryMonitor(warning_threshold_gb=999.0, critical_threshold_gb=999.0)
        _usage, status = monitor.check()
        assert status == "ok"


class TestMemoryMonitorEnforce:
    """MemoryMonitor.enforce() triggers GC at critical level."""

    def test_enforce_calls_gc_at_critical(self) -> None:
        from minivess.observability.memory_monitor import MemoryMonitor

        monitor = MemoryMonitor(
            warning_threshold_gb=0.0001, critical_threshold_gb=0.001
        )
        with patch("gc.collect") as mock_gc:
            monitor.enforce()
            mock_gc.assert_called()

    def test_enforce_does_not_gc_when_ok(self) -> None:
        from minivess.observability.memory_monitor import MemoryMonitor

        monitor = MemoryMonitor(warning_threshold_gb=999.0, critical_threshold_gb=999.0)
        with patch("gc.collect") as mock_gc:
            monitor.enforce()
            mock_gc.assert_not_called()


class TestMemoryMonitorGetUsageGb:
    """MemoryMonitor.get_usage_gb() convenience method."""

    def test_get_usage_returns_float(self) -> None:
        from minivess.observability.memory_monitor import MemoryMonitor

        monitor = MemoryMonitor()
        assert isinstance(monitor.get_usage_gb(), float)
        assert monitor.get_usage_gb() > 0.0


class TestMemoryMonitorPsutilFallback:
    """When psutil is not available, check returns 'unknown'."""

    def test_fallback_without_psutil(self) -> None:
        from minivess.observability.memory_monitor import MemoryMonitor

        monitor = MemoryMonitor()
        with patch.dict("sys.modules", {"psutil": None}):
            # Force re-import path that fails psutil
            usage, status = monitor._check_without_psutil()
            assert usage == 0.0
            assert status == "unknown"
