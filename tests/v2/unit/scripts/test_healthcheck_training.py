"""Tests for training health check script (Phase 4, Task 4.1)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


class TestHealthcheckFreshHeartbeat:
    def test_healthy_when_heartbeat_fresh(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from scripts.healthcheck_training import check_heartbeat_health

        monkeypatch.setenv("LOGS_DIR", str(tmp_path))
        monkeypatch.setenv("HEALTH_GRACE_PERIOD_MINUTES", "0")
        monkeypatch.setenv("STALL_THRESHOLD_MINUTES", "30")

        hb = {"timestamp": datetime.now(UTC).isoformat(), "status": "training"}
        (tmp_path / "heartbeat.json").write_text(json.dumps(hb), encoding="utf-8")

        healthy, msg = check_heartbeat_health()
        assert healthy is True
        assert "fresh" in msg.lower()


class TestHealthcheckStaleHeartbeat:
    def test_unhealthy_when_heartbeat_stale(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from scripts.healthcheck_training import check_heartbeat_health

        monkeypatch.setenv("LOGS_DIR", str(tmp_path))
        monkeypatch.setenv("HEALTH_GRACE_PERIOD_MINUTES", "0")
        monkeypatch.setenv("STALL_THRESHOLD_MINUTES", "5")

        old_ts = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
        hb = {"timestamp": old_ts, "status": "training"}
        (tmp_path / "heartbeat.json").write_text(json.dumps(hb), encoding="utf-8")

        healthy, msg = check_heartbeat_health()
        assert healthy is False
        assert "stale" in msg.lower()


class TestHealthcheckMissingHeartbeat:
    def test_unhealthy_when_no_heartbeat_after_grace(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from scripts.healthcheck_training import check_heartbeat_health

        monkeypatch.setenv("LOGS_DIR", str(tmp_path))
        monkeypatch.setenv("HEALTH_GRACE_PERIOD_MINUTES", "0")

        healthy, msg = check_heartbeat_health()
        assert healthy is False
        assert "not found" in msg.lower()


class TestHealthcheckGracePeriod:
    def test_healthy_during_grace_period(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from scripts.healthcheck_training import check_heartbeat_health

        monkeypatch.setenv("LOGS_DIR", str(tmp_path))
        monkeypatch.setenv("HEALTH_GRACE_PERIOD_MINUTES", "999")

        # No heartbeat.json, but within grace period
        healthy, msg = check_heartbeat_health()
        assert healthy is True
        assert "grace" in msg.lower()
