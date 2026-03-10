"""Tests for health adapter — QA monitoring merged into dashboard (T-DASH.3.1).

Validates the aggregated health check across all service adapters.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from minivess.dashboard.adapters.base import ServiceAdapter
from minivess.dashboard.adapters.health_adapter import HealthAdapter, HealthSummary


class _MockAdapter(ServiceAdapter):
    """Adapter stub for health testing."""

    def __init__(self, name: str, healthy: bool = True) -> None:
        super().__init__(name, "http://fake:9999")
        self._healthy = healthy

    def _fetch(self, endpoint: str) -> dict[str, Any]:
        if not self._healthy:
            msg = "down"
            raise ConnectionError(msg)
        return {"status": "ok"}


class TestHealthAdapter:
    """Test the aggregated health adapter."""

    def test_health_adapter_aggregates_all_adapters(self) -> None:
        adapters = [
            _MockAdapter("MLflow", healthy=True),
            _MockAdapter("Prefect", healthy=True),
        ]
        # Trigger status by querying
        for a in adapters:
            a.query("test")

        health = HealthAdapter(adapters)
        summary = health.check_all()
        assert len(summary.adapter_statuses) == 2

    def test_health_all_healthy(self) -> None:
        adapters = [
            _MockAdapter("MLflow", healthy=True),
            _MockAdapter("Prefect", healthy=True),
        ]
        for a in adapters:
            a.query("test")

        health = HealthAdapter(adapters)
        summary = health.check_all()
        assert summary.overall == "healthy"
        assert len(summary.alerts) == 0

    def test_health_degraded_when_partial_failure(self) -> None:
        ok = _MockAdapter("MLflow", healthy=True)
        ok.query("test")
        bad = _MockAdapter("Prefect", healthy=False)
        bad.query("test")

        health = HealthAdapter([ok, bad])
        summary = health.check_all()
        assert summary.overall == "degraded"
        assert len(summary.alerts) == 1

    def test_health_unhealthy_when_all_down(self) -> None:
        bad1 = _MockAdapter("MLflow", healthy=False)
        bad1.query("test")
        bad2 = _MockAdapter("Prefect", healthy=False)
        bad2.query("test")

        health = HealthAdapter([bad1, bad2])
        summary = health.check_all()
        assert summary.overall == "unhealthy"

    def test_health_summary_schema(self) -> None:
        health = HealthAdapter([_MockAdapter("X", healthy=True)])
        summary = health.check_all()
        d = summary.to_dict()
        assert "overall" in d
        assert "alerts" in d
        assert "last_checked" in d
        assert isinstance(d["alerts"], list)

    def test_health_refresh_interval_from_env(self) -> None:
        with patch.dict("os.environ", {"DASHBOARD_HEALTH_CHECK_INTERVAL_S": "60"}):
            health = HealthAdapter([])
            assert health.check_interval_s == 60.0

    def test_health_last_summary_cached(self) -> None:
        health = HealthAdapter([_MockAdapter("X", healthy=True)])
        assert health.last_summary is None
        health.check_all()
        assert health.last_summary is not None
        assert isinstance(health.last_summary, HealthSummary)
