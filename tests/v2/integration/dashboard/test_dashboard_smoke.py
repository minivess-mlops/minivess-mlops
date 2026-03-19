"""Integration smoke tests for the dashboard Docker services (T-DASH.4.2).

Requires Docker stack running — auto-skipped via @pytest.mark.requires_docker
and inline health checks when dashboard services are not reachable.
"""

from __future__ import annotations

import os
import socket
import urllib.request

import pytest


def _service_reachable(host: str, port: int) -> bool:
    """Check if a TCP service is reachable."""
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


@pytest.mark.requires_docker
@pytest.mark.slow
@pytest.mark.integration
class TestDashboardSmoke:
    """Smoke tests for the Dashboard Docker services."""

    def _api_url(self) -> str:
        port = os.environ.get("DASHBOARD_API_PORT", "8090")
        return f"http://localhost:{port}"

    def _api_port(self) -> int:
        return int(os.environ.get("DASHBOARD_API_PORT", "8090"))

    def _ui_url(self) -> str:
        port = os.environ.get("DASHBOARD_UI_PORT", "3002")
        return f"http://localhost:{port}"

    def _ui_port(self) -> int:
        return int(os.environ.get("DASHBOARD_UI_PORT", "3002"))

    def test_dashboard_api_health_endpoint(self) -> None:
        if not _service_reachable("localhost", self._api_port()):
            pytest.skip("Dashboard API not running")
        url = f"{self._api_url()}/health"
        req = urllib.request.Request(url, method="GET")  # noqa: S310
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            assert resp.status == 200

    def test_dashboard_api_mlflow_endpoint(self) -> None:
        if not _service_reachable("localhost", self._api_port()):
            pytest.skip("Dashboard API not running")
        url = f"{self._api_url()}/api/mlflow/experiments"
        req = urllib.request.Request(url, method="GET")  # noqa: S310
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            assert resp.status == 200

    def test_dashboard_ui_serves_html(self) -> None:
        if not _service_reachable("localhost", self._ui_port()):
            pytest.skip("Dashboard UI not running")
        url = self._ui_url()
        req = urllib.request.Request(url, method="GET")  # noqa: S310
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            assert resp.status == 200
            content = resp.read().decode("utf-8")
            assert "<html" in content.lower()
