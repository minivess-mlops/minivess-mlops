"""Tests for dashboard env vars and compose service configuration (T-DASH.0.2)."""

from __future__ import annotations

from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[4]


class TestDashboardEnvVars:
    """Validate dashboard ports in .env.example."""

    def test_dashboard_ui_port_in_env_example(self) -> None:
        env = (_REPO_ROOT / ".env.example").read_text(encoding="utf-8")
        assert "DASHBOARD_UI_PORT=" in env

    def test_dashboard_api_port_in_env_example(self) -> None:
        env = (_REPO_ROOT / ".env.example").read_text(encoding="utf-8")
        assert "DASHBOARD_API_PORT=" in env

    def test_dashboard_ports_not_conflict_with_grafana(self) -> None:
        env = (_REPO_ROOT / ".env.example").read_text(encoding="utf-8")
        # Extract port values
        ports: dict[str, str] = {}
        for line in env.splitlines():
            if "=" in line and not line.startswith("#"):
                key, _, val = line.partition("=")
                ports[key.strip()] = val.strip()
        assert ports.get("DASHBOARD_UI_PORT") != ports.get("GRAFANA_PORT")

    def test_dashboard_api_service_in_compose(self) -> None:
        compose_path = _REPO_ROOT / "deployment" / "docker-compose.flows.yml"
        with compose_path.open(encoding="utf-8") as f:
            compose = yaml.safe_load(f)
        services = compose.get("services", {})
        assert "dashboard-api" in services

    def test_dashboard_ui_service_in_compose(self) -> None:
        compose_path = _REPO_ROOT / "deployment" / "docker-compose.flows.yml"
        with compose_path.open(encoding="utf-8") as f:
            compose = yaml.safe_load(f)
        services = compose.get("services", {})
        assert "dashboard-ui" in services
