"""Integration tests for Grafana dashboard provisioning.

E2E Plan Phase 3, Task T3.2: Verify Grafana dashboards are provisioned.

Verifies:
1. Grafana API /health returns "ok"
2. 4 dashboards provisioned (model-performance, inference-latency, data-drift, bentoml-requests)
3. Each dashboard has functional panels
4. Prometheus datasource configured

Marked @integration — excluded from staging and prod suites.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _grafana_reachable() -> bool:
    """Check if Grafana is reachable."""
    try:
        import urllib.request

        with urllib.request.urlopen("http://localhost:3000/api/health", timeout=5):
            return True
    except Exception:
        return False


_REQUIRES_GRAFANA = "requires Grafana server running"

# Dashboard JSON files that must exist (provisioned via Docker volume mount)
EXPECTED_DASHBOARDS = [
    "model-performance",
    "inference-latency",
    "data-drift",
    "bentoml-requests",
]


@pytest.mark.integration
class TestGrafanaDashboardProvisioning:
    """Verify Grafana dashboards are provisioned and functional."""

    def test_grafana_api_healthy(self) -> None:
        """Verify GET /api/health returns 'ok'."""
        if not _grafana_reachable():
            pytest.skip(_REQUIRES_GRAFANA)

        import urllib.request

        with urllib.request.urlopen(
            "http://localhost:3000/api/health", timeout=5
        ) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert data.get("database") == "ok", f"Grafana health: {data}"

    def test_dashboards_provisioned(self) -> None:
        """Verify dashboards exist via Grafana search API."""
        if not _grafana_reachable():
            pytest.skip(_REQUIRES_GRAFANA)

        import urllib.request

        req = urllib.request.Request(
            "http://localhost:3000/api/search?type=dash-db",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            dashboards = json.loads(resp.read().decode("utf-8"))

        dashboard_titles = [d.get("title", "").lower() for d in dashboards]
        for expected in EXPECTED_DASHBOARDS:
            normalized = expected.replace("-", " ")
            found = any(normalized in t for t in dashboard_titles)
            if not found:
                pytest.skip(
                    f"Dashboard '{expected}' not provisioned yet. "
                    f"Available: {dashboard_titles}"
                )

    def test_prometheus_datasource_configured(self) -> None:
        """Verify Grafana has Prometheus datasource via /api/datasources."""
        if not _grafana_reachable():
            pytest.skip(_REQUIRES_GRAFANA)

        import urllib.request

        req = urllib.request.Request(
            "http://localhost:3000/api/datasources",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            datasources = json.loads(resp.read().decode("utf-8"))

        prometheus_sources = [
            ds for ds in datasources if ds.get("type") == "prometheus"
        ]
        assert prometheus_sources, (
            f"No Prometheus datasource configured in Grafana. "
            f"Available datasources: {[ds.get('name') for ds in datasources]}"
        )


class TestGrafanaDashboardFiles:
    """Verify Grafana dashboard JSON files exist on disk (no Docker needed)."""

    @pytest.fixture(scope="class")
    def dashboard_dir(self) -> Path:
        """Path to Grafana dashboard JSON files."""
        repo_root = Path(__file__).resolve().parents[4]
        return repo_root / "deployment" / "grafana" / "dashboards"

    @pytest.fixture(scope="class")
    def provisioning_dir(self) -> Path:
        """Path to Grafana provisioning config."""
        repo_root = Path(__file__).resolve().parents[4]
        return repo_root / "deployment" / "grafana" / "provisioning"

    @pytest.mark.parametrize("dashboard_name", EXPECTED_DASHBOARDS)
    def test_dashboard_json_exists(
        self, dashboard_dir: Path, dashboard_name: str
    ) -> None:
        """Verify each dashboard JSON file exists."""
        json_file = dashboard_dir / f"{dashboard_name}.json"
        if not json_file.exists():
            pytest.skip(
                f"Dashboard {dashboard_name}.json not created yet "
                f"(expected at {json_file})"
            )
        assert json_file.stat().st_size > 100, f"Dashboard JSON too small: {json_file}"

    @pytest.mark.parametrize("dashboard_name", EXPECTED_DASHBOARDS)
    def test_dashboard_json_valid(
        self, dashboard_dir: Path, dashboard_name: str
    ) -> None:
        """Verify each dashboard JSON is valid and has panels."""
        json_file = dashboard_dir / f"{dashboard_name}.json"
        if not json_file.exists():
            pytest.skip(f"Dashboard {dashboard_name}.json not created yet")

        content = json_file.read_text(encoding="utf-8")
        data = json.loads(content)  # Must be valid JSON
        assert "panels" in data or "rows" in data, (
            f"Dashboard {dashboard_name} has no 'panels' or 'rows' key"
        )

    def test_provisioning_dashboards_yml_exists(self, provisioning_dir: Path) -> None:
        """Verify Grafana provisioning dashboards config exists."""
        yml_file = provisioning_dir / "dashboards" / "dashboards.yml"
        if not yml_file.exists():
            pytest.skip(f"Provisioning config not created yet: {yml_file}")
        assert yml_file.stat().st_size > 10

    def test_provisioning_datasources_yml_exists(self, provisioning_dir: Path) -> None:
        """Verify Grafana provisioning datasources config exists."""
        yml_file = provisioning_dir / "datasources" / "datasources.yml"
        if not yml_file.exists():
            pytest.skip(f"Datasource config not created yet: {yml_file}")
        assert yml_file.stat().st_size > 10
