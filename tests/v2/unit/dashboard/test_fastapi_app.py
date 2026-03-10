"""Tests for FastAPI dashboard app (T-DASH.4.2 unit portion).

Validates the FastAPI app starts and serves expected endpoints.
Uses TestClient as context manager to trigger lifespan (adapter init).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from minivess.dashboard.app.main import create_app


class TestFastAPIApp:
    """Test the FastAPI dashboard application."""

    def test_app_starts(self) -> None:
        app = create_app()
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "ok"

    def test_health_endpoint_returns_schema(self) -> None:
        app = create_app()
        with TestClient(app) as client:
            response = client.get("/api/health/")
            assert response.status_code == 200
            data = response.json()
            assert "overall" in data
            assert "alerts" in data

    def test_mlflow_experiments_endpoint_exists(self) -> None:
        """MLflow endpoint exists (returns empty when no server)."""
        app = create_app()
        with TestClient(app) as client:
            response = client.get("/api/mlflow/experiments")
            assert response.status_code == 200

    def test_prefect_flows_endpoint_exists(self) -> None:
        app = create_app()
        with TestClient(app) as client:
            response = client.get("/api/prefect/flows")
            assert response.status_code == 200

    def test_serving_health_endpoint_exists(self) -> None:
        app = create_app()
        with TestClient(app) as client:
            response = client.get("/api/serving/health")
            assert response.status_code == 200

    def test_all_routers_importable(self) -> None:
        from minivess.dashboard.app.routers import health, mlflow, prefect, serving

        assert health.router is not None
        assert mlflow.router is not None
        assert prefect.router is not None
        assert serving.router is not None
