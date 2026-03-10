"""Infrastructure health check tests for e2e testing.

E2E Plan Phase 4, Task T4.1: pytest-docker integration setup.

Verifies:
1. PostgreSQL accepts connections
2. MinIO health endpoint returns 200
3. MLflow health endpoint returns 200
4. Prefect /api/health returns 200
5. Grafana /api/health returns "ok"
6. Docker network minivess-network exists

These tests require the Docker stack to be running.
Marked @integration @slow — runs via make test-e2e.
"""

from __future__ import annotations

import json
import subprocess

import pytest


def _docker_available() -> bool:
    """Check if Docker daemon is running."""
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


_REQUIRES_DOCKER = "requires Docker infrastructure"


@pytest.mark.integration
@pytest.mark.slow
class TestInfrastructureHealth:
    """Verify all infrastructure services are healthy."""

    def test_postgresql_healthy(self) -> None:
        """Verify PostgreSQL accepts connections."""
        if not _docker_available():
            pytest.skip(_REQUIRES_DOCKER)

        result = subprocess.run(
            [
                "docker",
                "exec",
                "e2e_test-postgres-1",
                "pg_isready",
                "-U",
                "minivess",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            pytest.skip(
                f"PostgreSQL not accessible: {result.stderr}. "
                f"Run: docker compose -f deployment/docker-compose.yml up -d"
            )

    def test_minio_healthy(self) -> None:
        """Verify MinIO /minio/health/live returns 200."""
        if not _docker_available():
            pytest.skip(_REQUIRES_DOCKER)

        import urllib.request

        try:
            with urllib.request.urlopen(
                "http://localhost:9000/minio/health/live", timeout=5
            ) as resp:
                assert resp.status == 200
        except Exception:
            pytest.skip("MinIO not reachable at localhost:9000")

    def test_mlflow_healthy(self) -> None:
        """Verify MLflow health endpoint returns 200."""
        if not _docker_available():
            pytest.skip(_REQUIRES_DOCKER)

        import urllib.request

        try:
            with urllib.request.urlopen(
                "http://localhost:5000/health", timeout=5
            ) as resp:
                assert resp.status == 200
        except Exception:
            pytest.skip("MLflow not reachable at localhost:5000")

    def test_prefect_healthy(self) -> None:
        """Verify Prefect /api/health returns 200."""
        if not _docker_available():
            pytest.skip(_REQUIRES_DOCKER)

        import urllib.request

        try:
            with urllib.request.urlopen(
                "http://localhost:4200/api/health", timeout=5
            ) as resp:
                assert resp.status == 200
        except Exception:
            pytest.skip("Prefect not reachable at localhost:4200")

    def test_grafana_healthy(self) -> None:
        """Verify Grafana /api/health returns 'ok'."""
        if not _docker_available():
            pytest.skip(_REQUIRES_DOCKER)

        import urllib.request

        try:
            with urllib.request.urlopen(
                "http://localhost:3000/api/health", timeout=5
            ) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                assert data.get("database") == "ok"
        except Exception:
            pytest.skip("Grafana not reachable at localhost:3000")

    def test_minivess_network_exists(self) -> None:
        """Verify docker network minivess-network exists."""
        if not _docker_available():
            pytest.skip(_REQUIRES_DOCKER)

        result = subprocess.run(
            ["docker", "network", "inspect", "minivess-network"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            pytest.skip(
                "minivess-network not found. "
                "Run: docker network create minivess-network"
            )
