"""Integration tests for inter-container network connectivity.

These tests verify that flow containers can reach infrastructure services
via the minivess-network. They require:
  - minivess-network to exist: docker network create minivess-network
  - Infra stack running: docker compose -f deployment/docker-compose.yml --profile dev up -d

All tests are skipped if Docker is not available on the test host.
Mark: @pytest.mark.integration — excluded from staging tier (pytest-staging.ini).

Rule #16: No regex. subprocess output parsed with str.split() / str.partition().
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

# ── Docker availability guard (module-level, not just mark) ──────────────────
# If docker is not installed or daemon not running, skip all tests in this file.
# This is a hard skip — not a failure — so CI runners without Docker still pass.

_DOCKER_AVAILABLE = False
try:
    result = subprocess.run(
        ["docker", "info"],
        capture_output=True,
        timeout=5,
        check=False,
    )
    _DOCKER_AVAILABLE = result.returncode == 0
except (FileNotFoundError, subprocess.TimeoutExpired):
    _DOCKER_AVAILABLE = False

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _DOCKER_AVAILABLE, reason="Docker not available on this host"
    ),
]

REPO_ROOT = Path(__file__).parent.parent.parent.parent
COMPOSE_FLOWS = REPO_ROOT / "deployment" / "docker-compose.flows.yml"
ENV_FILE = REPO_ROOT / ".env"


def _compose_run_curl(url: str, service: str = "data") -> tuple[int, str]:
    """Run curl inside a flow container and return (returncode, stdout)."""
    env_args = ["--env-file", str(ENV_FILE)] if ENV_FILE.exists() else []
    cmd = [
        "docker",
        "compose",
        *env_args,
        "-f",
        str(COMPOSE_FLOWS),
        "run",
        "--rm",
        "--no-deps",
        service,
        "curl",
        "-sf",
        "--max-time",
        "5",
        url,
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=30, check=False
    )
    return result.returncode, result.stdout


def test_flow_can_reach_mlflow() -> None:
    """Flow container must reach MLflow health endpoint via minivess-network."""
    returncode, _ = _compose_run_curl("http://minivess-mlflow:5000/health")
    assert returncode == 0, (
        "Flow container cannot reach http://minivess-mlflow:5000/health. "
        "Verify infra stack is running: "
        "docker compose -f deployment/docker-compose.yml --profile dev up -d"
    )


def test_flow_can_reach_minio() -> None:
    """Flow container must reach MinIO health endpoint via minivess-network."""
    returncode, _ = _compose_run_curl("http://minio:9000/minio/health/live")
    assert returncode == 0, (
        "Flow container cannot reach http://minio:9000/minio/health/live. "
        "Verify minio service is running and on minivess-network."
    )


def test_flow_can_reach_prefect() -> None:
    """Flow container must reach Prefect API health endpoint via minivess-network."""
    returncode, _ = _compose_run_curl("http://minivess-prefect:4200/api/health")
    assert returncode == 0, (
        "Flow container cannot reach http://minivess-prefect:4200/api/health. "
        "Verify Prefect server is running and on minivess-network."
    )
