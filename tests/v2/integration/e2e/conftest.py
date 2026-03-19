"""Conftest fixtures for e2e testing with pytest-docker.

Provides fixtures for:
1. docker_compose_file: Returns compose file paths
2. docker_compose_project_name: Returns "e2e_test" project name
3. wait_for_services: Session-scoped health check waiter
4. mlflow_client: MlflowClient connected to test stack
5. run_flow_in_docker: Runs a named flow via docker compose run

All fixtures use .env.example values for service URLs/ports.
No MINIVESS_ALLOW_HOST=1 — Docker-only execution.
"""

from __future__ import annotations

import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

REPO_ROOT = Path(__file__).resolve().parents[5]
COMPOSE_INFRA = REPO_ROOT / "deployment" / "docker-compose.yml"
COMPOSE_FLOWS = REPO_ROOT / "deployment" / "docker-compose.flows.yml"
ENV_FILE = REPO_ROOT / ".env"


@pytest.fixture(scope="session")
def docker_compose_file() -> list[str]:
    """Returns list of compose file paths for pytest-docker."""
    return [str(COMPOSE_INFRA), str(COMPOSE_FLOWS)]


@pytest.fixture(scope="session")
def docker_compose_project_name() -> str:
    """Returns project name to avoid conflicts with dev stack."""
    return "e2e_test"


def _check_http(url: str, timeout: int = 5) -> bool:
    """Check if an HTTP endpoint responds."""
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except Exception:
        return False


def _wait_for_service(
    name: str, url: str, max_wait: int = 120, interval: int = 5
) -> None:
    """Wait for a service to become healthy with exponential backoff."""
    start = time.monotonic()
    wait = interval
    while time.monotonic() - start < max_wait:
        if _check_http(url):
            return
        time.sleep(wait)
        wait = min(wait * 1.5, 30)
    raise TimeoutError(f"Service {name} at {url} not healthy after {max_wait}s")


def _docker_available() -> bool:
    """Check if Docker daemon is running."""
    try:
        result = subprocess.run(  # noqa: S603, S607
            ["docker", "info"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _any_service_reachable() -> bool:
    """Quick check if at least one e2e service is listening."""
    try:
        with socket.create_connection(("localhost", 5000), timeout=2):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def wait_for_services() -> None:
    """Session-scoped fixture that waits for all infra services to be healthy.

    Skips instead of erroring when Docker or services are not available.
    """
    if not _docker_available():
        pytest.skip("Docker daemon not available — e2e tests require full Docker stack")
    if not _any_service_reachable():
        pytest.skip(
            "E2E infrastructure services not running — "
            "start with docker compose up before running e2e tests"
        )
    services = {
        "MLflow": ("http://localhost:5000/health", 5000),
        "Prefect": ("http://localhost:4200/api/health", 4200),
        "MinIO": ("http://localhost:9000/minio/health/live", 9000),
        "Grafana": ("http://localhost:3000/api/health", 3000),
    }
    # Quick TCP check first to avoid long waits for services that aren't running
    for name, (_url, port) in services.items():
        try:
            with socket.create_connection(("localhost", port), timeout=2):
                pass
        except OSError:
            pytest.skip(f"E2E service {name} not listening on port {port}")
    # All services are TCP-reachable; now wait for HTTP health
    for name, (url, _port) in services.items():
        try:
            _wait_for_service(name, url, max_wait=60)
        except TimeoutError:
            pytest.skip(f"E2E service {name} at {url} not healthy after 60s")


@pytest.fixture(scope="session")
def mlflow_client(wait_for_services: None) -> object:
    """Returns MlflowClient connected to test stack."""
    from mlflow.tracking import MlflowClient

    return MlflowClient("http://localhost:5000")


@pytest.fixture(scope="session")
def run_flow_in_docker():
    """Fixture function that runs a named flow via docker compose run."""

    def _run(
        flow_name: str,
        experiment: str = "debug_all_models",
        extra_env: dict[str, str] | None = None,
        timeout: int = 3600,
    ) -> subprocess.CompletedProcess:
        """Run a flow in Docker via docker compose run.

        Args:
            flow_name: Name of the service in docker-compose.flows.yml
            experiment: Experiment name to use
            extra_env: Additional environment variables
            timeout: Max seconds to wait for completion

        Returns:
            CompletedProcess with stdout/stderr
        """
        cmd = [
            "docker",
            "compose",
            "-f",
            str(COMPOSE_FLOWS),
            "-p",
            "e2e_test",
        ]

        if ENV_FILE.exists():
            cmd.extend(["--env-file", str(ENV_FILE)])

        cmd.extend(
            [
                "run",
                "--rm",
                "-T",
                "-e",
                f"EXPERIMENT={experiment}",
                "-e",
                "MINIVESS_DEBUG_SUFFIX=_E2E_TEST",
            ]
        )

        if extra_env:
            for key, value in extra_env.items():
                cmd.extend(["-e", f"{key}={value}"])

        # GPU flows need --shm-size
        gpu_flows = {"train", "post_training", "analyze"}
        if flow_name in gpu_flows:
            cmd.extend(["--shm-size", "8g"])

        cmd.append(flow_name)

        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(REPO_ROOT),
        )

    return _run


@pytest.fixture(scope="session", autouse=False)
def cleanup_e2e_data(wait_for_services: None) -> Generator[None, None, None]:
    """Session-scoped fixture that cleans up test experiments after the session."""
    yield

    # Cleanup: delete _E2E_TEST experiments from MLflow
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient("http://localhost:5000")
        for exp in client.search_experiments():
            if "_E2E_TEST" in exp.name:
                client.delete_experiment(exp.experiment_id)
    except Exception:
        pass  # Best-effort cleanup
