from __future__ import annotations

import os
import tempfile
import warnings

import pytest
from prefect.testing.utilities import prefect_test_harness


@pytest.fixture(scope="session", autouse=True)
def _prefect_test_server():
    """Start an ephemeral Prefect server for the entire test session.

    Prefect 3.x flows/tasks require a running API server. The test harness
    spins up a temporary SQLite-backed server (~6s startup, shared across
    all tests). This replaces the old PREFECT_DISABLED=1 no-op approach.

    PREFECT_LOGGING_TO_API_WHEN_MISSING_FLOW=ignore allows @task functions
    to be called directly in tests (outside a @flow context) without
    raising MissingContextError.
    """
    os.environ["PREFECT_LOGGING_TO_API_WHEN_MISSING_FLOW"] = "ignore"
    with prefect_test_harness():
        yield
    os.environ.pop("PREFECT_LOGGING_TO_API_WHEN_MISSING_FLOW", None)


@pytest.fixture(scope="session", autouse=True)
def _allow_host_env():
    """Bypass Docker context gate for all tests.

    Sets MINIVESS_ALLOW_HOST=1 so that _require_docker_context() does not
    reject test runs executed outside Docker containers.
    """
    os.environ["MINIVESS_ALLOW_HOST"] = "1"
    yield
    os.environ.pop("MINIVESS_ALLOW_HOST", None)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom CLI options for GPU test tiers."""
    parser.addoption(
        "--run-gpu-heavy",
        action="store_true",
        default=False,
        help="Run GPU-heavy tests (SAM3 forward passes, large model training). "
        "These are skipped by default — they are for GPU deployment validation.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Configure markers and isolate Prefect home per xdist worker.

    When running with pytest-xdist (-n N), each worker gets its own
    PREFECT_HOME directory to prevent SQLite locking between concurrent
    Prefect test servers.
    """
    # Isolate Prefect home per xdist worker to avoid SQLite locking.
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id is not None:
        tmp_dir = tempfile.mkdtemp(prefix=f"prefect_home_{worker_id}_")
        os.environ["PREFECT_HOME"] = tmp_dir

    config.addinivalue_line(
        "markers", "real_data: requires real MiniVess dataset (not run in CI)"
    )
    config.addinivalue_line(
        "markers",
        "requires_mlflow_server: Tests requiring a running MLflow server"
        " (auto-skipped if unhealthy)",
    )


def _docker_daemon_available() -> bool:
    """Return True if a Docker daemon socket is reachable on this host."""
    import socket
    from pathlib import Path

    # Linux/macOS: check for the Unix socket
    if Path("/var/run/docker.sock").exists():
        return True
    # Fallback: try TCP (Docker Desktop on Windows or remote daemon)
    try:
        with socket.create_connection(("localhost", 2375), timeout=1):
            return True
    except OSError:
        return False


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Auto-tag and auto-skip tests based on location and markers."""
    _mlflow_healthy: bool | None = None
    _docker_available: bool | None = None
    _run_gpu_heavy = config.getoption("--run-gpu-heavy", default=False)

    for item in items:
        # Auto-tag all tests in tests/v2/integration/ or tests/integration/ with
        # @pytest.mark.integration so the staging tier can exclude them via -m filter.
        item_path = str(item.fspath)
        if "/integration/" in item_path:
            item.add_marker(pytest.mark.integration)

        # Auto-skip gpu_heavy tests unless --run-gpu-heavy is passed.
        # GPU-heavy tests (SAM3 forward passes, large model training) are NOT
        # part of the standard dev test suite. They are for GPU deployment
        # validation on machines with adequate VRAM.
        if not _run_gpu_heavy and item.get_closest_marker("gpu_heavy") is not None:
            item.add_marker(
                pytest.mark.skip(reason="GPU-heavy test — pass --run-gpu-heavy to run")
            )

        if item.get_closest_marker("requires_docker") is not None:
            nonlocal_docker = _docker_available
            if nonlocal_docker is None:
                nonlocal_docker = _docker_daemon_available()
                _docker_available = nonlocal_docker
            if not nonlocal_docker:
                item.add_marker(pytest.mark.skip(reason="Docker daemon not reachable"))

        if item.get_closest_marker("requires_mlflow_server") is not None:
            # Lazy-evaluate health once per session
            nonlocal_healthy = _mlflow_healthy
            if nonlocal_healthy is None:
                try:
                    from minivess.observability.health import check_mlflow_health

                    result = check_mlflow_health()
                    nonlocal_healthy = result.healthy
                except Exception:
                    nonlocal_healthy = False
                _mlflow_healthy = nonlocal_healthy

            if not _mlflow_healthy:
                item.add_marker(pytest.mark.skip(reason="MLflow server not reachable"))


# Suppress warnings that occur during import of third-party libraries.
# These must be set before the libraries are imported, so pytest's
# filterwarnings config (which applies after collection) is too late.
warnings.filterwarnings(
    "ignore",
    message=".*deprecated.*",
    category=DeprecationWarning,
    module="pyparsing.*",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    module="MetricsReloaded.*",
)
