from __future__ import annotations

import os
import tempfile
import warnings

import pytest
from prefect.testing.utilities import prefect_test_harness

# Exclude GPU instance tests from default collection.
# tests/gpu_instance/ contains SAM3 and other GPU-heavy tests that are
# NEVER part of the standard suite. Run them explicitly:
#     uv run pytest tests/gpu_instance/
#     make test-gpu
collect_ignore_glob = ["gpu_instance/*"]


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


@pytest.fixture(scope="session", autouse=True)
def _cleanup_spurious_file_dirs():
    """Remove spurious ``file:`` directories created by MLflow relative URI bugs.

    Some test interaction paths cause MLflow to interpret ``file:something`` as
    a relative path, creating a ``file:`` directory in the repo root. Cleanup
    runs BEFORE tests (from previous sessions) and AFTER (from current session).
    """
    import shutil
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    spurious = repo_root / "file:"
    if spurious.exists():
        shutil.rmtree(spurious, ignore_errors=True)

    yield

    if spurious.exists():
        shutil.rmtree(spurious, ignore_errors=True)


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
    config.addinivalue_line(
        "markers",
        "model_construction: Tests model adapter construction, LoRA application,"
        " config validation. Fast (CPU-only, <5s). Run via: make test-models",
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


# ── Zero-Skip Enforcement (Rule 28) ──────────────────────────────────────────
# Skips are bugs hiding as skips. This hook makes any skip a hard failure in
# staging and prod tiers. Every skip must be investigated and resolved — either
# by fixing the root cause, moving the test to the correct tier, or deleting it.
#
# The ONLY allowed skips are xfail (expected failures with documented reasons).
# pytest.skip(), importorskip(), and mark.skipif() in staging/prod = ERROR.
#
# Opt-out: set ALLOW_SKIPS=1 for debugging (NEVER in CI/Makefile).


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter,
    exitstatus: int,
    config: pytest.Config,
) -> None:
    """Fail the test run if ANY tests were skipped (Rule 28: Zero Silent Skips).

    This converts skips from invisible noise to hard errors. If a test cannot
    run in this tier, it belongs in a different tier — not skipped silently.
    """
    if os.environ.get("ALLOW_SKIPS") == "1":
        return

    skipped = terminalreporter.stats.get("skipped", [])
    if not skipped:
        return

    terminalreporter.section("ZERO-SKIP ENFORCEMENT (Rule 28)")
    terminalreporter.write_line(
        f"FATAL: {len(skipped)} test(s) were SKIPPED. Skips are not allowed."
    )
    terminalreporter.write_line("")
    for report in skipped:
        # report.longrepr is a tuple: (file, line, reason)
        if hasattr(report, "longrepr") and isinstance(report.longrepr, tuple):
            _file, _line, reason = report.longrepr
            terminalreporter.write_line(f"  SKIP: {report.nodeid}")
            terminalreporter.write_line(f"        Reason: {reason}")
        else:
            terminalreporter.write_line(f"  SKIP: {report.nodeid}")
    terminalreporter.write_line("")
    terminalreporter.write_line("Fix each skip by:")
    terminalreporter.write_line("  1. Install the missing package (uv sync --all-extras)")
    terminalreporter.write_line("  2. Move the test to the correct tier (cloud/gpu_instance/integration)")
    terminalreporter.write_line("  3. Delete the test if the feature is deprecated")
    terminalreporter.write_line("  4. Clear __pycache__ if stale bytecode (find . -name __pycache__ -exec rm -rf {} +)")
    terminalreporter.write_line("")

    # Override exit status to failure
    terminalreporter._session.exitstatus = pytest.ExitCode.TESTS_FAILED


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
