"""Parametrized MLflow backend fixture for L1 generic tests (#625).

Provides ``mlflow_backend`` fixture that yields a tracking URI for
either filesystem or subprocess MLflow server backends. Tests using
this fixture automatically run against both backends.

NOTE: SQLite is BANNED per project rules (PostgreSQL is ONLY database).
"""

from __future__ import annotations

import socket
import subprocess
import time
from collections.abc import Generator
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(url: str, *, timeout: float = 10.0) -> None:
    """Wait until an HTTP endpoint returns a 200 status code."""
    import urllib.request

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:  # noqa: S310
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(0.3)
    msg = f"Server at {url} did not start within {timeout}s"
    raise TimeoutError(msg)


@pytest.fixture(
    params=[
        "filesystem",
        pytest.param("server", marks=pytest.mark.slow),
    ]
)
def mlflow_backend(request: pytest.FixtureRequest, tmp_path: Path) -> Generator[str]:
    """Parametrize tests across MLflow backend types.

    Yields a tracking URI string suitable for ``mlflow.set_tracking_uri()``.

    Parameters
    ----------
    filesystem:
        Uses a tmp_path directory as the backend store.
    server:
        Starts a subprocess MLflow server with filesystem backend.
        Marked ``@pytest.mark.slow`` — excluded from staging tier.
    """
    if request.param == "filesystem":
        yield str(tmp_path / "mlruns")
    elif request.param == "server":
        store = str(tmp_path / "mlruns")
        port = _find_free_port()
        proc = subprocess.Popen(
            [
                "mlflow",
                "server",
                "--backend-store-uri",
                store,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            _wait_for_server(f"http://127.0.0.1:{port}/health", timeout=10)
            yield f"http://127.0.0.1:{port}"
        finally:
            proc.terminate()
            proc.wait(timeout=5)
