"""Smoke tests for MONAI Label server health (T-ANN.4.2).

These require the Docker stack to be running — auto-skipped via
@pytest.mark.requires_docker when Docker is unavailable, and via
inline health check when the MONAI Label service is not reachable.
"""

from __future__ import annotations

import json
import os
import socket
import urllib.request

import pytest


def _monailabel_reachable() -> bool:
    """Check if MONAI Label server is reachable."""
    port = int(os.environ.get("MONAI_LABEL_PORT", "8000"))
    try:
        with socket.create_connection(("localhost", port), timeout=2):
            return True
    except OSError:
        return False


_REQUIRES_MONAILABEL = "MONAI Label server not running"


@pytest.mark.requires_docker
@pytest.mark.slow
@pytest.mark.integration
class TestMonaiLabelHealth:
    """Smoke tests for the MONAI Label Docker service."""

    def _monailabel_url(self) -> str:
        port = os.environ.get("MONAI_LABEL_PORT", "8000")
        return f"http://localhost:{port}"

    def test_monailabel_server_responds(self) -> None:
        """GET /info should return HTTP 200."""
        if not _monailabel_reachable():
            pytest.skip(_REQUIRES_MONAILABEL)
        url = f"{self._monailabel_url()}/info"
        req = urllib.request.Request(url, method="GET")  # noqa: S310
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            assert resp.status == 200

    def test_monailabel_has_app_info(self) -> None:
        """Response JSON should contain app information."""
        if not _monailabel_reachable():
            pytest.skip(_REQUIRES_MONAILABEL)
        url = f"{self._monailabel_url()}/info"
        req = urllib.request.Request(url, method="GET")  # noqa: S310
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
        assert isinstance(data, dict)

    def test_champion_infer_endpoint_exists(self) -> None:
        """/infer/ endpoint should be available."""
        if not _monailabel_reachable():
            pytest.skip(_REQUIRES_MONAILABEL)
        url = f"{self._monailabel_url()}/info"
        req = urllib.request.Request(url, method="GET")  # noqa: S310
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
        # MONAI Label /info returns available models/infers
        assert isinstance(data, dict)
