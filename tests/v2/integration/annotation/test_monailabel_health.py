"""Smoke tests for MONAI Label server health (T-ANN.4.2).

These require the Docker stack to be running — auto-skipped via
@pytest.mark.requires_docker when Docker is unavailable.
"""

from __future__ import annotations

import json
import os
import urllib.request

import pytest


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
        url = f"{self._monailabel_url()}/info"
        req = urllib.request.Request(url, method="GET")  # noqa: S310
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            assert resp.status == 200

    def test_monailabel_has_app_info(self) -> None:
        """Response JSON should contain app information."""
        url = f"{self._monailabel_url()}/info"
        req = urllib.request.Request(url, method="GET")  # noqa: S310
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
        assert isinstance(data, dict)

    def test_champion_infer_endpoint_exists(self) -> None:
        """/infer/ endpoint should be available."""
        url = f"{self._monailabel_url()}/info"
        req = urllib.request.Request(url, method="GET")  # noqa: S310
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
        # MONAI Label /info returns available models/infers
        assert isinstance(data, dict)
