"""BentoML serving adapter for the dashboard."""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any

from minivess.dashboard.adapters.base import ServiceAdapter

logger = logging.getLogger(__name__)


class BentoMLAdapter(ServiceAdapter):
    """Adapter for BentoML serving endpoint.

    Reads BENTO_SERVER_URL from environment (Rule #22).
    """

    def __init__(self, *, cache_ttl_s: float = 30.0) -> None:
        port = os.environ.get("BENTOML_PORT", "3333")
        base_url = os.environ.get("BENTO_SERVER_URL", f"http://localhost:{port}")
        super().__init__("BentoML", base_url, cache_ttl_s=cache_ttl_s)

    def _fetch(self, endpoint: str) -> dict[str, Any]:
        """Fetch from BentoML endpoint."""
        url = f"{self.base_url}/{endpoint}"
        req = urllib.request.Request(url, method="GET")  # noqa: S310
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, dict):
                return data
            return {"data": data}

    def get_health(self) -> dict[str, Any]:
        """Check BentoML server health."""
        return self.query("health")

    def get_metrics(self) -> dict[str, Any]:
        """Get serving metrics."""
        return self.query("metrics")
