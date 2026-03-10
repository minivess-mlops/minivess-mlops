"""Prefect REST API adapter for the dashboard."""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any

from minivess.dashboard.adapters.base import ServiceAdapter

logger = logging.getLogger(__name__)


class PrefectAdapter(ServiceAdapter):
    """Adapter for Prefect Server REST API.

    Reads PREFECT_API_URL from environment (Rule #22).
    """

    def __init__(self, *, cache_ttl_s: float = 30.0) -> None:
        base_url = os.environ.get("PREFECT_API_URL", "http://localhost:4200/api")
        super().__init__("Prefect", base_url, cache_ttl_s=cache_ttl_s)

    def _fetch(self, endpoint: str) -> dict[str, Any]:
        """Fetch from Prefect REST API."""
        url = f"{self.base_url}/{endpoint}"
        req = urllib.request.Request(url, method="GET")  # noqa: S310
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, list):
                return {"items": data, "count": len(data)}
            result: dict[str, Any] = data
            return result

    def get_flow_runs(self) -> dict[str, Any]:
        """List recent flow runs."""
        return self.query("flow_runs")
