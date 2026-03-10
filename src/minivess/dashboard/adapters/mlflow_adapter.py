"""MLflow REST API adapter for the dashboard."""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any

from minivess.dashboard.adapters.base import ServiceAdapter

logger = logging.getLogger(__name__)


class MLflowAdapter(ServiceAdapter):
    """Adapter for MLflow experiment tracking REST API.

    Reads MLFLOW_TRACKING_URI from environment (Rule #22).
    """

    def __init__(self, *, cache_ttl_s: float = 30.0) -> None:
        base_url = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        super().__init__("MLflow", base_url, cache_ttl_s=cache_ttl_s)

    def _fetch(self, endpoint: str) -> dict[str, Any]:
        """Fetch from MLflow REST API."""
        url = f"{self.base_url}/api/2.0/mlflow/{endpoint}"
        req = urllib.request.Request(url, method="GET")  # noqa: S310
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            result: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
            return result

    def get_experiments(self) -> dict[str, Any]:
        """List all experiments."""
        return self.query("experiments/list")

    def get_runs(self, experiment_id: str = "0") -> dict[str, Any]:
        """Search runs in an experiment."""
        return self.query(f"runs/search?experiment_id={experiment_id}")
