"""Prefect flow deployment configuration.

Maps flows to work pools and defines deployment settings for
each flow in the pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Flow → Work Pool mapping
FLOW_WORK_POOL_MAP: dict[str, str] = {
    "acquisition": "cpu-pool",
    "data": "cpu-pool",
    "train": "gpu-pool",
    "analyze": "cpu-pool",
    "deploy": "cpu-pool",
    "dashboard": "cpu-pool",
    "qa": "cpu-pool",
}

# Flow → Docker image mapping
FLOW_IMAGE_MAP: dict[str, str] = {
    "acquisition": "minivess-acquisition:latest",
    "data": "minivess-data:latest",
    "train": "minivess-train:latest",
    "analyze": "minivess-analyze:latest",
    "deploy": "minivess-deploy:latest",
    "dashboard": "minivess-dashboard:latest",
    "qa": "minivess-qa:latest",
}


def get_flow_deployment_config(flow_name: str) -> dict[str, Any]:
    """Get deployment configuration for a specific flow.

    Parameters
    ----------
    flow_name:
        Name of the flow (e.g., 'train', 'data').

    Returns
    -------
    Dict with ``flow_name``, ``work_pool``, ``image``, ``concurrency_limit``.
    """
    work_pool = FLOW_WORK_POOL_MAP.get(flow_name, "cpu-pool")
    image = FLOW_IMAGE_MAP.get(flow_name, "minivess-base:latest")

    return {
        "flow_name": flow_name,
        "work_pool": work_pool,
        "image": image,
        "concurrency_limit": 1 if flow_name == "train" else 4,
    }
