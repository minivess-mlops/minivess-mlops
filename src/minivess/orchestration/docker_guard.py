"""Docker context guard — shared across all Prefect flow modules.

Enforces the STOP protocol (S)ource check: all flows must run inside Docker.
Escape hatch: MINIVESS_ALLOW_HOST=1 for pytest ONLY — never in scripts or
production.

See: docs/planning/minivess-vision-enforcement-plan.md (T-00)
"""

from __future__ import annotations

import os
from pathlib import Path


def require_docker_context(flow_name: str) -> None:
    """Raise RuntimeError if not running inside a Docker container.

    Checks for:
    1. MINIVESS_ALLOW_HOST=1 (test escape hatch)
    2. DOCKER_CONTAINER env var (set by our Docker images)
    3. /.dockerenv file (standard Docker marker)

    Parameters
    ----------
    flow_name:
        Human-readable flow name for the error message.
    """
    if os.environ.get("MINIVESS_ALLOW_HOST") == "1":
        return
    if os.environ.get("DOCKER_CONTAINER"):
        return
    if Path("/.dockerenv").exists():
        return
    raise RuntimeError(
        f"{flow_name} must run inside a Docker container.\n"
        f"Run: docker compose -f deployment/docker-compose.flows.yml run {flow_name}\n\n"
        "Escape hatch (pytest ONLY): export MINIVESS_ALLOW_HOST=1\n"
        "See: docs/planning/minivess-vision-enforcement-plan.md"
    )
