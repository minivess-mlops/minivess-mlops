"""Orchestration package — Prefect compatibility layer for MinIVess MLOps."""

from __future__ import annotations

from minivess.orchestration._prefect_compat import (
    PREFECT_AVAILABLE,
    flow,
    get_run_logger,
    task,
)

__all__ = ["PREFECT_AVAILABLE", "flow", "get_run_logger", "task"]
