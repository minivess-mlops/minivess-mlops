"""Orchestration package — Prefect integration for MinIVess MLOps."""

from __future__ import annotations

from prefect import flow, get_run_logger, task

from minivess.orchestration.deployments import get_work_pool

PREFECT_AVAILABLE = True

__all__ = ["PREFECT_AVAILABLE", "flow", "get_run_logger", "get_work_pool", "task"]
