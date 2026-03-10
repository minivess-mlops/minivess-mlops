"""Prefect data router for the dashboard."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/flows")
async def get_flow_runs(request: Request) -> dict[str, Any]:
    """List recent Prefect flow runs."""
    adapter = request.app.state.adapters["prefect"]
    result: dict[str, Any] = adapter.get_flow_runs()
    return result
