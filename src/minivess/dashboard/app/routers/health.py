"""Health and QA monitoring router (#342 merged into dashboard)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/")
async def get_health(request: Request) -> dict[str, Any]:
    """Aggregated health status across all adapters."""
    health_adapter = request.app.state.health_adapter
    summary = health_adapter.check_all()
    result: dict[str, Any] = summary.to_dict()
    return result


@router.get("/adapters")
async def get_adapter_statuses(request: Request) -> dict[str, Any]:
    """Individual adapter health statuses."""
    adapters = request.app.state.adapters
    return {name: adapter.status().__dict__ for name, adapter in adapters.items()}
