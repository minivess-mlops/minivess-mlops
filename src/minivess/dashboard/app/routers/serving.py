"""BentoML serving router for the dashboard."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def get_serving_health(request: Request) -> dict[str, Any]:
    """Check BentoML serving health."""
    adapter = request.app.state.adapters["bentoml"]
    result: dict[str, Any] = adapter.get_health()
    return result


@router.get("/metrics")
async def get_serving_metrics(request: Request) -> dict[str, Any]:
    """Get BentoML serving metrics."""
    adapter = request.app.state.adapters["bentoml"]
    result: dict[str, Any] = adapter.get_metrics()
    return result
