"""MLflow data router for the dashboard."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/experiments")
async def get_experiments(request: Request) -> dict[str, Any]:
    """List MLflow experiments."""
    adapter = request.app.state.adapters["mlflow"]
    result: dict[str, Any] = adapter.get_experiments()
    return result


@router.get("/runs")
async def get_runs(request: Request, experiment_id: str = "0") -> dict[str, Any]:
    """Search runs in an experiment."""
    adapter = request.app.state.adapters["mlflow"]
    result: dict[str, Any] = adapter.get_runs(experiment_id)
    return result
