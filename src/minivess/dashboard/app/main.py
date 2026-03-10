"""Dashboard FastAPI application — unified integration hub.

Serves as a single-pane-of-glass REST API aggregating data from
MLflow, Prefect, BentoML, and other platform services. The React
frontend consumes these endpoints.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from minivess.dashboard.adapters.bentoml_adapter import BentoMLAdapter
from minivess.dashboard.adapters.health_adapter import HealthAdapter
from minivess.dashboard.adapters.mlflow_adapter import MLflowAdapter
from minivess.dashboard.adapters.prefect_adapter import PrefectAdapter
from minivess.dashboard.app.routers import health, mlflow, prefect, serving

# Global adapter instances (initialized at lifespan startup)
_adapters: dict[str, Any] = {}
_health_adapter: HealthAdapter | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Initialize adapters on startup, clean up on shutdown."""
    global _health_adapter  # noqa: PLW0603

    mlflow_adapter = MLflowAdapter()
    prefect_adapter = PrefectAdapter()
    bentoml_adapter = BentoMLAdapter()

    _adapters["mlflow"] = mlflow_adapter
    _adapters["prefect"] = prefect_adapter
    _adapters["bentoml"] = bentoml_adapter

    _health_adapter = HealthAdapter([mlflow_adapter, prefect_adapter, bentoml_adapter])

    # Store in app state for router access
    app.state.adapters = _adapters
    app.state.health_adapter = _health_adapter

    yield

    _adapters.clear()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="MinIVess Dashboard API",
        description="Unified integration hub for MinIVess MLOps platform",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS for React frontend
    ui_port = os.environ.get("DASHBOARD_UI_PORT", "3002")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            f"http://localhost:{ui_port}",
            "http://localhost:5173",  # Vite dev server
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(health.router, prefix="/api/health", tags=["health"])
    app.include_router(mlflow.router, prefix="/api/mlflow", tags=["mlflow"])
    app.include_router(prefect.router, prefix="/api/prefect", tags=["prefect"])
    app.include_router(serving.router, prefix="/api/serving", tags=["serving"])

    @app.get("/health")
    async def root_health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
