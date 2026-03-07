"""Deploy flow configuration for BentoML + MLflow + MONAI Deploy.

Defines the configuration schema for Flow 4 (Deployment), including
champion model selection categories, ONNX export settings, BentoML
service configuration, and optional MONAI Deploy MAP packaging.
"""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path  # noqa: TC003 — Pydantic needs Path at runtime

from pydantic import BaseModel, Field


class ChampionCategory(StrEnum):
    """Champion model selection categories.

    Matches the rank-then-aggregate categories from
    ``champion_tagger.rank_then_aggregate()``.
    """

    BALANCED = "balanced"
    TOPOLOGY = "topology"
    OVERLAP = "overlap"


class DeployConfig(BaseModel):
    """Configuration for the deploy flow (Flow 4).

    Parameters
    ----------
    mlruns_dir:
        Root MLflow tracking directory.
    output_dir:
        Directory for deployment artifacts (ONNX, bentofile, docker-compose).
    champion_categories:
        Which champion categories to deploy.
    onnx_opset:
        ONNX opset version for export.
    bento_service_name:
        Name for the BentoML service.
    docker_registry:
        Optional Docker registry for pushing images.
    monai_deploy_enabled:
        Whether to generate MONAI Deploy MAP packaging.
    """

    mlruns_dir: Path = Field(description="Root MLflow tracking directory")
    output_dir: Path = Field(
        default_factory=lambda: Path(
            os.environ.get("DEPLOY_OUTPUT_DIR", "/app/outputs/deploy")
        ),
        description="Directory for deployment artifacts",
    )
    champion_categories: list[ChampionCategory] = Field(
        default_factory=lambda: [
            ChampionCategory.BALANCED,
            ChampionCategory.TOPOLOGY,
            ChampionCategory.OVERLAP,
        ],
        description="Which champion categories to deploy",
    )
    onnx_opset: int = Field(
        default=17,
        ge=11,
        description="ONNX opset version for export",
    )
    bento_service_name: str = Field(
        default="minivess-segmentation",
        description="Name for the BentoML service",
    )
    docker_registry: str | None = Field(
        default=None,
        description="Optional Docker registry for pushing images",
    )
    monai_deploy_enabled: bool = Field(
        default=False,
        description="Whether to generate MONAI Deploy MAP packaging",
    )
    export_aux_heads: bool = Field(
        default=False,
        description="Include auxiliary heads in ONNX export (default: mask-only)",
    )

    @classmethod
    def from_env(cls) -> DeployConfig:
        """Create DeployConfig from environment variables.

        Required:
            MLFLOW_TRACKING_URI: MLflow tracking URI (local path or remote).

        Optional:
            DEPLOY_OUTPUT_DIR: Output directory for artifacts (default: /app/outputs/deploy).

        Raises
        ------
        RuntimeError
            If MLFLOW_TRACKING_URI is not set.
        """
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            raise RuntimeError(
                "Required env var MLFLOW_TRACKING_URI not set.\n"
                "Set it with:\n"
                "  export MLFLOW_TRACKING_URI=file:///path/to/mlruns\n"
                "  # or: export MLFLOW_TRACKING_URI=http://mlflow-server:5000"
            )
        # Strip file:// prefix if present to get the actual path
        mlruns_path = tracking_uri.removeprefix("file://")
        return cls(mlruns_dir=Path(mlruns_path))
