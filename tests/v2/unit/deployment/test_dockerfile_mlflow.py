"""Tests for Dockerfile.mlflow — baked-in PostgreSQL + S3 deps.

MLflow needs psycopg2-binary (PostgreSQL backend) and boto3 (S3/MinIO artifacts).
Baking them in at build time avoids pip install at container startup.

Rule #16: No regex. Use Path.read_text().splitlines() and yaml.safe_load().
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent.parent.parent
DOCKERFILE_MLFLOW = ROOT / "deployment" / "docker" / "Dockerfile.mlflow"
DOCKER_COMPOSE = ROOT / "deployment" / "docker-compose.yml"


def test_dockerfile_mlflow_exists() -> None:
    assert DOCKERFILE_MLFLOW.exists(), (
        "deployment/docker/Dockerfile.mlflow not found. "
        "Create it to bake psycopg2-binary and boto3 into the MLflow image."
    )


def test_mlflow_dockerfile_has_pinned_psycopg2() -> None:
    content = DOCKERFILE_MLFLOW.read_text(encoding="utf-8")
    assert "psycopg2-binary" in content, (
        "Dockerfile.mlflow must install psycopg2-binary for PostgreSQL backend support."
    )


def test_mlflow_dockerfile_has_boto3() -> None:
    content = DOCKERFILE_MLFLOW.read_text(encoding="utf-8")
    assert "boto3" in content, (
        "Dockerfile.mlflow must install boto3 for S3/MinIO artifact storage."
    )


def test_mlflow_dockerfile_no_pip_at_runtime() -> None:
    """MLflow service command/entrypoint must not run pip install."""
    compose = yaml.safe_load(DOCKER_COMPOSE.read_text(encoding="utf-8"))
    mlflow_service = compose.get("services", {}).get("mlflow", {})

    command = mlflow_service.get("command", "") or ""
    entrypoint = mlflow_service.get("entrypoint", "") or ""
    combined = str(command) + str(entrypoint)

    assert "pip install" not in combined, (
        f"MLflow service command/entrypoint contains 'pip install'. "
        f"Bake deps into Dockerfile.mlflow instead. "
        f"Command: {command!r}"
    )


def test_mlflow_image_reference_in_compose() -> None:
    """MLflow service must use build: (custom Dockerfile), not bare image:."""
    compose = yaml.safe_load(DOCKER_COMPOSE.read_text(encoding="utf-8"))
    mlflow_service = compose.get("services", {}).get("mlflow", {})

    has_build = "build" in mlflow_service
    assert has_build, (
        "MLflow service in docker-compose.yml must use 'build:' referencing "
        "Dockerfile.mlflow. Change 'image: ghcr.io/mlflow/mlflow:...' to "
        "'build: {context: ., dockerfile: deployment/docker/Dockerfile.mlflow}'."
    )
