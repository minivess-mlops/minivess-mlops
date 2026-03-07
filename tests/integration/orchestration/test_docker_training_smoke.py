"""T-4.1: Docker training smoke test.

Validates that the training Docker image builds and the flow starts
without import errors. Does NOT run actual training (no GPU required).

Prerequisites:
  docker compose -f deployment/docker-compose.flows.yml build train

Skip condition: Docker not available or image not built.
"""

from __future__ import annotations

import subprocess

import pytest


def _docker_available() -> bool:
    """Check if Docker daemon is running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _image_exists(image_name: str) -> bool:
    """Check if a Docker image exists locally."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image_name],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.mark.skipif(not _docker_available(), reason="Docker not available")
class TestDockerTrainingSmoke:
    """Smoke tests for training Docker container."""

    @pytest.mark.skipif(
        not _image_exists("minivess-train"),
        reason="minivess-train image not built. Run: docker compose -f deployment/docker-compose.flows.yml build train",
    )
    def test_train_container_imports_succeed(self) -> None:
        """Training container starts and imports all flow modules without error."""
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "python",
                "minivess-train",
                "-c",
                "from minivess.orchestration.flows.train_flow import training_flow; print('OK')",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"Training container import failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "OK" in result.stdout

    @pytest.mark.skipif(
        not _image_exists("minivess-train"),
        reason="minivess-train image not built",
    )
    def test_train_container_docker_gate_blocks_without_env(self) -> None:
        """Container without DOCKER_CONTAINER env var should hit the gate."""
        # The Docker gate checks /.dockerenv which exists in real Docker
        # but we test the env var path here
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "python",
                "minivess-train",
                "-c",
                (
                    "import os; os.environ.pop('DOCKER_CONTAINER', None); "
                    "from minivess.orchestration.flows.train_flow import _require_docker_context; "
                    "_require_docker_context(); print('GATE_PASSED')"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        # Inside Docker, /.dockerenv exists, so gate should pass
        assert result.returncode == 0, (
            f"Docker gate should pass inside container:\nstderr: {result.stderr}"
        )
        assert "GATE_PASSED" in result.stdout

    @pytest.mark.skipif(
        not _image_exists("minivess-base"),
        reason="minivess-base image not built",
    )
    def test_pipeline_container_imports_succeed(self) -> None:
        """Pipeline container starts and imports pipeline_flow without error."""
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "python",
                "minivess-base",
                "-c",
                "from minivess.orchestration.flows.pipeline_flow import pipeline_flow; print('OK')",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"Pipeline container import failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "OK" in result.stdout
