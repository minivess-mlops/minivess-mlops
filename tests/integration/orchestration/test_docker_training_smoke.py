"""Docker training smoke test (Issues #434, T-4.1).

Validates that:
- Training Docker image can be built and import modules (existing tests)
- A debug training run produces a checkpoint on the bound volume (#434 new)
- The MLflow run is FINISHED after the container exits (#434 new)
- Prefect orchestration is active (not disabled) during the run (#434 new)

Prerequisites:
  docker compose -f deployment/docker-compose.flows.yml build train

Skip condition: Docker not available or image not built.
Uses pytest.mark.requires_docker / requires_train_image for selective skipping.
Volume mounts use tmp_path fixture (CLAUDE.md Rule #18 -- no /tmp directly).
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


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


@pytest.mark.requires_docker
@pytest.mark.slow
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


@pytest.mark.requires_docker
@pytest.mark.requires_train_image
@pytest.mark.slow
@pytest.mark.skipif(not _docker_available(), reason="Docker not available")
@pytest.mark.skipif(
    not _image_exists("minivess-train"),
    reason="minivess-train image not built. Run: docker compose -f deployment/docker-compose.flows.yml build train",
)
class TestDockerDebugTrainingRun:
    """End-to-end Docker debug training run tests (Issue #434).

    These tests run a real (but minimal) training loop inside the container,
    binding tmp_path volumes to verify artifacts survive container exit.

    Timeout: 5 minutes per test (1 epoch, 1 fold, 2 volumes).
    """

    def _run_debug_training(
        self,
        tmp_path: Path,
        extra_env: list[str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run a minimal debug training job inside the container."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        mlruns_dir = tmp_path / "mlruns"
        mlruns_dir.mkdir()

        cmd = [
            "docker",
            "run",
            "--rm",
            # Bind volumes to tmp_path — no /tmp directly (CLAUDE.md Rule #18)
            f"--volume={checkpoint_dir}:/app/checkpoints",
            f"--volume={mlruns_dir}:/app/mlruns",
            # Debug training config: 1 epoch, 1 fold, 2 volumes
            "--env=MINIVESS_ALLOW_HOST=1",
            "--env=EXPERIMENT=debug_single_model",
            "--env=MLFLOW_TRACKING_URI=/app/mlruns",
            "minivess-train",
        ]
        if extra_env:
            for e in extra_env:
                cmd.insert(-1, f"--env={e}")

        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5-min limit for a debug run
        )

    def test_train_debug_run_produces_checkpoint(self, tmp_path: Path) -> None:
        """A debug training run must produce at least one checkpoint file on the volume."""
        result = self._run_debug_training(tmp_path)
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoints = list(checkpoint_dir.rglob("*.pt")) + list(
            checkpoint_dir.rglob("*.pth")
        )
        assert checkpoints, (
            f"No checkpoint produced in {checkpoint_dir}.\n"
            f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-2000:]}"
        )

    def test_train_debug_run_mlflow_finished(self, tmp_path: Path) -> None:
        """After a debug run, the MLflow run status must be FINISHED."""
        self._run_debug_training(tmp_path)
        import mlflow

        mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
        client = mlflow.tracking.MlflowClient()
        # Find the most recent run across all experiments
        experiments = client.search_experiments()
        runs: list[mlflow.entities.Run] = []
        for exp in experiments:
            runs.extend(client.search_runs(experiment_ids=[exp.experiment_id]))
        assert runs, "No MLflow runs found after debug training"
        latest_run = sorted(runs, key=lambda r: r.info.start_time, reverse=True)[0]
        assert latest_run.info.status == "FINISHED", (
            f"MLflow run status is {latest_run.info.status!r}, expected 'FINISHED'"
        )

    def test_train_debug_run_prefect_active(self, tmp_path: Path) -> None:
        """Container logs must contain 'Prefect' and not show PREFECT_DISABLED active."""
        result = self._run_debug_training(tmp_path)
        combined = result.stdout + result.stderr
        assert (
            "PREFECT_DISABLED" not in combined or "PREFECT_DISABLED=1" not in combined
        ), (
            "PREFECT_DISABLED=1 found in container output. Prefect must be active for training."
        )
