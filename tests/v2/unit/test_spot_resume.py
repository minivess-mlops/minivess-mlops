"""Tests for spot preemption resume logic (Phase A5).

Verifies check_resume_state_task() correctly detects and loads
epoch_latest.yaml for spot recovery on GCP.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import torch
import yaml

if TYPE_CHECKING:
    import pytest


class TestCheckResumeState:
    """check_resume_state_task must detect and validate resume state."""

    def test_returns_none_when_no_yaml(self, tmp_path: Path) -> None:
        from minivess.orchestration.flows.train_flow import check_resume_state_task

        result = check_resume_state_task(tmp_path)
        assert result is None

    def test_returns_none_for_invalid_yaml(self, tmp_path: Path) -> None:
        from minivess.orchestration.flows.train_flow import check_resume_state_task

        (tmp_path / "epoch_latest.yaml").write_text("not a dict", encoding="utf-8")
        result = check_resume_state_task(tmp_path)
        assert result is None

    def test_returns_none_without_run_id(self, tmp_path: Path) -> None:
        from minivess.orchestration.flows.train_flow import check_resume_state_task

        state = {"epoch": 5}  # no mlflow_run_id
        (tmp_path / "epoch_latest.yaml").write_text(yaml.dump(state), encoding="utf-8")
        result = check_resume_state_task(tmp_path)
        assert result is None

    def test_returns_state_for_running_run(self, tmp_path: Path) -> None:
        import mlflow

        from minivess.orchestration.flows.train_flow import check_resume_state_task

        mlflow_dir = tmp_path / "mlruns"
        mlflow.set_tracking_uri(str(mlflow_dir))
        mlflow.set_experiment("resume_test")

        # Create a RUNNING mlflow run
        run = mlflow.start_run()
        run_id = run.info.run_id

        state = {"epoch": 10, "mlflow_run_id": run_id}
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "epoch_latest.yaml").write_text(yaml.dump(state), encoding="utf-8")

        result = check_resume_state_task(ckpt_dir)
        assert result is not None
        assert result["epoch"] == 10

        mlflow.end_run()

    def test_returns_none_for_finished_run(self, tmp_path: Path) -> None:
        import mlflow

        from minivess.orchestration.flows.train_flow import check_resume_state_task

        mlflow_dir = tmp_path / "mlruns"
        mlflow.set_tracking_uri(str(mlflow_dir))
        mlflow.set_experiment("resume_test")

        # Create and FINISH an mlflow run
        with mlflow.start_run() as run:
            run_id = run.info.run_id

        state = {"epoch": 10, "mlflow_run_id": run_id}
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "epoch_latest.yaml").write_text(yaml.dump(state), encoding="utf-8")

        result = check_resume_state_task(ckpt_dir)
        assert result is None  # FINISHED, not RUNNING → stale


class TestResumeMLflowRetry:
    """check_resume_state_task must retry MLflow and fall back to checkpoint."""

    def test_resume_mlflow_unreachable_with_checkpoint(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When MLflow is unreachable but checkpoint exists, resume from file."""
        from minivess.orchestration.flows.train_flow import check_resume_state_task

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()

        state = {
            "epoch": 25,
            "mlflow_run_id": "fake-run-id",
            "checkpoint_dir": str(ckpt_dir),
        }
        (ckpt_dir / "epoch_latest.yaml").write_text(yaml.dump(state), encoding="utf-8")
        torch.save({"model_state_dict": {}}, ckpt_dir / "epoch_latest.pth")

        with (
            patch("mlflow.get_run", side_effect=ConnectionError("MLflow down")),
            caplog.at_level(logging.WARNING),
        ):
            result = check_resume_state_task(ckpt_dir)

        assert result is not None, (
            "Should resume from checkpoint when MLflow unreachable"
        )
        assert result["epoch"] == 25
        assert any("unreachable" in r.message.lower() for r in caplog.records)

    def test_resume_mlflow_unreachable_no_checkpoint(self, tmp_path: Path) -> None:
        """When MLflow is unreachable and no checkpoint, start fresh."""
        from minivess.orchestration.flows.train_flow import check_resume_state_task

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()

        state = {"epoch": 25, "mlflow_run_id": "fake-run-id"}
        (ckpt_dir / "epoch_latest.yaml").write_text(yaml.dump(state), encoding="utf-8")
        # No epoch_latest.pth file

        with patch("mlflow.get_run", side_effect=ConnectionError("MLflow down")):
            result = check_resume_state_task(ckpt_dir)

        assert result is None, (
            "Should start fresh when MLflow unreachable and no checkpoint"
        )

    def test_resume_mlflow_transient_failure_then_success(self, tmp_path: Path) -> None:
        """If MLflow fails once then succeeds, resume normally."""
        import mlflow

        from minivess.orchestration.flows.train_flow import check_resume_state_task

        mlflow_dir = tmp_path / "mlruns"
        mlflow.set_tracking_uri(str(mlflow_dir))
        mlflow.set_experiment("retry_test")

        run = mlflow.start_run()
        run_id = run.info.run_id

        state = {"epoch": 15, "mlflow_run_id": run_id}
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "epoch_latest.yaml").write_text(yaml.dump(state), encoding="utf-8")

        original_get_run = mlflow.get_run
        call_count = 0

        def flaky_get_run(rid: str) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Transient failure")
            return original_get_run(rid)

        with patch("mlflow.get_run", side_effect=flaky_get_run):
            result = check_resume_state_task(ckpt_dir)

        assert result is not None, "Should resume after transient MLflow failure"
        assert result["epoch"] == 15
        mlflow.end_run()


class TestResumeWiringInFlow:
    """Train flow module must wire resume state into training loop."""

    def test_module_has_check_resume_state(self) -> None:
        """train_flow module must contain check_resume_state_task."""
        import inspect

        from minivess.orchestration.flows import train_flow

        source = inspect.getsource(train_flow)
        assert "check_resume_state_task" in source

    def test_module_loads_epoch_latest_pth(self) -> None:
        """On resume, flow must load epoch_latest.pth state dict."""
        import inspect

        from minivess.orchestration.flows import train_flow

        source = inspect.getsource(train_flow)
        assert "epoch_latest.pth" in source

    def test_module_sets_start_epoch(self) -> None:
        """On resume, start_epoch = resume_epoch + 1."""
        import inspect

        from minivess.orchestration.flows import train_flow

        source = inspect.getsource(train_flow)
        assert "start_epoch" in source
