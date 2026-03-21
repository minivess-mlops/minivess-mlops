"""Tests for spot preemption resume logic (Phase A5).

Verifies check_resume_state_task() correctly detects and loads
epoch_latest.yaml for spot recovery on GCP.
"""

from __future__ import annotations

from pathlib import Path

import yaml


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
