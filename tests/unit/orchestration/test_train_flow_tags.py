"""Tests for train_flow MLflow checkpoint tagging — Issue #555.

Verifies:
- log_fold_results_task() tags checkpoint_dir_fold_{N} on the MLflow run
- TrainingFlowResult has checkpoint_dirs field

Plan: docs/planning/prefect-flow-connectivity-execution-plan.xml Phase 0 (T0.3)
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def mock_fold_result() -> dict:
    return {
        "best_val_loss": 0.5,
        "final_epoch": 10,
        "history": {"val_loss": [0.8, 0.6, 0.5]},
    }


class TestLogFoldResultsTaskCheckpointTag:
    def test_log_fold_results_task_tags_checkpoint_dir(
        self, tmp_path: Path, mock_fold_result: dict
    ) -> None:
        """log_fold_results_task must set checkpoint_dir_fold_{N} tag."""
        import mlflow
        from mlflow.tracking import MlflowClient

        from minivess.orchestration.flows.train_flow import log_fold_results_task

        tracking_uri = str(tmp_path / "mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("test_train_ckpt_tag")
        ckpt_dir = tmp_path / "checkpoints" / "fold_0"
        ckpt_dir.mkdir(parents=True)

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            log_fold_results_task(
                fold_id=0,
                result=mock_fold_result,
                mlflow_run_id=run_id,
                checkpoint_dir=ckpt_dir,
            )

        client = MlflowClient(tracking_uri=tracking_uri)
        run_data = client.get_run(run_id)
        assert run_data.data.tags.get("checkpoint_dir_fold_0") == str(ckpt_dir), (
            "checkpoint_dir_fold_0 tag must be set for post-training to find checkpoints"
        )

    def test_log_fold_results_task_tags_different_fold_indices(
        self, tmp_path: Path, mock_fold_result: dict
    ) -> None:
        """checkpoint_dir tag key must reflect the fold_id."""
        import mlflow
        from mlflow.tracking import MlflowClient

        from minivess.orchestration.flows.train_flow import log_fold_results_task

        tracking_uri = str(tmp_path / "mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("test_train_fold_idx")
        ckpt_dir = tmp_path / "checkpoints" / "fold_2"
        ckpt_dir.mkdir(parents=True)

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            log_fold_results_task(
                fold_id=2,
                result=mock_fold_result,
                mlflow_run_id=run_id,
                checkpoint_dir=ckpt_dir,
            )

        client = MlflowClient(tracking_uri=tracking_uri)
        run_data = client.get_run(run_id)
        assert "checkpoint_dir_fold_2" in run_data.data.tags
        assert "checkpoint_dir_fold_0" not in run_data.data.tags

    def test_log_fold_results_task_no_tag_when_checkpoint_dir_is_none(
        self, tmp_path: Path, mock_fold_result: dict
    ) -> None:
        """When checkpoint_dir is None, no checkpoint_dir tag is written."""
        import mlflow
        from mlflow.tracking import MlflowClient

        from minivess.orchestration.flows.train_flow import log_fold_results_task

        tracking_uri = str(tmp_path / "mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("test_train_no_ckpt")

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            log_fold_results_task(
                fold_id=0,
                result=mock_fold_result,
                mlflow_run_id=run_id,
                checkpoint_dir=None,
            )

        client = MlflowClient(tracking_uri=tracking_uri)
        run_data = client.get_run(run_id)
        assert "checkpoint_dir_fold_0" not in run_data.data.tags

    def test_log_fold_results_task_still_logs_metrics(
        self, tmp_path: Path, mock_fold_result: dict
    ) -> None:
        """Existing metric logging must still work after signature change."""
        import mlflow
        from mlflow.tracking import MlflowClient

        from minivess.orchestration.flows.train_flow import log_fold_results_task

        tracking_uri = str(tmp_path / "mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("test_train_metrics")
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            log_fold_results_task(
                fold_id=0,
                result=mock_fold_result,
                mlflow_run_id=run_id,
                checkpoint_dir=ckpt_dir,
            )

        client = MlflowClient(tracking_uri=tracking_uri)
        run_data = client.get_run(run_id)
        metrics = run_data.data.metrics
        assert "fold/0/best_val_loss" in metrics
        assert metrics["fold/0/best_val_loss"] == pytest.approx(0.5)


class TestParentRunIdTag:
    """Parent training run must tag itself with parent_run_id for downstream discovery."""

    def test_training_run_has_parent_run_id_tag(self, tmp_path: Path) -> None:
        """The parent MLflow run must have a parent_run_id tag equal to its own run_id."""
        import mlflow
        from mlflow.tracking import MlflowClient

        tracking_uri = str(tmp_path / "mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("test_parent_run_id")

        with mlflow.start_run(tags={"flow_name": "training-flow"}) as run:
            run_id = run.info.run_id
            mlflow.set_tag("parent_run_id", run_id)

        client = MlflowClient(tracking_uri=tracking_uri)
        run_data = client.get_run(run_id)
        assert run_data.data.tags.get("parent_run_id") == run_id, (
            "parent_run_id tag must be set to the run's own ID for downstream flows"
        )

    def test_parent_run_id_matches_active_run(self, tmp_path: Path) -> None:
        """parent_run_id must equal the active run's info.run_id (self-referential)."""
        import mlflow
        from mlflow.tracking import MlflowClient

        tracking_uri = str(tmp_path / "mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("test_parent_self_ref")

        with mlflow.start_run() as run:
            mlflow.set_tag("parent_run_id", run.info.run_id)
            captured_id = run.info.run_id

        client = MlflowClient(tracking_uri=tracking_uri)
        run_data = client.get_run(captured_id)
        assert run_data.data.tags["parent_run_id"] == captured_id


class TestTrainingFlowResultCheckpointDirs:
    def test_training_flow_result_has_checkpoint_dirs_field(
        self, tmp_path: Path
    ) -> None:
        """TrainingFlowResult must have checkpoint_dirs dict field."""
        from minivess.orchestration.flows.train_flow import TrainingFlowResult

        result = TrainingFlowResult(
            flow_name="train",
            n_folds=2,
            loss_name="dice_ce",
            model_family="dynunet",
            checkpoint_dirs={0: tmp_path / "fold_0", 1: tmp_path / "fold_1"},
        )
        assert result.checkpoint_dirs[0] == tmp_path / "fold_0"
        assert len(result.checkpoint_dirs) == 2

    def test_training_flow_result_checkpoint_dirs_defaults_empty(self) -> None:
        """checkpoint_dirs defaults to empty dict when not provided."""
        from minivess.orchestration.flows.train_flow import TrainingFlowResult

        result = TrainingFlowResult(
            flow_name="train",
            n_folds=1,
            loss_name="dice_ce",
            model_family="dynunet",
        )
        assert result.checkpoint_dirs == {}
