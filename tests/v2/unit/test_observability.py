"""Unit tests for MLflow tracking, DuckDB analytics, and trainer integration."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from pathlib import Path

from minivess.adapters.segresnet import SegResNetAdapter
from minivess.config.models import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    ModelFamily,
    TrainingConfig,
)
from minivess.pipeline.trainer import SegmentationTrainer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def experiment_config() -> ExperimentConfig:
    """Minimal ExperimentConfig for tracking tests."""
    return ExperimentConfig(
        experiment_name="test-experiment",
        run_name="test-run",
        data=DataConfig(dataset_name="minivess"),
        model=ModelConfig(
            family=ModelFamily.MONAI_SEGRESNET,
            name="test-model",
            in_channels=1,
            out_channels=2,
        ),
        training=TrainingConfig(
            max_epochs=2,
            batch_size=1,
            learning_rate=1e-3,
            mixed_precision=False,
            warmup_epochs=1,
        ),
    )


@pytest.fixture()
def local_tracking_uri(tmp_path: Path) -> str:
    """Return a file-backed MLflow tracking URI in tmp_path."""
    mlruns_dir = tmp_path / "mlruns"
    return str(mlruns_dir)


@pytest.fixture()
def synthetic_loader() -> list[dict[str, torch.Tensor]]:
    """Tiny synthetic data loader (single batch)."""
    return [
        {
            "image": torch.randn(1, 1, 16, 16, 8),
            "label": torch.randint(0, 2, (1, 1, 16, 16, 8)),
        }
    ]


# ---------------------------------------------------------------------------
# T1: resolve_tracking_uri + ExperimentTracker with local backend
# ---------------------------------------------------------------------------


class TestResolveTrackingUri:
    """Test tracking URI resolution priority: env → config → default."""

    def test_default_is_mlruns(self) -> None:
        """Without env var or explicit param, default should be 'mlruns'."""
        from minivess.observability.tracking import resolve_tracking_uri

        # Clear env var if set
        env_backup = os.environ.pop("MLFLOW_TRACKING_URI", None)
        try:
            uri = resolve_tracking_uri()
            assert uri == "mlruns"
        finally:
            if env_backup is not None:
                os.environ["MLFLOW_TRACKING_URI"] = env_backup

    def test_env_var_override(self, tmp_path: Path) -> None:
        """MLFLOW_TRACKING_URI env var should take precedence."""
        from minivess.observability.tracking import resolve_tracking_uri

        env_backup = os.environ.get("MLFLOW_TRACKING_URI")
        try:
            os.environ["MLFLOW_TRACKING_URI"] = str(tmp_path / "custom-mlruns")
            uri = resolve_tracking_uri()
            assert str(tmp_path / "custom-mlruns") in uri
        finally:
            if env_backup is not None:
                os.environ["MLFLOW_TRACKING_URI"] = env_backup
            else:
                os.environ.pop("MLFLOW_TRACKING_URI", None)

    def test_explicit_param_wins(self) -> None:
        """Explicit tracking_uri parameter should override everything."""
        from minivess.observability.tracking import resolve_tracking_uri

        uri = resolve_tracking_uri(tracking_uri="http://custom:5000")
        assert uri == "http://custom:5000"


class TestExperimentTrackerLocalBackend:
    """Test ExperimentTracker with file-backed MLflow (no server needed)."""

    def test_tracker_creates_with_local_backend(
        self,
        experiment_config: ExperimentConfig,
        local_tracking_uri: str,
    ) -> None:
        from minivess.observability.tracking import ExperimentTracker

        tracker = ExperimentTracker(
            experiment_config,
            tracking_uri=local_tracking_uri,
        )
        assert tracker.config == experiment_config

    def test_start_run_creates_experiment(
        self,
        experiment_config: ExperimentConfig,
        local_tracking_uri: str,
    ) -> None:
        """start_run should create an MLflow experiment and run."""
        import mlflow

        from minivess.observability.tracking import ExperimentTracker

        tracker = ExperimentTracker(
            experiment_config,
            tracking_uri=local_tracking_uri,
        )
        with tracker.start_run() as run_id:
            assert run_id is not None
            assert isinstance(run_id, str)
            assert len(run_id) > 0

        # Verify experiment was created
        mlflow.set_tracking_uri(local_tracking_uri)
        experiment = mlflow.get_experiment_by_name("test-experiment")
        assert experiment is not None

    def test_log_epoch_metrics(
        self,
        experiment_config: ExperimentConfig,
        local_tracking_uri: str,
    ) -> None:
        """Logged metrics should be retrievable from the run."""
        from mlflow.tracking import MlflowClient

        from minivess.observability.tracking import ExperimentTracker

        tracker = ExperimentTracker(
            experiment_config,
            tracking_uri=local_tracking_uri,
        )
        with tracker.start_run() as run_id:
            tracker.log_epoch_metrics(
                {"train_loss": 0.5, "val_loss": 0.3},
                step=1,
            )

        client = MlflowClient(tracking_uri=local_tracking_uri)
        run = client.get_run(run_id)
        assert float(run.data.metrics["train_loss"]) == pytest.approx(0.5)
        assert float(run.data.metrics["val_loss"]) == pytest.approx(0.3)

    def test_log_model_info(
        self,
        experiment_config: ExperimentConfig,
        local_tracking_uri: str,
    ) -> None:
        """Model info should be logged as params."""
        from mlflow.tracking import MlflowClient

        from minivess.observability.tracking import ExperimentTracker

        tracker = ExperimentTracker(
            experiment_config,
            tracking_uri=local_tracking_uri,
        )
        model = SegResNetAdapter(experiment_config.model)
        with tracker.start_run() as run_id:
            tracker.log_model_info(model)

        client = MlflowClient(tracking_uri=local_tracking_uri)
        run = client.get_run(run_id)
        assert "model_family" in run.data.params
        assert run.data.params["model_family"] == "segresnet"
        assert "trainable_parameters" in run.data.metrics

    def test_log_artifact(
        self,
        experiment_config: ExperimentConfig,
        local_tracking_uri: str,
        tmp_path: Path,
    ) -> None:
        """log_artifact should save a file to the run's artifact store."""
        from mlflow.tracking import MlflowClient

        from minivess.observability.tracking import ExperimentTracker

        # Create a dummy artifact
        artifact_file = tmp_path / "metrics.json"
        artifact_file.write_text('{"dice": 0.85}', encoding="utf-8")

        tracker = ExperimentTracker(
            experiment_config,
            tracking_uri=local_tracking_uri,
        )
        with tracker.start_run() as run_id:
            tracker.log_artifact(artifact_file)

        client = MlflowClient(tracking_uri=local_tracking_uri)
        artifacts = client.list_artifacts(run_id)
        artifact_names = [a.path for a in artifacts]
        assert "metrics.json" in artifact_names

    def test_log_test_set_hash(
        self,
        experiment_config: ExperimentConfig,
        local_tracking_uri: str,
    ) -> None:
        """log_test_set_hash should record SHA256 as a tag."""
        from mlflow.tracking import MlflowClient

        from minivess.observability.tracking import ExperimentTracker

        tracker = ExperimentTracker(
            experiment_config,
            tracking_uri=local_tracking_uri,
        )
        test_paths = ["data/test/img_001.nii.gz", "data/test/img_002.nii.gz"]
        with tracker.start_run() as run_id:
            hash_value = tracker.log_test_set_hash(test_paths)

        assert len(hash_value) == 64  # SHA256 hex digest

        client = MlflowClient(tracking_uri=local_tracking_uri)
        run = client.get_run(run_id)
        assert run.data.tags["test_set_sha256"] == hash_value

    def test_log_config_params(
        self,
        experiment_config: ExperimentConfig,
        local_tracking_uri: str,
    ) -> None:
        """start_run should auto-log config as params."""
        from mlflow.tracking import MlflowClient

        from minivess.observability.tracking import ExperimentTracker

        tracker = ExperimentTracker(
            experiment_config,
            tracking_uri=local_tracking_uri,
        )
        with tracker.start_run() as run_id:
            pass  # _log_config called automatically

        client = MlflowClient(tracking_uri=local_tracking_uri)
        run = client.get_run(run_id)
        assert run.data.params["model_name"] == "test-model"
        assert run.data.params["learning_rate"] == "0.001"
        assert run.data.params["max_epochs"] == "2"


# ---------------------------------------------------------------------------
# T2: Trainer + Tracker integration
# ---------------------------------------------------------------------------


class TestTrainerMLflowIntegration:
    """Test that SegmentationTrainer logs to MLflow via ExperimentTracker."""

    def test_trainer_accepts_tracker(
        self,
        experiment_config: ExperimentConfig,
        local_tracking_uri: str,
    ) -> None:
        """Trainer constructor should accept an optional tracker."""
        from minivess.observability.tracking import ExperimentTracker

        tracker = ExperimentTracker(
            experiment_config,
            tracking_uri=local_tracking_uri,
        )
        model = SegResNetAdapter(experiment_config.model)
        trainer = SegmentationTrainer(
            model,
            experiment_config.training,
            device="cpu",
            tracker=tracker,
        )
        assert trainer.tracker is tracker

    def test_trainer_works_without_tracker(
        self, synthetic_loader: list[dict[str, torch.Tensor]]
    ) -> None:
        """Trainer without tracker should work as before (backward compat)."""
        config = TrainingConfig(
            max_epochs=2,
            batch_size=1,
            learning_rate=1e-3,
            mixed_precision=False,
            warmup_epochs=1,
        )
        model = SegResNetAdapter(
            ModelConfig(
                family=ModelFamily.MONAI_SEGRESNET,
                name="no-tracker",
                in_channels=1,
                out_channels=2,
            )
        )
        trainer = SegmentationTrainer(model, config, device="cpu")
        summary = trainer.fit(synthetic_loader, synthetic_loader)
        assert "best_val_loss" in summary
        assert summary["final_epoch"] == 2

    def test_trainer_logs_to_mlflow(
        self,
        experiment_config: ExperimentConfig,
        local_tracking_uri: str,
        synthetic_loader: list[dict[str, torch.Tensor]],
    ) -> None:
        """fit() with tracker should log epoch metrics to MLflow."""
        from mlflow.tracking import MlflowClient

        from minivess.observability.tracking import ExperimentTracker

        tracker = ExperimentTracker(
            experiment_config,
            tracking_uri=local_tracking_uri,
        )
        model = SegResNetAdapter(experiment_config.model)
        trainer = SegmentationTrainer(
            model,
            experiment_config.training,
            device="cpu",
            tracker=tracker,
        )

        with tracker.start_run() as run_id:
            summary = trainer.fit(synthetic_loader, synthetic_loader)

        # Verify metrics logged
        client = MlflowClient(tracking_uri=local_tracking_uri)
        run = client.get_run(run_id)
        assert "train_loss" in run.data.metrics
        assert "val_loss" in run.data.metrics
        assert summary["final_epoch"] == 2

    def test_trainer_logs_learning_rate(
        self,
        experiment_config: ExperimentConfig,
        local_tracking_uri: str,
        synthetic_loader: list[dict[str, torch.Tensor]],
    ) -> None:
        """fit() should log learning rate at each epoch."""
        from mlflow.tracking import MlflowClient

        from minivess.observability.tracking import ExperimentTracker

        tracker = ExperimentTracker(
            experiment_config,
            tracking_uri=local_tracking_uri,
        )
        model = SegResNetAdapter(experiment_config.model)
        trainer = SegmentationTrainer(
            model,
            experiment_config.training,
            device="cpu",
            tracker=tracker,
        )

        with tracker.start_run() as run_id:
            trainer.fit(synthetic_loader, synthetic_loader)

        client = MlflowClient(tracking_uri=local_tracking_uri)
        run = client.get_run(run_id)
        assert "learning_rate" in run.data.metrics

    def test_trainer_logs_checkpoint_artifact(
        self,
        experiment_config: ExperimentConfig,
        local_tracking_uri: str,
        synthetic_loader: list[dict[str, torch.Tensor]],
        tmp_path: Path,
    ) -> None:
        """fit() with checkpoint_dir and tracker should log checkpoint artifact."""
        from mlflow.tracking import MlflowClient

        from minivess.observability.tracking import ExperimentTracker

        tracker = ExperimentTracker(
            experiment_config,
            tracking_uri=local_tracking_uri,
        )
        model = SegResNetAdapter(experiment_config.model)
        trainer = SegmentationTrainer(
            model,
            experiment_config.training,
            device="cpu",
            tracker=tracker,
        )
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        with tracker.start_run() as run_id:
            trainer.fit(
                synthetic_loader,
                synthetic_loader,
                checkpoint_dir=checkpoint_dir,
            )

        client = MlflowClient(tracking_uri=local_tracking_uri)
        # Checkpoint is logged under "checkpoints/" artifact path
        top_artifacts = client.list_artifacts(run_id)
        top_names = [a.path for a in top_artifacts]
        assert "checkpoints" in top_names
        # Verify the actual checkpoint file exists inside
        sub_artifacts = client.list_artifacts(run_id, path="checkpoints")
        sub_names = [a.path for a in sub_artifacts]
        assert any("best_model" in name for name in sub_names)


# ---------------------------------------------------------------------------
# T4: DuckDB RunAnalytics
# ---------------------------------------------------------------------------


class TestRunAnalytics:
    """Test DuckDB analytics over MLflow runs (file-backed)."""

    def test_analytics_load_experiment_runs(
        self,
        experiment_config: ExperimentConfig,
        local_tracking_uri: str,
    ) -> None:
        """load_experiment_runs should return a DataFrame of completed runs."""

        from minivess.observability.analytics import RunAnalytics
        from minivess.observability.tracking import ExperimentTracker

        # Create a run with some metrics
        tracker = ExperimentTracker(
            experiment_config,
            tracking_uri=local_tracking_uri,
        )
        with tracker.start_run():
            tracker.log_epoch_metrics({"val_dice": 0.85}, step=1)

        analytics = RunAnalytics(tracking_uri=local_tracking_uri)
        df = analytics.load_experiment_runs("test-experiment")
        assert len(df) == 1
        assert "run_id" in df.columns
        assert "metric_val_dice" in df.columns
        analytics.close()

    def test_analytics_query_sql(
        self,
        experiment_config: ExperimentConfig,
        local_tracking_uri: str,
    ) -> None:
        """SQL queries against registered DataFrames should work."""
        from minivess.observability.analytics import RunAnalytics
        from minivess.observability.tracking import ExperimentTracker

        # Create two runs
        for i in range(2):
            tracker = ExperimentTracker(
                experiment_config,
                tracking_uri=local_tracking_uri,
            )
            with tracker.start_run(run_name=f"run-{i}"):
                tracker.log_epoch_metrics(
                    {"val_dice": 0.8 + i * 0.05}, step=1
                )

        analytics = RunAnalytics(tracking_uri=local_tracking_uri)
        df = analytics.load_experiment_runs("test-experiment")
        analytics.register_dataframe("runs", df)
        result = analytics.query(
            "SELECT COUNT(*) AS n FROM runs"
        )
        assert result.iloc[0]["n"] == 2
        analytics.close()

    def test_analytics_top_models(
        self,
        experiment_config: ExperimentConfig,
        local_tracking_uri: str,
    ) -> None:
        """top_models should return N best runs by metric."""
        from minivess.observability.analytics import RunAnalytics
        from minivess.observability.tracking import ExperimentTracker

        for i in range(3):
            tracker = ExperimentTracker(
                experiment_config,
                tracking_uri=local_tracking_uri,
            )
            with tracker.start_run(run_name=f"run-{i}"):
                tracker.log_epoch_metrics(
                    {"val_dice": 0.7 + i * 0.1}, step=1
                )

        analytics = RunAnalytics(tracking_uri=local_tracking_uri)
        df = analytics.load_experiment_runs("test-experiment")
        top = analytics.top_models(df, metric="metric_val_dice", n=2)
        assert len(top) == 2
        # Best should be first
        assert float(top.iloc[0]["metric_val_dice"]) >= float(
            top.iloc[1]["metric_val_dice"]
        )
        analytics.close()


# ---------------------------------------------------------------------------
# T5: Integration — train → track → analyze
# ---------------------------------------------------------------------------


class TestTrainTrackAnalyzeRoundtrip:
    """End-to-end: create model → fit with tracker → analyze with DuckDB."""

    def test_full_roundtrip(
        self,
        experiment_config: ExperimentConfig,
        local_tracking_uri: str,
        synthetic_loader: list[dict[str, torch.Tensor]],
    ) -> None:
        """Train → log to MLflow → query via DuckDB analytics."""
        from minivess.observability.analytics import RunAnalytics
        from minivess.observability.tracking import ExperimentTracker

        # Train with tracker
        tracker = ExperimentTracker(
            experiment_config,
            tracking_uri=local_tracking_uri,
        )
        model = SegResNetAdapter(experiment_config.model)
        trainer = SegmentationTrainer(
            model,
            experiment_config.training,
            device="cpu",
            tracker=tracker,
        )

        with tracker.start_run() as _run_id:
            summary = trainer.fit(synthetic_loader, synthetic_loader)

        assert summary["final_epoch"] == 2

        # Analyze
        analytics = RunAnalytics(tracking_uri=local_tracking_uri)
        df = analytics.load_experiment_runs("test-experiment")
        assert len(df) == 1
        assert "metric_train_loss" in df.columns
        assert "metric_val_loss" in df.columns

        # SQL query
        analytics.register_dataframe("runs", df)
        result = analytics.query(
            "SELECT metric_train_loss, metric_val_loss FROM runs"
        )
        assert len(result) == 1
        assert result.iloc[0]["metric_train_loss"] > 0
        analytics.close()
