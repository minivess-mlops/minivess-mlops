from __future__ import annotations

import hashlib
import logging
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import mlflow
from mlflow.tracking import MlflowClient

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from minivess.adapters.base import ModelAdapter
    from minivess.config.models import ExperimentConfig

logger = logging.getLogger(__name__)

_DEFAULT_TRACKING_URI = "http://localhost:5000"


class ExperimentTracker:
    """MLflow experiment tracking wrapper for segmentation training."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        tracking_uri: str = _DEFAULT_TRACKING_URI,
    ) -> None:
        self.config = config
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri=tracking_uri)
        self._run_id: str | None = None

    @contextmanager
    def start_run(
        self,
        *,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> Generator[str, None, None]:
        """Context manager for an MLflow run.

        Yields the run_id for external use.
        """
        mlflow.set_experiment(self.config.experiment_name)
        effective_name = run_name or self.config.run_name

        merged_tags = {
            "model_family": self.config.model.family.value,
            "model_name": self.config.model.name,
            "started_at": datetime.now(UTC).isoformat(),
        }
        if tags:
            merged_tags.update(tags)

        with mlflow.start_run(run_name=effective_name, tags=merged_tags) as run:
            self._run_id = run.info.run_id
            self._log_config()
            yield run.info.run_id
            self._run_id = None

    def _log_config(self) -> None:
        """Log experiment configuration as MLflow params."""
        mlflow.log_params(
            {
                "model_family": self.config.model.family.value,
                "model_name": self.config.model.name,
                "in_channels": self.config.model.in_channels,
                "out_channels": self.config.model.out_channels,
                "batch_size": self.config.training.batch_size,
                "learning_rate": self.config.training.learning_rate,
                "max_epochs": self.config.training.max_epochs,
                "optimizer": self.config.training.optimizer,
                "scheduler": self.config.training.scheduler,
                "seed": self.config.training.seed,
                "num_folds": self.config.training.num_folds,
                "mixed_precision": self.config.training.mixed_precision,
            }
        )

    def log_epoch_metrics(
        self,
        metrics: dict[str, float],
        *,
        step: int,
        prefix: str = "",
    ) -> None:
        """Log metrics for a training/validation epoch."""
        prefixed = {f"{prefix}{k}" if prefix else k: v for k, v in metrics.items()}
        mlflow.log_metrics(prefixed, step=step)

    def log_model_info(self, model: ModelAdapter) -> None:
        """Log model configuration and parameter count."""
        model_config = model.get_config()
        mlflow.log_params({f"model_{k}": str(v) for k, v in model_config.items()})
        mlflow.log_metric("trainable_parameters", model.trainable_parameters())

    def log_artifact(self, local_path: Path, *, artifact_path: str = "") -> None:
        """Log a file as an MLflow artifact."""
        mlflow.log_artifact(str(local_path), artifact_path=artifact_path or None)

    def log_test_set_hash(self, test_paths: list[str]) -> str:
        """Compute and log SHA-256 hash of test set file paths for audit trail."""
        content = "\n".join(sorted(test_paths))
        hash_value = hashlib.sha256(content.encode("utf-8")).hexdigest()
        mlflow.set_tag("test_set_sha256", hash_value)
        mlflow.set_tag("test_set_locked_at", datetime.now(UTC).isoformat())
        logger.info("Test set hash recorded: %s", hash_value[:16])
        return hash_value

    def register_model(
        self,
        model_name: str,
        *,
        stage: str = "Staging",
    ) -> None:
        """Register the current run's model in MLflow Model Registry."""
        if self._run_id is None:
            msg = "Cannot register model outside of a run context"
            raise RuntimeError(msg)
        model_uri = f"runs:/{self._run_id}/model"
        mv = mlflow.register_model(model_uri, model_name)
        logger.info(
            "Registered model %s version %s -> %s",
            model_name,
            mv.version,
            stage,
        )
