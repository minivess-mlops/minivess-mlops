from __future__ import annotations

import hashlib
import logging
import os
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

from minivess.config.defaults import DEFAULT_TRACKING_URI as _DEFAULT_TRACKING_URI

logger = logging.getLogger(__name__)


def resolve_tracking_uri(*, tracking_uri: str | None = None) -> str:
    """Resolve MLflow tracking URI from multiple sources.

    Priority order:
    1. Explicit ``tracking_uri`` parameter (highest)
    2. ``MLFLOW_TRACKING_URI`` environment variable
    3. Default: ``"mlruns"`` (local file backend, no server needed)

    Parameters
    ----------
    tracking_uri:
        Explicit URI to use. If provided, returned as-is.

    Returns
    -------
    Resolved tracking URI string.
    """
    if tracking_uri is not None:
        return tracking_uri
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri
    return _DEFAULT_TRACKING_URI


class ExperimentTracker:
    """MLflow experiment tracking wrapper for segmentation training."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        tracking_uri: str | None = None,
    ) -> None:
        self.config = config
        resolved_uri = resolve_tracking_uri(tracking_uri=tracking_uri)
        mlflow.set_tracking_uri(resolved_uri)
        self.client = MlflowClient(tracking_uri=resolved_uri)
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
        merged_tags = dict(sorted(merged_tags.items()))

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
        prefixed = dict(sorted(
            {f"{prefix}{k}" if prefix else k: v for k, v in metrics.items()}.items()
        ))
        mlflow.log_metrics(prefixed, step=step)

    def log_model_info(self, model: ModelAdapter) -> None:
        """Log model configuration and parameter count."""
        model_config = model.get_config().to_dict()
        mlflow.log_params(dict(sorted(
            {f"model_{k}": str(v) for k, v in model_config.items()}.items()
        )))
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

    def log_git_hash(self) -> str | None:
        """Log current git commit hash as tag and artifact."""
        import subprocess

        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            git_hash = result.stdout.strip()
            mlflow.set_tag("git_commit", git_hash)
            logger.info("Logged git hash: %s", git_hash[:8])
            return git_hash
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Could not determine git hash")
            return None

    def log_frozen_deps(self) -> None:
        """Log frozen dependencies (uv pip freeze) as artifact."""
        import subprocess
        import tempfile

        try:
            result = subprocess.run(
                ["uv", "pip", "freeze"],
                capture_output=True,
                text=True,
                check=True,
            )
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", prefix="frozen_deps_", delete=False, encoding="utf-8"
            ) as f:
                f.write(result.stdout)
                f.flush()
                mlflow.log_artifact(f.name, artifact_path="environment")
            logger.info("Logged frozen dependencies")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Could not freeze dependencies (uv not available)")

    def log_hydra_config(self, config_dict: dict, *, filename: str = "resolved_config.yaml") -> None:
        """Log resolved Hydra config as YAML artifact."""
        import tempfile

        import yaml

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix="config_", delete=False, encoding="utf-8"
        ) as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=True)
            f.flush()
            mlflow.log_artifact(f.name, artifact_path="config")
        logger.info("Logged Hydra config as %s", filename)

    def log_split_file(self, split_path: Path) -> None:
        """Log split JSON file as artifact."""
        if split_path.exists():
            mlflow.log_artifact(str(split_path), artifact_path="splits")
            logger.info("Logged split file: %s", split_path.name)

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
