from __future__ import annotations

import contextlib
import hashlib
import logging
import os
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
from mlflow.tracking import MlflowClient

if TYPE_CHECKING:
    from collections.abc import Generator

    from minivess.adapters.base import ModelAdapter
    from minivess.config.models import ExperimentConfig
    from minivess.data.profiler import DatasetProfile
    from minivess.pipeline.evaluation import FoldResult

from minivess.config.defaults import DEFAULT_TRACKING_URI as _DEFAULT_TRACKING_URI
from minivess.serving.model_logger import log_single_model

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
    return str(_DEFAULT_TRACKING_URI)


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

        Yields the run_id for external use.  On exception, sets run status
        to FAILED with error metadata.  All auto-logging calls are wrapped
        in try/except so metadata collection failures never abort training.
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

        with mlflow.start_run(
            run_name=effective_name,
            tags=merged_tags,
            log_system_metrics=True,
        ) as run:
            self._run_id = run.info.run_id
            try:
                self._log_config()
                self._log_system_info_safe()
                self._log_git_hash_safe()
                yield run.info.run_id
            except Exception as exc:
                # Tag the run as failed before re-raising
                try:
                    mlflow.set_tag("error_type", type(exc).__name__)
                    mlflow.end_run(status="FAILED")
                except Exception:
                    logger.warning("Failed to set FAILED status", exc_info=True)
                raise
            finally:
                self._run_id = None

    def _log_config(self) -> None:
        """Log experiment configuration as MLflow params.

        Logs ALL ``TrainingConfig`` fields (not a hardcoded subset).
        Architecture-specific parameters from ``ModelConfig.architecture_params``
        are logged with an ``arch_`` prefix (e.g. ``arch_filters``).
        """
        params: dict[str, str | int | float | bool] = {
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
            # Previously missing TrainingConfig fields:
            "weight_decay": self.config.training.weight_decay,
            "warmup_epochs": self.config.training.warmup_epochs,
            "gradient_clip_val": self.config.training.gradient_clip_val,
            "gradient_checkpointing": self.config.training.gradient_checkpointing,
            "early_stopping_patience": self.config.training.early_stopping_patience,
        }

        # Log architecture-specific params with arch_ prefix
        for key, value in self.config.model.architecture_params.items():
            params[f"arch_{key}"] = str(value)

        mlflow.log_params(params)

    def _log_system_info_safe(self) -> None:
        """Log system info as params. Never raises."""
        try:
            from minivess.observability.system_info import get_all_system_info

            mlflow.log_params(get_all_system_info())
        except Exception:
            logger.warning("Failed to log system info", exc_info=True)

    def _log_git_hash_safe(self) -> None:
        """Log git commit as tag. Never raises."""
        try:
            self.log_git_hash()
        except Exception:
            logger.warning("Failed to log git hash", exc_info=True)

    def log_epoch_metrics(
        self,
        metrics: dict[str, float],
        *,
        step: int,
        prefix: str = "",
    ) -> None:
        """Log metrics for a training/validation epoch."""
        prefixed = dict(
            sorted(
                {f"{prefix}{k}" if prefix else k: v for k, v in metrics.items()}.items()
            )
        )
        mlflow.log_metrics(prefixed, step=step)

    def log_model_info(self, model: ModelAdapter) -> None:
        """Log model configuration and parameter count.

        Skips keys already logged by ``_log_config()`` to avoid
        ``MlflowException`` on duplicate param keys.  Logs
        ``trainable_parameters`` as both param (filterable) and metric
        (visible in UI).
        """
        # Keys already logged by _log_config
        existing_keys = {
            "model_family",
            "model_name",
            "model_in_channels",
            "model_out_channels",
        }

        model_config = model.get_config().to_dict()
        new_params = {
            f"model_{k}": str(v)
            for k, v in model_config.items()
            if f"model_{k}" not in existing_keys
        }
        if new_params:
            mlflow.log_params(dict(sorted(new_params.items())))

        n_params = model.trainable_parameters()
        # Log as param (filterable in dashboard) AND metric (visible in UI)
        with contextlib.suppress(Exception):
            mlflow.log_param("trainable_parameters", n_params)
        mlflow.log_metric("trainable_parameters", n_params)

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

        tmp_path: str | None = None
        try:
            result = subprocess.run(
                ["uv", "pip", "freeze"],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".txt",
                prefix="frozen_deps_",
                delete=False,
                encoding="utf-8",
            ) as f:
                f.write(result.stdout)
                tmp_path = f.name
            mlflow.log_artifact(tmp_path, artifact_path="environment")
            logger.info("Logged frozen dependencies")
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            logger.warning("Could not freeze dependencies (uv not available)")
        finally:
            if tmp_path is not None:
                Path(tmp_path).unlink(missing_ok=True)

    def log_hydra_config(
        self, config_dict: dict[str, object], *, filename: str = "resolved_config.yaml"
    ) -> None:
        """Log resolved Hydra config as YAML artifact."""
        import tempfile

        import yaml

        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".yaml",
                prefix="config_",
                delete=False,
                encoding="utf-8",
            ) as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=True)
                tmp_path = f.name
            mlflow.log_artifact(tmp_path, artifact_path="config")
            logger.info("Logged Hydra config as %s", filename)
        finally:
            if tmp_path is not None:
                Path(tmp_path).unlink(missing_ok=True)

    def log_dataset_profile(self, profile: DatasetProfile) -> None:
        """Log dataset profile as MLflow params and JSON artifact.

        Parameters
        ----------
        profile:
            Aggregated dataset statistics from ``scan_dataset()``.
        """
        import json
        import tempfile

        # Log flat params
        mlflow.log_params(profile.to_mlflow_params())

        # Log full profile as JSON artifact
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                prefix="dataset_profile_",
                delete=False,
                encoding="utf-8",
            ) as f:
                json.dump(profile.to_json_dict(), f, indent=2)
                tmp_path = f.name
            mlflow.log_artifact(tmp_path, artifact_path="dataset")
            logger.info("Logged dataset profile: %d volumes", profile.num_volumes)
        finally:
            if tmp_path is not None:
                Path(tmp_path).unlink(missing_ok=True)

    def log_dynaconf_config(self, settings_path: Path) -> None:
        """Log Dynaconf settings.toml as artifact and extract key params.

        Parameters
        ----------
        settings_path:
            Path to the Dynaconf ``settings.toml`` file.
        """
        if not settings_path.exists():
            logger.warning("Dynaconf settings not found: %s", settings_path)
            return

        # Log file as artifact
        mlflow.log_artifact(str(settings_path), artifact_path="config")

        # Parse TOML and extract key params with cfg_ prefix
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]

        try:
            text = settings_path.read_text(encoding="utf-8")
            config = tomllib.loads(text)
            default = config.get("default", {})

            # Extract selected keys as params
            cfg_keys = [
                "project_name",
                "data_dir",
                "dvc_remote",
                "mlflow_tracking_uri",
            ]
            cfg_params: dict[str, str] = {}
            for key in cfg_keys:
                if key in default:
                    cfg_params[f"cfg_{key}"] = str(default[key])

            if cfg_params:
                mlflow.log_params(cfg_params)
            logger.info("Logged Dynaconf config from %s", settings_path.name)
        except Exception:
            logger.warning("Failed to parse Dynaconf config", exc_info=True)

    def log_dvc_provenance(self, *, project_root: Path | None = None) -> None:
        """Log DVC data versioning metadata as tags and artifacts.

        Following best practices (tags for identifiers, artifacts for
        full provenance snapshots):

        - ``dvc_data_hash``: content-addressed hash from ``.dvc`` file (tag)
        - ``dvc_data_nfiles``: number of tracked files (tag)
        - ``dvc_data_path``: DVC-tracked data path (tag)
        - ``dvc.yaml``, ``dvc.lock``: logged as artifacts when present

        Parameters
        ----------
        project_root:
            Project root directory. Defaults to 3 levels up from this file.
        """
        if project_root is None:
            project_root = Path(__file__).resolve().parents[3]

        # Parse .dvc file for data hash
        dvc_file = project_root / "data" / "minivess.dvc"
        if dvc_file.exists():
            try:
                text = dvc_file.read_text(encoding="utf-8")
                for line in text.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("- md5:") or stripped.startswith("- hash:"):
                        mlflow.set_tag(
                            "dvc_data_hash", stripped.split(":", 1)[1].strip()
                        )
                    elif stripped.startswith("nfiles:"):
                        mlflow.set_tag(
                            "dvc_data_nfiles", stripped.split(":", 1)[1].strip()
                        )
                    elif stripped.startswith("path:"):
                        mlflow.set_tag(
                            "dvc_data_path",
                            f"data/{stripped.split(':', 1)[1].strip()}",
                        )
            except Exception:
                logger.warning("Failed to parse DVC file", exc_info=True)

        # Log dvc.yaml and dvc.lock as artifacts (full provenance)
        dvc_yaml = project_root / "dvc.yaml"
        if dvc_yaml.exists():
            mlflow.log_artifact(str(dvc_yaml), artifact_path="config")

        dvc_lock = project_root / "dvc.lock"
        if dvc_lock.exists():
            mlflow.log_artifact(str(dvc_lock), artifact_path="config")

        # DVC remote URL from .dvc/config
        dvc_config = project_root / ".dvc" / "config"
        if dvc_config.exists():
            try:
                for line in dvc_config.read_text(encoding="utf-8").splitlines():
                    stripped = line.strip()
                    if stripped.startswith("url ="):
                        mlflow.set_tag(
                            "dvc_remote_url", stripped.split("=", 1)[1].strip()
                        )
                        break
            except Exception:
                logger.warning("Failed to parse DVC config", exc_info=True)

        logger.info("Logged DVC provenance metadata")

    def log_split_file(self, split_path: Path) -> None:
        """Log split JSON file as artifact."""
        if split_path.exists():
            mlflow.log_artifact(str(split_path), artifact_path="splits")
            logger.info("Logged split file: %s", split_path.name)

    def log_evaluation_results(
        self,
        fold_result: FoldResult,
        *,
        fold_id: int,
        loss_name: str,
    ) -> None:
        """Log MetricsReloaded evaluation results as MLflow metrics.

        Flattens per-metric CIs into ``eval_fold{fold_id}_{metric}``
        keys for easy comparison across folds and losses.

        Parameters
        ----------
        fold_result:
            Evaluation results with aggregated CIs.
        fold_id:
            Fold index (0-based).
        loss_name:
            Loss function name (for logging context).
        """
        flat_metrics: dict[str, float] = {}
        for metric_name, ci in fold_result.aggregated.items():
            prefix = f"eval_fold{fold_id}_{metric_name}"
            flat_metrics.update(ci.to_dict(prefix))

        mlflow.log_metrics(dict(sorted(flat_metrics.items())))
        logger.info(
            "Logged evaluation results for fold %d, loss=%s: %d metrics",
            fold_id,
            loss_name,
            len(flat_metrics),
        )

    def log_pyfunc_model(
        self,
        checkpoint_path: Path,
        model_config_dict: dict[str, object],
        *,
        artifact_path: str = "model",
    ) -> str:
        """Log a segmentation model as MLflow pyfunc artifact.

        Delegates to :func:`model_logger.log_single_model`.
        Must be called within a :meth:`start_run` context.

        Parameters
        ----------
        checkpoint_path:
            Path to the ``.pth`` checkpoint file.
        model_config_dict:
            Model architecture configuration dict.
        artifact_path:
            MLflow artifact path for the logged model.

        Returns
        -------
        Model URI string (``runs:/{run_id}/{artifact_path}``).

        Raises
        ------
        RuntimeError
            If called outside of a run context.
        """
        if self._run_id is None:
            msg = "Cannot log pyfunc model outside of a run context"
            raise RuntimeError(msg)

        log_single_model(
            checkpoint_path=checkpoint_path,
            model_config_dict=model_config_dict,
            artifact_path=artifact_path,
        )
        model_uri = f"runs:/{self._run_id}/{artifact_path}"
        logger.info("Logged pyfunc model: %s", model_uri)
        return model_uri

    def register_model(
        self,
        model_name: str,
        *,
        alias: str = "challenger",
    ) -> None:
        """Register the current run's model in MLflow Model Registry.

        Uses aliases (not deprecated stages) per MLflow 2.9+.

        Parameters
        ----------
        model_name:
            Name for the registered model.
        alias:
            Alias to assign (e.g. ``"champion"``, ``"challenger"``).
        """
        if self._run_id is None:
            msg = "Cannot register model outside of a run context"
            raise RuntimeError(msg)
        model_uri = f"runs:/{self._run_id}/model"
        mv = mlflow.register_model(model_uri, model_name)
        self.client.set_registered_model_alias(model_name, alias, mv.version)
        logger.info(
            "Registered model %s version %s with alias '%s'",
            model_name,
            mv.version,
            alias,
        )

    def log_post_training_tags(
        self,
        best_metrics: dict[str, float],
        *,
        fold_id: int | None = None,
        loss_type: str | None = None,
    ) -> None:
        """Set post-training tags for downstream selection.

        Parameters
        ----------
        best_metrics:
            Mapping of metric name to best value (e.g. ``{"val_dice": 0.843}``).
        fold_id:
            Fold index (0-based).
        loss_type:
            Loss function name.
        """
        for name, value in best_metrics.items():
            mlflow.set_tag(f"best_{name}", f"{value:.6f}")
        if fold_id is not None:
            mlflow.set_tag("fold_id", str(fold_id))
        if loss_type is not None:
            mlflow.set_tag("loss_type", loss_type)
