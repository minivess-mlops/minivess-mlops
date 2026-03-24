"""Post-training Prefect flow (Flow 2.5).

Sits between train(2) and analyze(3) in the pipeline. Orchestrates
post-hoc plugins (checkpoint averaging, subsampled ensemble, merging,
calibration, conformal) based on config. Each plugin is best-effort:
failure does not block other plugins or the downstream analyze flow.

Task DAG:
    Weight-based plugins (parallel, no calibration data):
        - checkpoint averaging, subsampled ensemble, model merging
    Data-dependent plugins (parallel, need calibration data):
        - Calibration, CRC conformal, ConSeCo FP control
    Aggregate results → log summary
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prefect import flow, get_run_logger, task

from minivess.config.post_training_config import PostTrainingConfig
from minivess.observability.lineage import LineageEmitter, emit_flow_lineage
from minivess.observability.tracking import resolve_tracking_uri
from minivess.orchestration.constants import (
    EXPERIMENT_TRAINING,
    FLOW_NAME_POST_TRAINING,
    FLOW_NAME_TRAIN,
    resolve_experiment_name,
)
from minivess.orchestration.mlflow_helpers import (
    find_upstream_safely,
    log_completion_safe,
)
from minivess.pipeline.post_training_plugin import (
    PluginInput,
    PluginRegistry,
    PostTrainingPlugin,
)

logger = logging.getLogger(__name__)


def resolve_checkpoint_paths_from_contract(
    *,
    parent_run_id: str | None,
    tracking_uri: str,
) -> list[Path]:
    """Discover fold checkpoint files from an upstream training run via FlowContract.

    Reads ``checkpoint_dir_fold_N`` tags from *parent_run_id* and resolves them
    to individual checkpoint files. Prefers ``best.ckpt`` over ``epoch_*.ckpt``;
    when no ``best.ckpt`` exists, picks the lexicographically latest
    ``epoch_*.ckpt``. Silently skips dirs that don't exist on the volume.

    Parameters
    ----------
    parent_run_id:
        MLflow run ID of the upstream training run. Returns ``[]`` when None.
    tracking_uri:
        MLflow tracking URI used to construct FlowContract.

    Returns
    -------
    list[Path]
        One Path per fold checkpoint found. Empty when none discovered.
    """
    if parent_run_id is None:
        return []

    from minivess.orchestration.constants import CHECKPOINT_BEST_FILENAME
    from minivess.orchestration.flow_contract import FlowContract

    fc = FlowContract(tracking_uri=tracking_uri)
    fold_infos = fc.find_fold_checkpoints(parent_run_id=parent_run_id)

    checkpoint_paths: list[Path] = []
    for info in fold_infos:
        ckpt_dir: Path = info["checkpoint_dir"]
        if not ckpt_dir.exists():
            logger.warning(
                "Checkpoint dir for fold %d not found on volume: %s",
                info["fold_id"],
                ckpt_dir,
            )
            continue
        # Use canonical checkpoint filename from constants.py (cross-flow contract)
        best = ckpt_dir / CHECKPOINT_BEST_FILENAME
        if best.exists():
            checkpoint_paths.append(best)
            continue
        # Fall back to last.pth or lexicographically latest epoch_*.{ckpt,pth}
        last = ckpt_dir / "last.pth"
        if last.exists():
            checkpoint_paths.append(last)
            continue
        epoch_ckpts = sorted(
            list(ckpt_dir.glob("epoch_*.ckpt")) + list(ckpt_dir.glob("epoch_*.pth"))
        )
        if epoch_ckpts:
            checkpoint_paths.append(epoch_ckpts[-1])
        else:
            logger.warning(
                "No checkpoint files found in %s for fold %d", ckpt_dir, info["fold_id"]
            )

    return checkpoint_paths


def _require_docker_context() -> None:
    """Require Docker container context or MINIVESS_ALLOW_HOST=1."""
    if os.environ.get("MINIVESS_ALLOW_HOST") == "1":
        return
    if os.environ.get("DOCKER_CONTAINER"):
        return
    if Path("/.dockerenv").exists():
        return
    raise RuntimeError(
        "Post-training flow must run inside a Docker container.\n"
        "Run: docker compose -f deployment/docker-compose.flows.yml run post_training\n"
        "Escape hatch for tests: MINIVESS_ALLOW_HOST=1"
    )


# Weight-based plugins (no calibration data needed)
_WEIGHT_PLUGINS = {"checkpoint_averaging", "subsampled_ensemble", "model_merging"}

# Data-dependent plugins (need calibration data)
_DATA_PLUGINS = {"swag", "calibration", "crc_conformal", "conseco_fp_control"}


def _build_registry() -> PluginRegistry:
    """Build plugin registry with all available plugins."""
    registry = PluginRegistry()

    from minivess.pipeline.post_training_plugins.calibration import CalibrationPlugin
    from minivess.pipeline.post_training_plugins.checkpoint_averaging import (
        CheckpointAveragingPlugin,
    )
    from minivess.pipeline.post_training_plugins.conseco_fp_control import ConSeCoPlugin
    from minivess.pipeline.post_training_plugins.crc_conformal import CRCConformalPlugin
    from minivess.pipeline.post_training_plugins.model_merging import ModelMergingPlugin
    from minivess.pipeline.post_training_plugins.subsampled_ensemble import (
        SubsampledEnsemblePlugin,
    )
    from minivess.pipeline.post_training_plugins.swag import SWAGPlugin

    for plugin_cls in (
        CheckpointAveragingPlugin,
        SubsampledEnsemblePlugin,
        SWAGPlugin,
        ModelMergingPlugin,
        CalibrationPlugin,
        CRCConformalPlugin,
        ConSeCoPlugin,
    ):
        registry.register(plugin_cls())

    return registry


def _plugin_config_dict(config: PostTrainingConfig, plugin_name: str) -> dict[str, Any]:
    """Extract plugin-specific config as a dict."""
    sub_config = getattr(config, plugin_name, None)
    if sub_config is None:
        return {}
    return dict(sub_config.model_dump())


@task(name="run-post-training-plugin")
def run_plugin_task(
    plugin: PostTrainingPlugin,
    plugin_input: PluginInput,
) -> dict[str, Any]:
    """Execute a single post-training plugin (Prefect task)."""
    log = get_run_logger()
    log.info("Running post-training plugin: %s", plugin.name)

    errors = plugin.validate_inputs(plugin_input)
    if errors:
        msg = f"Plugin {plugin.name} validation failed: {errors}"
        log.error(msg)
        raise ValueError(msg)

    output = plugin.execute(plugin_input)
    log.info(
        "Plugin %s complete: %d model(s), %d metric(s)",
        plugin.name,
        len(output.model_paths),
        len(output.metrics),
    )
    return {
        "status": "success",
        "model_paths": [str(p) for p in output.model_paths],
        "metrics": output.metrics,
        "artifacts": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in output.artifacts.items()
            if _is_json_serializable(v)
        },
    }


def _is_json_serializable(v: Any) -> bool:
    """Check if a value is JSON-serializable (crude check)."""
    return isinstance(v, str | int | float | bool | list | dict | type(None))


@dataclass
class PostTrainingFlowResult:
    """Typed result returned by post_training_flow().

    Replaces the plain dict[str, Any] return for type safety and
    consistency with TrainingFlowResult and DeployResult.
    """

    flow_name: str = "post-training-flow"
    status: str = "completed"
    mlflow_run_id: str | None = None
    upstream_training_run_id: str | None = None
    checkpoint_averaging_completed: bool = False
    calibration_completed: bool = False
    conformal_completed: bool = False
    failed_operations: list[str] = field(default_factory=list)


@flow(name=FLOW_NAME_POST_TRAINING)
def post_training_flow(
    *,
    config: PostTrainingConfig | None = None,
    checkpoint_paths: list[Path] | None = None,
    run_metadata: list[dict[str, Any]] | None = None,
    output_dir: Path | None = None,
    calibration_data: dict[str, Any] | None = None,
    trigger_source: str = "manual",
    upstream_training_run_id: str | None = None,
) -> PostTrainingFlowResult:
    """Post-training flow (Flow 2.5) — orchestrate post-hoc plugins.

    Parameters
    ----------
    config:
        Post-training configuration. Uses defaults if None.
    checkpoint_paths:
        Paths to training checkpoint files.
    run_metadata:
        Per-checkpoint metadata (loss_type, fold_id, etc.).
    output_dir:
        Directory for output artifacts.
    calibration_data:
        Optional calibration data for data-dependent plugins.
    trigger_source:
        What triggered this flow.
    """
    _require_docker_context()

    log = get_run_logger()
    log.info("Post-training flow started (trigger: %s)", trigger_source)

    if config is None:
        config = PostTrainingConfig()
    if run_metadata is None:
        run_metadata = []
    if output_dir is None:
        output_dir = Path(
            os.environ.get("POST_TRAINING_OUTPUT_DIR", "/app/outputs/post_training")
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-discover upstream training run and checkpoints (#587).
    tracking_uri = resolve_tracking_uri()
    upstream_exp = os.environ.get(
        "UPSTREAM_EXPERIMENT", resolve_experiment_name(EXPERIMENT_TRAINING)
    )

    if upstream_training_run_id is None:
        upstream = find_upstream_safely(
            tracking_uri=tracking_uri,
            experiment_name=upstream_exp,
            upstream_flow=FLOW_NAME_TRAIN,
        )
        upstream_training_run_id = upstream["run_id"] if upstream else None

    if checkpoint_paths is None or len(checkpoint_paths) == 0:
        checkpoint_paths = resolve_checkpoint_paths_from_contract(
            parent_run_id=upstream_training_run_id,
            tracking_uri=tracking_uri,
        )
        if checkpoint_paths:
            log.info(
                "Auto-discovered %d checkpoint(s) from upstream run %s",
                len(checkpoint_paths),
                upstream_training_run_id,
            )
        else:
            log.warning(
                "No checkpoints found from upstream run %s — plugins may skip",
                upstream_training_run_id,
            )

    enabled = config.enabled_plugin_names()
    if not enabled:
        log.info("No post-training plugins enabled")
        return PostTrainingFlowResult(status="completed")

    registry = _build_registry()
    plugin_results: dict[str, dict[str, Any]] = {}

    # --- Weight-based plugins ---
    weight_enabled = [name for name in enabled if name in _WEIGHT_PLUGINS]
    for plugin_name in weight_enabled:
        plugin_cfg = _plugin_config_dict(config, plugin_name)
        plugin_cfg["output_dir"] = str(output_dir / plugin_name)

        pi = PluginInput(
            checkpoint_paths=list(checkpoint_paths),
            config=plugin_cfg,
            run_metadata=list(run_metadata),
        )

        try:
            plugin = registry.get(plugin_name)
            result = run_plugin_task(plugin, pi)
            plugin_results[plugin_name] = result
        except Exception:
            log.exception("Plugin %s failed", plugin_name)
            log.error(
                "Plugin %s FAILED — re-raising (Rule #25: loud failures)",
                plugin_name,
            )
            raise

    # --- Data-dependent plugins ---
    data_enabled = [name for name in enabled if name in _DATA_PLUGINS]
    for plugin_name in data_enabled:
        plugin_cfg = _plugin_config_dict(config, plugin_name)

        pi = PluginInput(
            checkpoint_paths=list(checkpoint_paths),
            config=plugin_cfg,
            calibration_data=calibration_data,
            run_metadata=list(run_metadata),
        )

        try:
            plugin = registry.get(plugin_name)
            result = run_plugin_task(plugin, pi)
            plugin_results[plugin_name] = result
        except Exception:
            log.exception("Plugin %s failed", plugin_name)
            log.error(
                "Plugin %s FAILED — re-raising (Rule #25: loud failures)",
                plugin_name,
            )
            raise

    n_run = len(plugin_results)
    log.info("Post-training flow complete: %d plugin(s) executed", n_run)

    # Log results to MLflow
    mlflow_run_id: str | None = None
    log_exp = os.environ.get("EXPERIMENT", upstream_exp)

    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(log_exp)
        # MLflow tags must be strings — None causes TypeError in to_proto().
        run_tags = {"flow_name": FLOW_NAME_POST_TRAINING}
        if upstream_training_run_id is not None:
            run_tags["upstream_training_run_id"] = upstream_training_run_id
        with mlflow.start_run(tags=run_tags) as active_run:
            mlflow_run_id = active_run.info.run_id
            mlflow.log_metric("n_plugins_run", float(n_run))
            # Log per-plugin metrics with post_ prefix
            for plugin_name, plugin_result in plugin_results.items():
                for metric_name, metric_value in plugin_result.get(
                    "metrics", {}
                ).items():
                    if isinstance(metric_value, int | float):
                        mlflow.log_metric(
                            f"post_{plugin_name}_{metric_name}", float(metric_value)
                        )

    except Exception:
        log.warning("Failed to log post_training_flow to MLflow", exc_info=True)

    # Log flow completion (best-effort, non-blocking)
    log_completion_safe(
        flow_name="post-training-flow",
        tracking_uri=tracking_uri,
        run_id=mlflow_run_id,
    )

    # OpenLineage lineage emission (Issue #799 — IEC 62304 §8 traceability)
    try:
        _emitter = LineageEmitter(namespace="minivess")
        emit_flow_lineage(
            emitter=_emitter,
            job_name="post-training-flow",
            inputs=[{"namespace": "minivess", "name": "checkpoints"}],
            outputs=[{"namespace": "minivess", "name": "averaged_checkpoints"}],
        )
    except Exception:
        logger.warning("OpenLineage emission failed (non-blocking)", exc_info=True)

    avg_ran = any(
        k in plugin_results for k in ("checkpoint_averaging", "subsampled_ensemble")
    )
    calibration_ran = "calibration" in plugin_results
    conformal_ran = any(k in plugin_results for k in ("crc_conformal", "conseco_fp"))
    return PostTrainingFlowResult(
        status="completed",
        mlflow_run_id=mlflow_run_id,
        upstream_training_run_id=upstream_training_run_id,
        checkpoint_averaging_completed=avg_ran
        and plugin_results.get("checkpoint_averaging", {}).get("status") == "success",
        calibration_completed=calibration_ran
        and plugin_results.get("calibration", {}).get("status") == "success",
        conformal_completed=conformal_ran,
    )


def run_factorial_post_training(
    *,
    checkpoint_paths: list[Path],
    methods: list[str],
    output_dir: Path,
    n_subsampled_ensemble_models: int = 3,
    subsample_fraction: float = 0.7,
    seed: int = 42,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
    upstream_run_id: str | None = None,
    upstream_tags: dict[str, str] | None = None,
    recalibration: str = "none",
    calibration_data: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Systematically apply post-training methods to checkpoints.

    For each method in ``methods``, produces one output checkpoint
    (or ensemble) and returns metadata for MLflow tagging. When
    ``tracking_uri`` is provided, creates one MLflow run per method
    with inherited upstream tags (Issue #885 — per-method factorial
    discovery for Biostatistics).

    Parameters
    ----------
    checkpoint_paths:
        List of training checkpoint file paths.
    methods:
        Factorial methods to apply: ``["none", "checkpoint_averaging", "subsampled_ensemble"]``.
    output_dir:
        Directory for output checkpoint files.
    n_subsampled_ensemble_models:
        Number of independent averaged models for subsampled_ensemble.
    subsample_fraction:
        Fraction of checkpoints per subsampled_ensemble model.
    seed:
        Random seed for subsampled_ensemble subsampling.
    tracking_uri:
        MLflow tracking URI. When None, no MLflow runs are created.
    experiment_name:
        MLflow experiment name (e.g. ``"minivess_training"``).
    upstream_run_id:
        MLflow run ID of the upstream training run for tag inheritance.
    upstream_tags:
        Tags from the upstream training run to inherit (model_family,
        loss_function, fold_id, with_aux_calib).

    Returns
    -------
    List of result dicts, one per method. Each dict contains:
        method, output_path, post_training_method, checkpoint_size_mb,
        mlflow_run_id (None when tracking_uri is not provided).
    """
    import shutil

    results: list[dict[str, Any]] = []

    for method in methods:
        method_dir = output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)

        if method == "none":
            # Identity: copy the first checkpoint unchanged
            src = checkpoint_paths[0]
            dst = method_dir / f"{src.stem}_none.pt"
            shutil.copy2(src, dst)
            result = _factorial_result(method, dst)

        elif method == "checkpoint_averaging":
            dst = method_dir / "checkpoint_averaged.pt"
            _average_checkpoints(checkpoint_paths, dst)
            result = _factorial_result(method, dst)

        elif method == "subsampled_ensemble":
            import random

            rng = random.Random(seed)
            # Subsampled ensemble: subsample + average multiple times
            n_ckpts = max(1, int(len(checkpoint_paths) * subsample_fraction))
            ensemble_paths: list[Path] = []
            for i in range(n_subsampled_ensemble_models):
                subset = rng.sample(
                    list(checkpoint_paths), min(n_ckpts, len(checkpoint_paths))
                )
                dst = method_dir / f"subsampled_ensemble_{i}.pt"
                _average_checkpoints(subset, dst)
                ensemble_paths.append(dst)
            # Return the first model path as representative
            result = _factorial_result(method, ensemble_paths[0])

        elif method == "swag":
            if calibration_data is None:
                raise ValueError(
                    "SWAG requires calibration_data with 'train_loader' and 'model'. "
                    "Build DataLoaders from splits before calling "
                    "run_factorial_post_training(methods=['swag'], calibration_data=...)."
                )
            from minivess.pipeline.post_training_plugin import PluginInput
            from minivess.pipeline.post_training_plugins.swag import SWAGPlugin

            swag_plugin = SWAGPlugin()
            plugin_input = PluginInput(
                checkpoint_paths=checkpoint_paths,
                config={
                    "swa_lr": 0.01,
                    "swa_epochs": 2,
                    "max_rank": 20,
                    "update_bn": True,
                    "output_dir": str(method_dir),
                },
                calibration_data=calibration_data,
            )
            validation_errors = swag_plugin.validate_inputs(plugin_input)
            if validation_errors:
                raise ValueError(f"SWAG validation failed: {validation_errors}")
            swag_output = swag_plugin.execute(plugin_input)
            dst = (
                swag_output.model_paths[0]
                if swag_output.model_paths
                else method_dir / "swag_model.pt"
            )
            result = _factorial_result(method, dst)

        else:
            logger.warning("Unknown factorial method: %s — skipping", method)
            continue

        # Create per-method MLflow run (Issue #885)
        mlflow_run_id = _create_factorial_mlflow_run(
            method=method,
            result=result,
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            upstream_run_id=upstream_run_id,
            upstream_tags=upstream_tags,
            recalibration=recalibration,
        )
        result["mlflow_run_id"] = mlflow_run_id
        results.append(result)

    return results


def _create_factorial_mlflow_run(
    *,
    method: str,
    result: dict[str, Any],
    tracking_uri: str | None,
    experiment_name: str | None,
    upstream_run_id: str | None,
    upstream_tags: dict[str, str] | None,
    recalibration: str = "none",
) -> str | None:
    """Create an MLflow run for a single factorial post-training variant.

    Returns the MLflow run_id, or None if tracking_uri is not provided.

    Run naming convention: ``{model}__{loss}__{fold}__{method}``
    Tag schema (synthesis Part 2.4):
        - flow_name: "post-training-flow"
        - post_training_method: {method}
        - upstream_training_run_id: {upstream_run_id}
        - Inherited: model_family, loss_function, fold_id, with_aux_calib
    """
    if tracking_uri is None:
        return None

    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        if experiment_name:
            mlflow.set_experiment(experiment_name)

        # Build tags: inherit upstream + add post-training-specific
        tags: dict[str, str] = {}
        if upstream_tags:
            # Inherit factorial-relevant tags from upstream training run
            for key in (
                "model_family",
                "loss_function",
                "fold_id",
                "with_aux_calib",
            ):
                if key in upstream_tags:
                    tags[key] = upstream_tags[key]

        tags["flow_name"] = FLOW_NAME_POST_TRAINING
        tags["post_training_method"] = method
        tags["recalibration"] = recalibration
        if upstream_run_id is not None:
            tags["upstream_training_run_id"] = upstream_run_id

        # Build run_name: {model}__{loss}__{fold}__{method}
        model = tags.get("model_family", "unknown")
        loss = tags.get("loss_function", "unknown")
        fold = tags.get("fold_id", "x")
        run_name = f"{model}__{loss}__fold{fold}__{method}"

        with mlflow.start_run(run_name=run_name, tags=tags) as active_run:
            # Log checkpoint size as metric
            checkpoint_size = result.get("checkpoint_size_mb", 0.0)
            mlflow.log_metric("checkpoint_size_mb", float(checkpoint_size))
            run_id: str = active_run.info.run_id
            return run_id

    except Exception:
        logger.warning(
            "Failed to create MLflow run for factorial method %s",
            method,
            exc_info=True,
        )
        return None


def _factorial_result(method: str, output_path: Path) -> dict[str, Any]:
    """Build result dict for a factorial post-training variant."""
    size_mb = (
        output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0.0
    )
    return {
        "method": method,
        "output_path": str(output_path),
        "post_training_method": method,
        "checkpoint_size_mb": round(size_mb, 2),
        "mlflow_run_id": None,
    }


def _average_checkpoints(checkpoint_paths: list[Path], output_path: Path) -> None:
    """Average model state_dicts from multiple checkpoint files.

    Parameters
    ----------
    checkpoint_paths:
        Paths to checkpoint files (each must contain ``model_state_dict``).
    output_path:
        Path to write the averaged checkpoint.
    """
    import torch

    if not checkpoint_paths:
        return

    # Load first checkpoint as base
    avg_state = torch.load(checkpoint_paths[0], weights_only=True)
    state_dict = avg_state.get("model_state_dict", avg_state)

    # Accumulate remaining checkpoints
    for ckpt_path in checkpoint_paths[1:]:
        loaded = torch.load(ckpt_path, weights_only=True)
        other_state = loaded.get("model_state_dict", loaded)
        for key in state_dict:
            if key in other_state:
                state_dict[key] = state_dict[key] + other_state[key]

    # Average
    n = len(checkpoint_paths)
    if n > 1:
        for key in state_dict:
            state_dict[key] = state_dict[key] / n

    # Save
    if "model_state_dict" in avg_state:
        avg_state["model_state_dict"] = state_dict
    else:
        avg_state = state_dict
    torch.save(avg_state, output_path)


if __name__ == "__main__":
    # Docker entry point.  UPSTREAM_EXPERIMENT and EXPERIMENT are read inside
    # the flow function body — no need to pass them as arguments here.
    post_training_flow()
