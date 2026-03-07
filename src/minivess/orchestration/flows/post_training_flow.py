"""Post-training Prefect flow (Flow 2.5).

Sits between train(2) and analyze(3) in the pipeline. Orchestrates
post-hoc plugins (SWA, Multi-SWA, merging, calibration, conformal)
based on config. Each plugin is best-effort: failure does not block
other plugins or the downstream analyze flow.

Task DAG:
    Weight-based plugins (parallel, no calibration data):
        - SWA, Multi-SWA, model merging
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
from minivess.orchestration.constants import FLOW_NAME_POST_TRAINING
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

# Weight-based plugins (no calibration data needed)
_WEIGHT_PLUGINS = {"swa", "multi_swa", "model_merging"}

# Data-dependent plugins (need calibration data)
_DATA_PLUGINS = {"calibration", "crc_conformal", "conseco_fp_control"}


def _build_registry() -> PluginRegistry:
    """Build plugin registry with all available plugins."""
    registry = PluginRegistry()

    from minivess.pipeline.post_training_plugins.calibration import CalibrationPlugin
    from minivess.pipeline.post_training_plugins.conseco_fp_control import ConSeCoPlugin
    from minivess.pipeline.post_training_plugins.crc_conformal import CRCConformalPlugin
    from minivess.pipeline.post_training_plugins.model_merging import ModelMergingPlugin
    from minivess.pipeline.post_training_plugins.multi_swa import MultiSWAPlugin
    from minivess.pipeline.post_training_plugins.swa import SWAPlugin

    for plugin_cls in (
        SWAPlugin,
        MultiSWAPlugin,
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
        log.warning("Plugin %s validation failed: %s", plugin.name, errors)
        return {
            "status": "validation_failed",
            "errors": errors,
            "model_paths": [],
            "metrics": {},
        }

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
    swa_completed: bool = False
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
    log = get_run_logger()
    log.info("Post-training flow started (trigger: %s)", trigger_source)

    if config is None:
        config = PostTrainingConfig()
    if checkpoint_paths is None:
        checkpoint_paths = []
    if run_metadata is None:
        run_metadata = []
    if output_dir is None:
        output_dir = Path(
            os.environ.get("POST_TRAINING_OUTPUT_DIR", "/app/outputs/post_training")
        )

    output_dir.mkdir(parents=True, exist_ok=True)

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
            plugin_results[plugin_name] = {
                "status": "error",
                "model_paths": [],
                "metrics": {},
            }

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
            plugin_results[plugin_name] = {
                "status": "error",
                "model_paths": [],
                "metrics": {},
            }

    n_run = len(plugin_results)
    log.info("Post-training flow complete: %d plugin(s) executed", n_run)

    # Log results to MLflow
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
    mlflow_run_id: str | None = None

    # Find upstream training run (explicit param takes priority over auto-discovery)
    if upstream_training_run_id is None:
        upstream = find_upstream_safely(
            tracking_uri=tracking_uri,
            experiment_name="minivess_training",
            upstream_flow="train",
        )
        upstream_training_run_id = upstream["run_id"] if upstream else None

    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("minivess_training")
        with mlflow.start_run(
            tags={
                "flow_name": "post-training-flow",
                "upstream_training_run_id": upstream_training_run_id,
            }
        ) as active_run:
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

    swa_ran = any(k in plugin_results for k in ("swa", "multi_swa"))
    calibration_ran = "calibration" in plugin_results
    conformal_ran = any(k in plugin_results for k in ("crc_conformal", "conseco_fp"))
    return PostTrainingFlowResult(
        status="completed",
        mlflow_run_id=mlflow_run_id,
        upstream_training_run_id=upstream_training_run_id,
        swa_completed=swa_ran
        and plugin_results.get("swa", {}).get("status") == "success",
        calibration_completed=calibration_ran
        and plugin_results.get("calibration", {}).get("status") == "success",
        conformal_completed=conformal_ran,
    )


if __name__ == "__main__":
    post_training_flow()
