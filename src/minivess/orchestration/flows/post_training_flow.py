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
from pathlib import Path
from typing import Any

from minivess.config.post_training_config import PostTrainingConfig
from minivess.orchestration import flow, get_run_logger, task
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


@flow(name="Post-Training Pipeline")
def post_training_flow(
    *,
    config: PostTrainingConfig | None = None,
    checkpoint_paths: list[Path] | None = None,
    run_metadata: list[dict[str, Any]] | None = None,
    output_dir: Path | None = None,
    calibration_data: dict[str, Any] | None = None,
    trigger_source: str = "manual",
) -> dict[str, Any]:
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
        output_dir = Path("outputs/post_training")

    output_dir.mkdir(parents=True, exist_ok=True)

    enabled = config.enabled_plugin_names()
    if not enabled:
        log.info("No post-training plugins enabled")
        return {
            "status": "success",
            "n_plugins_run": 0,
            "plugin_results": {},
            "trigger_source": trigger_source,
        }

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

    return {
        "status": "success",
        "n_plugins_run": n_run,
        "plugin_results": plugin_results,
        "trigger_source": trigger_source,
    }
