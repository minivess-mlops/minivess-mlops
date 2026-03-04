"""Dynamic discovery of all implemented methods for quasi-E2E testing.

This module is the SINGLE ENTRY POINT for discovering what the repo can do.
It queries enums, registries, factories, and capability schemas to produce
a deterministic list of test combinations.

Usage:
    python -m minivess.testing.capability_discovery --check
    python -m minivess.testing.capability_discovery --variant practical \
        --output configs/experiments/generated_combos.yaml
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Default YAML path relative to this file:
#   src/minivess/testing/capability_discovery.py (4 parents -> repo root)
_REPO_ROOT: Path = Path(__file__).resolve().parents[3]
_DEFAULT_CAPABILITY_YAML: Path = _REPO_ROOT / "configs" / "method_capabilities.yaml"


# ---------------------------------------------------------------------------
# Schema model
# ---------------------------------------------------------------------------


class CapabilitySchema(BaseModel):
    """Parsed representation of method_capabilities.yaml."""

    version: str
    implemented_models: list[str]
    loss_exclusions: dict[str, list[str]] = Field(default_factory=dict)
    post_training_exclusions: dict[str, list[str]] = Field(default_factory=dict)
    ensemble_exclusions: dict[str, list[str]] = Field(default_factory=dict)
    deployment_exclusions: dict[str, list[str]] = Field(default_factory=dict)
    model_default_loss: dict[str, str] = Field(default_factory=dict)
    model_extra_losses: dict[str, list[str]] = Field(default_factory=dict)
    not_implemented: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Discovery functions
# ---------------------------------------------------------------------------


def load_capability_schema(
    yaml_path: Path | None = None,
) -> CapabilitySchema:
    """Load configs/method_capabilities.yaml into validated Pydantic model."""
    resolved = yaml_path if yaml_path is not None else _DEFAULT_CAPABILITY_YAML
    if not resolved.exists():
        msg = f"Capability schema not found: {resolved}"
        raise FileNotFoundError(msg)
    with resolved.open(encoding="utf-8") as fh:
        data: Any = yaml.safe_load(fh)
    return CapabilitySchema(**data)


def discover_implemented_models(
    yaml_path: Path | None = None,
) -> list[str]:
    """Return model names that have working build_adapter() implementations.

    Reads from method_capabilities.yaml ``implemented_models`` list.
    """
    schema = load_capability_schema(yaml_path)
    return sorted(schema.implemented_models)


def discover_all_losses() -> list[str]:
    """Return all loss names from build_loss_function() dispatch.

    Introspects the loss classification sets in loss_functions.py.
    """
    from minivess.pipeline.loss_functions import (
        _EXPERIMENTAL_LOSSES,
        _HYBRID_LOSSES,
        _LIBRARY_COMPOUND_LOSSES,
        _LIBRARY_LOSSES,
    )

    all_losses: set[str] = set()
    all_losses.update(_LIBRARY_LOSSES)
    all_losses.update(_LIBRARY_COMPOUND_LOSSES)
    all_losses.update(_HYBRID_LOSSES.keys())
    all_losses.update(_EXPERIMENTAL_LOSSES.keys())
    return sorted(all_losses)


def discover_metrics() -> list[str]:
    """Return all metric names from MetricRegistry YAML."""
    from minivess.pipeline.metric_registry import load_metric_registry

    registry = load_metric_registry()
    return registry.all_names()


def discover_post_training_plugins() -> list[str]:
    """Return all registered plugin names from PluginRegistry."""
    from minivess.config.post_training_config import PostTrainingConfig

    config = PostTrainingConfig()
    # Return all plugin field names from the config
    plugin_names: list[str] = []
    for field_name in config.model_fields:
        if field_name in (
            "mlflow_experiment",
            "calibration_fraction",
        ):
            continue
        sub = getattr(config, field_name, None)
        if hasattr(sub, "enabled"):
            plugin_names.append(field_name)
    return sorted(plugin_names)


def discover_ensemble_strategies(
    yaml_path: Path | None = None,
) -> list[str]:
    """Return all EnsembleStrategy values minus excluded ones."""
    from minivess.config.models import EnsembleStrategy

    schema = load_capability_schema(yaml_path)
    excluded = set()
    for strategy, conditions in schema.ensemble_exclusions.items():
        if "*" in conditions:
            excluded.add(strategy)
    return sorted(s.value for s in EnsembleStrategy if s.value not in excluded)


def discover_deployment_methods(
    yaml_path: Path | None = None,
) -> list[str]:
    """Return enabled deployment methods."""
    schema = load_capability_schema(yaml_path)
    all_methods = ["onnx", "bentoml", "monai_deploy"]
    excluded = set()
    for method, conditions in schema.deployment_exclusions.items():
        if "*" in conditions:
            excluded.add(method)
    return sorted(m for m in all_methods if m not in excluded)


def get_valid_losses_for_model(
    model: str,
    yaml_path: Path | None = None,
) -> list[str]:
    """All losses EXCEPT those in loss_exclusions containing model."""
    schema = load_capability_schema(yaml_path)
    if model not in schema.implemented_models:
        msg = f"Model {model!r} not in implemented_models"
        raise ValueError(msg)
    all_losses = discover_all_losses()
    excluded_for_model: set[str] = set()
    for loss_name, excluded_models in schema.loss_exclusions.items():
        if model in excluded_models or "*" in excluded_models:
            excluded_for_model.add(loss_name)
    return sorted(loss for loss in all_losses if loss not in excluded_for_model)


# ---------------------------------------------------------------------------
# Consistency check (--check mode for pre-commit)
# ---------------------------------------------------------------------------


def check_consistency(yaml_path: Path | None = None) -> list[str]:
    """Validate capability schema consistency. Returns list of errors."""
    from minivess.config.models import ModelFamily

    errors: list[str] = []
    schema = load_capability_schema(yaml_path)

    # 1. Every ModelFamily enum must be accounted for
    all_accounted = set(schema.implemented_models) | set(schema.not_implemented)
    for member in ModelFamily:
        if member.value not in all_accounted:
            errors.append(
                f"ModelFamily.{member.name} ({member.value}) not in "
                f"implemented_models or not_implemented"
            )

    # 2. Default losses must exist
    all_losses = discover_all_losses()
    for model, loss in schema.model_default_loss.items():
        if loss not in all_losses:
            errors.append(f"Default loss {loss!r} for {model!r} not in known losses")

    # 3. Extra losses must exist
    for model, extras in schema.model_extra_losses.items():
        for loss in extras:
            if loss not in all_losses:
                errors.append(f"Extra loss {loss!r} for {model!r} not in known losses")

    # 4. Exclusion keys must reference real methods
    for loss_name in schema.loss_exclusions:
        if loss_name not in all_losses:
            errors.append(f"loss_exclusions key {loss_name!r} not a known loss")

    return errors


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for capability discovery."""
    import argparse

    parser = argparse.ArgumentParser(description="Method capability discovery")
    parser.add_argument("--check", action="store_true", help="Validate consistency")
    parser.add_argument(
        "--yaml", type=Path, default=None, help="Path to capability YAML"
    )
    args = parser.parse_args()

    if args.check:
        errors = check_consistency(args.yaml)
        if errors:
            for err in errors:
                print(f"ERROR: {err}", file=sys.stderr)
            sys.exit(1)
        else:
            print("Capability schema is consistent.")
            sys.exit(0)

    # Default: print summary
    models = discover_implemented_models(args.yaml)
    losses = discover_all_losses()
    metrics = discover_metrics()
    plugins = discover_post_training_plugins()
    strategies = discover_ensemble_strategies(args.yaml)
    deploy = discover_deployment_methods(args.yaml)

    print(f"Models:     {len(models)} — {', '.join(models)}")
    print(f"Losses:     {len(losses)} — {', '.join(losses)}")
    print(f"Metrics:    {len(metrics)} — {', '.join(metrics)}")
    print(f"Plugins:    {len(plugins)} — {', '.join(plugins)}")
    print(f"Ensembles:  {len(strategies)} — {', '.join(strategies)}")
    print(f"Deployment: {len(deploy)} — {', '.join(deploy)}")


if __name__ == "__main__":
    main()
