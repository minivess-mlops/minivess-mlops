"""Hydra Compose API bridge for experiment configuration.

Provides ``compose_experiment_config()`` which uses Hydra's Compose API
to merge base config + config groups + experiment overrides into a single
flat dict. Falls back to direct YAML loading when Hydra is unavailable.

Usage::

    from minivess.config.compose import compose_experiment_config

    # Default base config (data + training + checkpoint merged):
    cfg = compose_experiment_config()

    # With experiment override:
    cfg = compose_experiment_config(experiment_name="dynunet_losses")

    # With arbitrary overrides:
    cfg = compose_experiment_config(
        overrides=["model=vesselfm", "training.max_epochs=10"]
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"


def _hydra_available() -> bool:
    """Check if hydra-core is installed."""
    try:
        import hydra  # noqa: F401
        from omegaconf import OmegaConf  # noqa: F401

        return True
    except ImportError:
        return False


def _compose_with_hydra(
    config_dir: Path,
    overrides: list[str],
) -> dict[str, Any]:
    """Compose config using Hydra Compose API.

    Parameters
    ----------
    config_dir:
        Absolute path to the configs directory.
    overrides:
        List of Hydra override strings.

    Returns
    -------
    Merged config as a plain dict.
    """
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(
        config_dir=str(config_dir),
        version_base="1.3",
    ):
        cfg = compose(config_name="base", overrides=overrides)

    # Convert OmegaConf → plain dict
    container = OmegaConf.to_container(cfg, resolve=True)
    plain: dict[str, Any] = (
        {str(k): v for k, v in container.items()} if isinstance(container, dict) else {}
    )

    # Flatten config groups: Hydra nests group contents under the group
    # name (e.g., data.data_dir), but run_experiment.py expects flat keys.
    # We lift nested group values to the top level while preserving the
    # nested structure for callers that want it.
    return _flatten_config_groups(plain)


def _compose_with_manual_merge(
    config_dir: Path,
    experiment_name: str | None,
    overrides: list[str],
) -> dict[str, Any]:
    """Manual YAML merge fallback when Hydra is not available.

    Loads base.yaml, resolves defaults list, merges config group files,
    then applies experiment overrides.

    Parameters
    ----------
    config_dir:
        Path to configs directory.
    experiment_name:
        Optional experiment name to load from experiment/ dir.
    overrides:
        List of "key=value" override strings (subset of Hydra syntax).

    Returns
    -------
    Merged config as a plain dict.
    """
    base_path = config_dir / "base.yaml"
    if not base_path.exists():
        msg = f"base.yaml not found at {base_path}"
        raise FileNotFoundError(msg)

    base = _load_yaml(base_path)
    defaults = base.pop("defaults", [])

    # Merge config groups from defaults list
    merged: dict[str, Any] = {}
    for item in defaults:
        if isinstance(item, str):
            if item == "_self_":
                merged.update(base)
            continue
        if isinstance(item, dict):
            for group, choice in item.items():
                group_file = config_dir / group / f"{choice}.yaml"
                if group_file.exists():
                    group_cfg = _load_yaml(group_file)
                    merged.update(group_cfg)

    # If _self_ wasn't in defaults, apply base last
    if "_self_" not in [d for d in defaults if isinstance(d, str)]:
        merged.update(base)

    # Apply experiment override
    if experiment_name:
        exp_file = config_dir / "experiment" / f"{experiment_name}.yaml"
        if exp_file.exists():
            exp_cfg = _load_yaml(exp_file)
            # Remove Hydra-specific keys
            exp_cfg.pop("defaults", None)
            merged.update(exp_cfg)

    # Apply scalar overrides
    for override in overrides:
        if "=" in override:
            key, value = override.split("=", 1)
            # Handle nested keys like "training.max_epochs=10"
            parts = key.split(".")
            if len(parts) == 1:
                merged[key] = _parse_value(value)
            else:
                # Set both the nested path and the flat key
                _set_nested(merged, parts, _parse_value(value))

    return merged


def compose_experiment_config(
    experiment_name: str | None = None,
    overrides: list[str] | None = None,
    config_dir: Path | None = None,
) -> dict[str, Any]:
    """Compose experiment config using Hydra Compose API.

    Falls back to manual YAML merging if Hydra is not available.

    Parameters
    ----------
    experiment_name:
        Name of the experiment to load from ``configs/experiment/<name>.yaml``.
        Passed as ``+experiment=<name>`` override to Hydra.
    overrides:
        Additional Hydra override strings (e.g., ``["model=vesselfm"]``).
    config_dir:
        Path to configs directory. Defaults to ``PROJECT_ROOT / "configs"``.

    Returns
    -------
    Merged configuration as a plain ``dict[str, Any]``.
    """
    if config_dir is None:
        config_dir = DEFAULT_CONFIG_DIR

    if overrides is None:
        overrides = []

    # Build override list
    hydra_overrides = list(overrides)
    if experiment_name:
        hydra_overrides.insert(0, f"+experiment={experiment_name}")

    if _hydra_available():
        try:
            return _compose_with_hydra(config_dir, hydra_overrides)
        except Exception:
            logger.warning(
                "Hydra composition failed, falling back to manual merge",
                exc_info=True,
            )

    return _compose_with_manual_merge(config_dir, experiment_name, overrides)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML file as dict."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        return {}
    return dict(data)


def _parse_value(value: str) -> Any:
    """Parse a string value to its Python type."""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() == "null" or value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


# Config group names whose contents should be lifted to top level.
_FLATTEN_GROUPS = {"data", "model", "training", "checkpoint"}


def _flatten_config_groups(cfg: dict[str, Any]) -> dict[str, Any]:
    """Flatten Hydra config group nesting to produce a flat config.

    Hydra nests config group contents under the group name::

        {
            "data": {"data_dir": "...", "num_folds": 3},
            "training": {"seed": 42},
            "model": {"model": "dynunet"},
        }

    This flattens to::

        {"data_dir": "...", "num_folds": 3, "seed": 42, "model": "dynunet", ...}

    Top-level keys from experiment overrides take precedence over group defaults
    (i.e., group values only fill in keys that don't already exist at the top level).
    If a group value is already a scalar (e.g., experiment set model: "dynunet"),
    it's left as-is.
    """
    result = dict(cfg)

    # Top-level keys that are NOT group names — these come from experiment
    # overrides or base.yaml _self_ and should NOT be clobbered by groups.
    top_level_keys = {k for k in result if k not in _FLATTEN_GROUPS}

    for group_name in _FLATTEN_GROUPS:
        group_val = result.get(group_name)
        if not isinstance(group_val, dict):
            # Already a scalar (e.g., experiment override set model: "dynunet")
            continue
        for key, value in group_val.items():
            if key == group_name:
                # Self-referencing key: model.model = "dynunet" or
                # checkpoint.checkpoint = {tracked_metrics: ...}
                # → lift to top level only if not overridden by experiment
                if key not in top_level_keys:
                    result[key] = value
            elif key not in top_level_keys:
                # Only fill in keys that don't already exist at top level
                result[key] = value
    return result


def _set_nested(d: dict[str, Any], parts: list[str], value: Any) -> None:
    """Set a nested value in a dict given a list of key parts.

    Also sets the leaf key at the top level for flat-config compat.
    """
    # Set the flat key (last part) at top level
    d[parts[-1]] = value

    # Also set the nested path
    current = d
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value
