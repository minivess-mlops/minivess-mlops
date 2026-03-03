#!/usr/bin/env python
"""Single-command YAML-driven experiment runner.

Usage::

    # Run experiment from YAML config
    uv run python scripts/run_experiment.py --config configs/experiments/dynunet_losses.yaml

    # Dry run (validate only, no training)
    uv run python scripts/run_experiment.py --config configs/experiments/dynunet_losses.yaml --dry-run

    # Debug mode override
    uv run python scripts/run_experiment.py --config configs/experiments/dynunet_losses.yaml --debug
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

# Attempt to import ModelFamily for early validation.
# Graceful degradation: if minivess package is unavailable, validation is skipped.
try:
    from minivess.config.models import ModelFamily as _ModelFamily

    _HAS_MODEL_FAMILY = True
except (ImportError, ModuleNotFoundError):
    _HAS_MODEL_FAMILY = False
    _ModelFamily = None  # type: ignore[assignment,misc]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

# Required fields — 'experiment_name' always required.
# Legacy configs use 'model' + 'losses'; topology configs use 'conditions'.
_ALWAYS_REQUIRED: list[str] = ["experiment_name"]
# At least one of these groups must be present:
_LOSSES_FIELDS: list[str] = ["model", "losses"]
_CONDITIONS_FIELDS: list[str] = ["conditions"]

# Kept for backward compatibility — referenced by tests that import it
REQUIRED_FIELDS: list[str] = _ALWAYS_REQUIRED

# Attempt to import optional adaptive profile dependencies at module level
# so that unittest.mock.patch("run_experiment.detect_hardware") works correctly.
# These modules may not exist; graceful degradation is handled at call sites.
try:
    from minivess.config.adaptive_profiles import (
        compute_adaptive_profile,
        detect_hardware,
    )

    _HAS_ADAPTIVE_PROFILES = True
except (ImportError, ModuleNotFoundError):
    _HAS_ADAPTIVE_PROFILES = False

    def detect_hardware() -> Any:  # type: ignore[misc]
        """Stub: adaptive_profiles module not available."""
        raise ImportError(
            "minivess.config.adaptive_profiles is not installed. "
            "Cannot run hardware detection."
        )

    def compute_adaptive_profile(  # type: ignore[misc]
        budget: Any, dataset_profile: Any, model_name: str = "dynunet"
    ) -> Any:
        """Stub: adaptive_profiles module not available."""
        raise ImportError(
            "minivess.config.adaptive_profiles is not installed. "
            "Cannot compute adaptive profile."
        )


def load_experiment_config(yaml_path: Path) -> dict[str, Any]:
    """Load and validate experiment YAML config.

    Parameters
    ----------
    yaml_path:
        Path to the YAML experiment config file.

    Returns
    -------
    dict[str, Any]
        Parsed experiment configuration.

    Raises
    ------
    ValueError
        If a required field is missing.
    """
    with open(yaml_path, encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    # Always-required fields
    for field in _ALWAYS_REQUIRED:
        if field not in config:
            raise ValueError(
                f"Missing required field '{field}' in experiment config: {yaml_path}"
            )

    # Must have either losses-based fields OR conditions-based fields
    has_losses = all(f in config for f in _LOSSES_FIELDS)
    has_conditions = all(f in config for f in _CONDITIONS_FIELDS)
    if not has_losses and not has_conditions:
        raise ValueError(
            f"Experiment config must have either {_LOSSES_FIELDS} (losses mode) "
            f"or {_CONDITIONS_FIELDS} (conditions mode): {yaml_path}"
        )

    return config


def detect_experiment_mode(config: dict[str, Any]) -> str:
    """Detect experiment mode from config keys.

    Returns
    -------
    ``"conditions"`` if the config has a ``conditions`` key,
    ``"losses"`` otherwise (legacy mode).
    """
    if "conditions" in config:
        return "conditions"
    return "losses"


def extract_extra_target_keys(condition: dict[str, Any]) -> list[str]:
    """Extract auxiliary GT target keys from a condition's wrappers.

    For multitask wrappers, returns the ``gt_key`` for each auxiliary head.
    These keys are needed by the data pipeline to load precomputed targets.

    Parameters
    ----------
    condition:
        Condition config dict with optional ``wrappers`` key.

    Returns
    -------
    List of target key names (e.g., ``["sdf", "centerline_dist"]``).
    """
    keys: list[str] = []
    for wrapper in condition.get("wrappers", []):
        if wrapper.get("type") == "multitask":
            for head in wrapper.get("auxiliary_heads", []):
                gt_key = head.get("gt_key", head.get("name", ""))
                if gt_key:
                    keys.append(gt_key)
    return keys


def _build_train_argv(
    config: dict[str, Any],
    profile: dict[str, Any],
    loss_name: str,
    experiment_name: str,
) -> list[str]:
    """Build the argv list passed to train_monitored.py (or train.py).

    This is the single canonical place where all CLI flags for the training
    sub-process are assembled.  Both ``_run_losses_mode()`` and
    ``_run_single_condition()`` delegate here so that new flags (e.g.
    ``--model-family``) are automatically propagated to every execution path.

    Parameters
    ----------
    config:
        Full experiment configuration dict.
    profile:
        Resolved compute profile dict (output of :func:`resolve_compute_profile`).
    loss_name:
        Loss function name for this particular training run.
    experiment_name:
        MLflow experiment name for this run.

    Returns
    -------
    list[str]
        Ordered list of CLI argument strings suitable for ``parse_args()``.

    Raises
    ------
    ValueError
        If ``model_family`` in *config* is not a valid :class:`ModelFamily` value.
    """
    compute = config.get("compute", "cpu")
    data_dir = Path(config.get("data_dir", "data/raw"))

    # Resolve splits file path (relative to project root)
    splits_file = config.get("splits_file")
    if splits_file:
        splits_path = PROJECT_ROOT / splits_file
    else:
        _seed = config.get("seed", 42)
        _nfolds = config.get("num_folds", 3)
        splits_path = (
            PROJECT_ROOT / "configs" / "splits" / f"{_nfolds}fold_seed{_seed}.json"
        )

    # Validate and resolve model_family early — fail loudly for typos.
    model_family_raw: str = config.get("model_family") or "dynunet"
    if _HAS_MODEL_FAMILY and _ModelFamily is not None:
        try:
            _ModelFamily(model_family_raw)
        except ValueError as exc:
            valid = [e.value for e in _ModelFamily]  # type: ignore[union-attr]
            raise ValueError(
                f"Invalid model_family {model_family_raw!r}. Valid values: {valid}"
            ) from exc
    model_family_str: str = model_family_raw

    argv: list[str] = [
        "--compute",
        profile["name"] if compute != "auto" else "cpu",
        "--model-family",
        model_family_str,
        "--loss",
        loss_name,
        "--data-dir",
        str(data_dir),
        "--num-folds",
        str(config.get("num_folds", 3)),
        "--seed",
        str(config.get("seed", 42)),
        "--experiment-name",
        experiment_name,
        "--splits-file",
        str(splits_path),
    ]
    if config.get("max_epochs") is not None:
        argv += ["--max-epochs", str(config["max_epochs"])]
    if config.get("debug", False):
        argv.append("--debug")

    return argv


def resolve_compute_profile(
    compute: str,
    model_name: str,
    dataset_profile: Any,
) -> dict[str, Any]:
    """Resolve compute profile — 'auto' triggers adaptive, otherwise static.

    Parameters
    ----------
    compute:
        Compute mode. ``"auto"`` triggers hardware detection; any other value
        is treated as a static profile name (e.g. ``"cpu"``, ``"gpu_low"``).
    model_name:
        Model identifier used when computing adaptive profiles.
    dataset_profile:
        Dataset statistics object (``DatasetProfile``). Required when
        ``compute="auto"``; ignored otherwise.

    Returns
    -------
    dict[str, Any]
        Resolved profile parameters with keys:
        ``name``, ``batch_size``, ``patch_size``, ``num_workers``,
        ``mixed_precision``, ``gradient_accumulation_steps``, ``cache_rate``.
    """
    if compute == "auto" and dataset_profile is not None:
        budget = detect_hardware()
        adaptive = compute_adaptive_profile(
            budget, dataset_profile, model_name=model_name
        )
        return {
            "name": adaptive.name,
            "batch_size": adaptive.batch_size,
            "patch_size": adaptive.patch_size,
            "num_workers": adaptive.num_workers,
            "mixed_precision": adaptive.mixed_precision,
            "gradient_accumulation_steps": adaptive.gradient_accumulation_steps,
            "cache_rate": adaptive.cache_rate,
        }

    # Explicit static profile, or 'auto' with no dataset profile (fall back to cpu)
    from minivess.config.compute_profiles import get_compute_profile

    profile_name = compute if compute != "auto" else "cpu"
    profile = get_compute_profile(profile_name)
    return {
        "name": profile.name,
        "batch_size": profile.batch_size,
        "patch_size": profile.patch_size,
        "num_workers": profile.num_workers,
        "mixed_precision": profile.mixed_precision,
        "gradient_accumulation_steps": profile.gradient_accumulation_steps,
        "cache_rate": 1.0,
    }


def apply_debug_to_config(config: dict[str, Any]) -> dict[str, Any]:
    """Apply debug overrides to experiment config.

    Returns a new dict without mutating the original.

    Parameters
    ----------
    config:
        Experiment configuration dictionary.

    Returns
    -------
    dict[str, Any]
        New configuration dict with debug overrides applied when
        ``config["debug"]`` is truthy.
    """
    config = dict(config)
    if config.get("debug", False):
        config["max_epochs"] = 1
        config["num_folds"] = min(config.get("num_folds", 2), 2)
    return config


def run_dry_run(config: dict[str, Any]) -> dict[str, Any]:
    """Validate experiment config without training.

    Runs preflight checks and compute profile resolution, returning
    a summary dict. Never raises — errors are captured in the result.

    Parameters
    ----------
    config:
        Experiment configuration dictionary.

    Returns
    -------
    dict[str, Any]
        Summary with keys ``"validation"`` and ``"preflight"``.
    """
    results: dict[str, Any] = {"validation": {}, "preflight": None}

    data_dir = Path(config.get("data_dir", "data/raw"))

    # Attempt preflight checks if the module is available
    try:
        from minivess.pipeline.preflight import run_preflight

        preflight = run_preflight(data_dir=data_dir)
        results["preflight"] = {
            "passed": preflight.passed,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                }
                for c in preflight.checks
            ],
        }
    except (ImportError, ModuleNotFoundError):
        results["preflight"] = {
            "passed": None,
            "info": "preflight module not available; skipped",
        }
    except Exception as exc:
        results["preflight"] = {"passed": False, "error": str(exc)}

    # Validate compute profile resolution
    compute = config.get("compute", "cpu")
    try:
        profile_info = resolve_compute_profile(
            compute, config.get("model", "dynunet"), None
        )
        results["validation"]["compute"] = profile_info
    except Exception as exc:
        results["validation"]["compute_error"] = str(exc)

    return results


def _run_single_condition(
    condition: dict[str, Any],
    config: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Run training for a single condition across all folds.

    Parameters
    ----------
    condition:
        Condition config dict with 'name', 'wrappers', 'd2c_enabled' keys.
    config:
        Full experiment config dict.
    profile:
        Resolved compute profile dict.

    Returns
    -------
    dict with 'status' key ('completed' or 'failed') and optional 'error'.
    """
    import importlib.util

    cond_name = condition["name"]
    loss_name = config.get("loss", "cbdice_cldice")

    logger.info("Launching condition: %s (loss=%s)", cond_name, loss_name)

    train_argv = _build_train_argv(
        config,
        profile,
        loss_name,
        f"{config.get('experiment_name', 'experiment')}_{cond_name}",
    )

    try:
        train_script = PROJECT_ROOT / "scripts" / "train_monitored.py"
        if not train_script.exists():
            train_script = PROJECT_ROOT / "scripts" / "train.py"
        if train_script.exists():
            module_name = train_script.stem
            spec = importlib.util.spec_from_file_location(module_name, train_script)
            if spec is not None and spec.loader is not None:
                train_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(train_mod)  # type: ignore[union-attr]
                parsed = train_mod.parse_args(train_argv)
                if "checkpoint" in config:
                    parsed.checkpoint_config = config["checkpoint"]
                if "architecture_params" in config:
                    parsed.architecture_params = config["architecture_params"]
                parsed.split_mode = config.get("split_mode", "file")
                # Pass condition config for model/loss wrapping
                parsed.condition = condition
                # Pass precomputed dir for auxiliary targets
                precomp_dir = Path(config.get("data_dir", "data/raw")) / "precomputed"
                parsed.precomputed_dir = precomp_dir if precomp_dir.exists() else None
                train_mod.run_monitored_experiment(parsed)
        return {"status": "completed"}
    except Exception as exc:
        logger.exception("Condition %s FAILED", cond_name)
        return {"status": "failed", "error": str(exc)}


def run_conditions_mode(config: dict[str, Any]) -> dict[str, Any]:
    """Execute conditions-based experiment (iterate over conditions).

    Parameters
    ----------
    config:
        Experiment config with 'conditions' key.

    Returns
    -------
    dict with 'completed' and 'failed' counts plus per-condition results.
    """
    conditions: list[dict[str, Any]] = config.get("conditions", [])
    compute = config.get("compute", "cpu")

    # Resolve compute profile
    dataset_profile = None
    try:
        from minivess.data.profiler import scan_dataset

        data_dir = Path(config.get("data_dir", "data/raw"))
        dataset_profile = scan_dataset(data_dir)
    except (ImportError, ModuleNotFoundError, Exception):
        pass

    profile = resolve_compute_profile(
        compute, config.get("model_family", "dynunet"), dataset_profile
    )
    logger.info("Resolved compute profile: %s", profile["name"])

    completed = 0
    failed = 0
    results: dict[str, Any] = {}

    for condition in conditions:
        cond_name = condition["name"]
        result = _run_single_condition(condition, config, profile)
        results[cond_name] = result
        if result["status"] == "completed":
            completed += 1
        else:
            failed += 1
            logger.warning(
                "Condition %s failed: %s", cond_name, result.get("error", "")
            )

    logger.info(
        "Conditions experiment complete: %d/%d succeeded",
        completed,
        len(conditions),
    )

    return {"completed": completed, "failed": failed, "results": results}


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="YAML-driven experiment runner for MinIVess MLOps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to experiment YAML config file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and exit without training",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (overrides config)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point.

    Loads the experiment config from a YAML file, applies optional debug
    overrides, resolves the compute profile, runs an optional preflight
    check, and then delegates training to ``scripts/train.py``.

    Parameters
    ----------
    argv:
        CLI argument list. Defaults to ``sys.argv[1:]`` when ``None``.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    # Load and validate config
    config = load_experiment_config(args.config)
    logger.info("Loaded experiment config: %s", config.get("experiment_name"))

    # Apply CLI overrides
    if args.debug:
        config["debug"] = True

    config = apply_debug_to_config(config)

    if args.dry_run:
        results = run_dry_run(config)
        logger.info("Dry run results: %s", results)
        return

    # Dispatch based on experiment mode
    mode = detect_experiment_mode(config)
    logger.info("Experiment mode: %s", mode)

    if mode == "conditions":
        run_conditions_mode(config)
        return

    # Legacy losses mode
    _run_losses_mode(config)


def _run_losses_mode(config: dict[str, Any]) -> None:
    """Run legacy losses-based experiment (iterate over loss functions)."""
    data_dir = Path(config.get("data_dir", "data/raw"))
    compute = config.get("compute", "cpu")

    # Attempt to load dataset profile for adaptive compute
    dataset_profile = None
    try:
        from minivess.data.profiler import scan_dataset

        dataset_profile = scan_dataset(data_dir)
        logger.info("Dataset profile loaded for adaptive compute")
    except (ImportError, ModuleNotFoundError):
        logger.debug("data.profiler not available; skipping dataset scan")
    except Exception as exc:
        logger.warning("Dataset scan failed: %s", exc)

    profile = resolve_compute_profile(
        compute, config.get("model", "dynunet"), dataset_profile
    )
    logger.info("Resolved compute profile: %s", profile["name"])

    # Delegate to train.py for each loss — with per-loss error isolation
    losses: list[str] = config.get("losses", ["cbdice_cldice"])
    failed_losses: dict[str, str] = {}
    completed_losses: list[str] = []

    for loss in losses:
        logger.info("Launching training run: loss=%s", loss)
        train_argv = _build_train_argv(
            config,
            profile,
            loss,
            config.get("experiment_name", "experiment"),
        )

        try:
            # Import and run the memory-safe monitored training script
            train_script = PROJECT_ROOT / "scripts" / "train_monitored.py"
            if not train_script.exists():
                # Fallback to train.py for backwards compatibility
                train_script = PROJECT_ROOT / "scripts" / "train.py"
            if train_script.exists():
                import importlib.util

                module_name = train_script.stem
                spec = importlib.util.spec_from_file_location(module_name, train_script)
                if spec is not None and spec.loader is not None:
                    train_mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(train_mod)  # type: ignore[union-attr]
                    parsed = train_mod.parse_args(train_argv)
                    # Pass checkpoint config from experiment YAML to train_monitored
                    if "checkpoint" in config:
                        parsed.checkpoint_config = config["checkpoint"]
                    # Pass architecture_params from experiment YAML
                    if "architecture_params" in config:
                        parsed.architecture_params = config["architecture_params"]
                    # Pass split_mode from experiment YAML
                    parsed.split_mode = config.get("split_mode", "file")
                    train_mod.run_monitored_experiment(parsed)
            else:
                logger.warning(
                    "train_monitored.py not found at %s, skipping execution",
                    train_script,
                )
            completed_losses.append(loss)
        except Exception:
            logger.exception(
                "Training FAILED for loss=%s — continuing to next loss", loss
            )
            failed_losses[loss] = str(loss)

    # Summary
    logger.info(
        "Experiment complete: %d/%d losses succeeded",
        len(completed_losses),
        len(losses),
    )
    if failed_losses:
        logger.warning("Failed losses: %s", list(failed_losses.keys()))


if __name__ == "__main__":
    main()
