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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

# Required fields that must be present in the experiment YAML
REQUIRED_FIELDS: list[str] = ["experiment_name", "model", "losses"]

# Attempt to import optional adaptive profile dependencies at module level
# so that unittest.mock.patch("run_experiment.detect_hardware") works correctly.
# These modules may not exist; graceful degradation is handled at call sites.
try:
    from minivess.config.adaptive_profiles import (  # type: ignore[import]
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

    def compute_adaptive_profile(
        budget: Any, dataset_profile: Any, model_name: str = "dynunet"
    ) -> Any:  # type: ignore[misc]
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

    for field in REQUIRED_FIELDS:
        if field not in config:
            raise ValueError(
                f"Missing required field '{field}' in experiment config: {yaml_path}"
            )

    return config


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
        from minivess.pipeline.preflight import run_preflight  # type: ignore[import]

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

    # Resolve compute profile
    data_dir = Path(config.get("data_dir", "data/raw"))
    compute = config.get("compute", "cpu")

    # Attempt to load dataset profile for adaptive compute
    dataset_profile = None
    try:
        from minivess.data.profiler import scan_dataset  # type: ignore[import]

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

    # Delegate to train.py for each loss
    losses: list[str] = config.get("losses", ["dice_ce"])
    for loss in losses:
        logger.info("Launching training run: loss=%s", loss)
        train_argv = [
            "--compute",
            profile["name"] if compute != "auto" else "cpu",
            "--loss",
            loss,
            "--data-dir",
            str(data_dir),
            "--num-folds",
            str(config.get("num_folds", 3)),
            "--seed",
            str(config.get("seed", 42)),
            "--experiment-name",
            config.get("experiment_name", "experiment"),
        ]
        if config.get("max_epochs") is not None:
            train_argv += ["--max-epochs", str(config["max_epochs"])]
        if config.get("debug", False):
            train_argv.append("--debug")

        # Import and run the existing train script
        train_script = PROJECT_ROOT / "scripts" / "train.py"
        if train_script.exists():
            import importlib.util

            spec = importlib.util.spec_from_file_location("train", train_script)
            if spec is not None and spec.loader is not None:
                train_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(train_mod)  # type: ignore[union-attr]
                parsed = train_mod.parse_args(train_argv)
                train_mod.run_monitored_experiment(parsed)
        else:
            logger.warning("train.py not found at %s, skipping execution", train_script)


if __name__ == "__main__":
    main()
