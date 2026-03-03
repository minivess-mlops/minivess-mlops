#!/usr/bin/env python
"""Runner script for the three SAM3 debug experiments.

Runs sam3_vanilla_debug, sam3_topolora_debug, and sam3_hybrid_debug configs
sequentially via run_experiment.main().

Usage::

    # Run all three SAM3 debug variants
    uv run python scripts/run_sam3_debug_experiment.py

    # Dry-run: validate configs without training
    uv run python scripts/run_sam3_debug_experiment.py --dry-run

    # Run a single variant
    uv run python scripts/run_sam3_debug_experiment.py --variant vanilla

Each variant runs in an isolated try/except block so a failure in one variant
does not prevent the others from running. A summary is printed at the end.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs" / "experiments"

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

logger = logging.getLogger(__name__)

# Ordered list of SAM3 debug configs to run
SAM3_DEBUG_CONFIGS: list[dict[str, str]] = [
    {
        "name": "sam3_vanilla_debug",
        "config": "sam3_vanilla_debug.yaml",
        "description": "SAM3 V1: Vanilla ViT-32L + dice_ce",
    },
    {
        "name": "sam3_topolora_debug",
        "config": "sam3_topolora_debug.yaml",
        "description": "SAM3 V2: TopoLoRA (r=16) + cbdice_cldice",
    },
    {
        "name": "sam3_hybrid_debug",
        "config": "sam3_hybrid_debug.yaml",
        "description": "SAM3 V3: Hybrid DynUNet+SAM3 + cbdice_cldice",
    },
]

_VARIANT_MAP: dict[str, str] = {
    "vanilla": "sam3_vanilla_debug",
    "topolora": "sam3_topolora_debug",
    "hybrid": "sam3_hybrid_debug",
}


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Run SAM3 debug experiments (all three variants)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configs without training",
    )
    parser.add_argument(
        "--variant",
        choices=list(_VARIANT_MAP.keys()),
        default=None,
        help="Run a single variant instead of all three (vanilla, topolora, hybrid)",
    )
    return parser


def _run_single_variant(
    variant: dict[str, str],
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run a single SAM3 debug variant via run_experiment.main().

    Parameters
    ----------
    variant:
        Dict with 'name', 'config', and 'description' keys.
    dry_run:
        When True, pass --dry-run to validate config without training.

    Returns
    -------
    dict with 'status' ('completed' or 'failed') and optional 'error'.
    """
    config_path = CONFIGS_DIR / variant["config"]
    if not config_path.exists():
        return {
            "status": "failed",
            "error": f"Config not found: {config_path}",
        }

    argv: list[str] = ["--config", str(config_path)]
    if dry_run:
        argv.append("--dry-run")

    logger.info(
        "Running variant '%s': %s (dry_run=%s)",
        variant["name"],
        variant["description"],
        dry_run,
    )

    try:
        from run_experiment import main as experiment_main

        experiment_main(argv)
        return {"status": "completed"}
    except SystemExit as exc:
        # argparse may call sys.exit(0) on --help; treat non-zero as failure
        if exc.code != 0:
            return {"status": "failed", "error": f"SystemExit({exc.code})"}
        return {"status": "completed"}
    except Exception as exc:
        logger.exception("Variant '%s' FAILED", variant["name"])
        return {"status": "failed", "error": str(exc)}


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the SAM3 debug runner.

    Parameters
    ----------
    argv:
        CLI argument list. Defaults to sys.argv[1:] when None.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    # Determine which variants to run
    if args.variant is not None:
        target_name = _VARIANT_MAP[args.variant]
        variants_to_run = [v for v in SAM3_DEBUG_CONFIGS if v["name"] == target_name]
    else:
        variants_to_run = list(SAM3_DEBUG_CONFIGS)

    logger.info(
        "SAM3 debug runner: %d variant(s) to run (dry_run=%s)",
        len(variants_to_run),
        args.dry_run,
    )

    results: dict[str, dict[str, Any]] = {}
    for variant in variants_to_run:
        name = variant["name"]
        result = _run_single_variant(variant, dry_run=args.dry_run)
        results[name] = result
        status = result["status"]
        if status == "completed":
            logger.info("Variant '%s': COMPLETED", name)
        else:
            logger.warning(
                "Variant '%s': FAILED — %s", name, result.get("error", "unknown error")
            )

    # Summary report
    completed = [k for k, v in results.items() if v["status"] == "completed"]
    failed = [k for k, v in results.items() if v["status"] == "failed"]

    logger.info(
        "SAM3 debug run summary: %d/%d succeeded",
        len(completed),
        len(variants_to_run),
    )
    if completed:
        logger.info("Completed: %s", completed)
    if failed:
        logger.warning("Failed: %s", failed)
        for name in failed:
            logger.warning("  %s: %s", name, results[name].get("error", ""))

    if failed and not args.dry_run:
        sys.exit(1)


if __name__ == "__main__":
    main()
