#!/usr/bin/env python3
"""Clean up incomplete MLflow training runs.

Scans an MLflow experiment and moves incomplete runs (those that did not
finish all folds x epochs) to a .trash/ directory under the mlruns root.
Trash-based deletion is reversible — runs can be restored by moving them
back to their original location.

Usage:
    # Dry run (default) — show what would be cleaned up:
    uv run python scripts/cleanup_mlflow_runs.py --experiment-id 843896622863223169

    # Actually move incomplete runs to trash:
    uv run python scripts/cleanup_mlflow_runs.py --experiment-id 843896622863223169 --execute

    # Custom expected entries (e.g. 2 folds x 50 epochs = 100):
    uv run python scripts/cleanup_mlflow_runs.py --experiment-id 843896622863223169 --expected-entries 100

    # Clean all experiments:
    uv run python scripts/cleanup_mlflow_runs.py --all
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from minivess.pipeline.mlruns_cleanup import cleanup_incomplete_runs

logger = logging.getLogger(__name__)

# Default: our experiments use 3 folds x 100 epochs = 300 entries
DEFAULT_EXPECTED_ENTRIES = 300


def find_mlruns_dir() -> Path:
    """Locate the mlruns directory relative to the repo root."""
    # Try relative to script location
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    mlruns = repo_root / "mlruns"
    if mlruns.is_dir():
        return mlruns

    # Try current working directory
    cwd_mlruns = Path.cwd() / "mlruns"
    if cwd_mlruns.is_dir():
        return cwd_mlruns

    msg = "Could not find mlruns/ directory"
    raise FileNotFoundError(msg)


def discover_experiments(mlruns_dir: Path) -> list[str]:
    """Find all experiment ID directories in mlruns/."""
    experiments = []
    for item in sorted(mlruns_dir.iterdir()):
        if item.is_dir() and item.name not in {".trash", "models"}:
            # MLflow experiment IDs are numeric strings
            try:
                int(item.name)
                experiments.append(item.name)
            except ValueError:
                continue
    return experiments


def main() -> int:
    """Run the cleanup CLI."""
    parser = argparse.ArgumentParser(
        description="Clean up incomplete MLflow training runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--experiment-id",
        help="MLflow experiment ID to clean up",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Clean all experiments in mlruns/",
    )
    parser.add_argument(
        "--expected-entries",
        type=int,
        default=DEFAULT_EXPECTED_ENTRIES,
        help=f"Number of train_loss entries for a complete run (default: {DEFAULT_EXPECTED_ENTRIES})",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually move runs to trash (default: dry run)",
    )
    parser.add_argument(
        "--mlruns-dir",
        type=Path,
        help="Path to mlruns/ directory (auto-detected if omitted)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve mlruns directory
    mlruns_dir = args.mlruns_dir.resolve() if args.mlruns_dir else find_mlruns_dir()

    logger.info("Using mlruns directory: %s", mlruns_dir)

    dry_run = not args.execute
    if dry_run:
        logger.info("DRY RUN mode — no files will be moved. Use --execute to apply.")

    # Determine which experiments to process
    if args.all:
        experiment_ids = discover_experiments(mlruns_dir)
        logger.info("Found %d experiments", len(experiment_ids))
    else:
        experiment_ids = [args.experiment_id]

    total_moved = 0
    for exp_id in experiment_ids:
        result = cleanup_incomplete_runs(
            mlruns_dir,
            exp_id,
            expected_entries=args.expected_entries,
            dry_run=dry_run,
        )
        print(result.summary())
        print()
        total_moved += result.moved

    if dry_run and total_moved == 0:
        # Check if there were any identified
        logger.info("Run with --execute to actually move incomplete runs to trash.")

    if not dry_run:
        logger.info("Total runs moved to trash: %d", total_moved)
        logger.info(
            "To restore, move from %s/.trash/<exp>/<run> back to <exp>/<run>",
            mlruns_dir,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
