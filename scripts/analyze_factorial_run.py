"""Factorial run analysis — preemption rate monitoring and job metrics.

Issue: #962 (TDD Task 3.5)

Reads job dicts (from ``sky jobs queue`` JSON output) and computes
preemption metrics: total jobs, preempted count, preemption rate,
and average recovery time.

Usage::

    # Parse sky jobs queue JSON and compute metrics
    uv run python scripts/analyze_factorial_run.py --jobs-json jobs.json

CLAUDE.md Rule #6:  Use pathlib.Path throughout.
CLAUDE.md Rule #8:  from __future__ import annotations.
CLAUDE.md Rule #16: import re is BANNED — use json.loads(), not regex.
CLAUDE.md Rule #25: Loud failures — raise on empty input.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def compute_preemption_metrics(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute preemption metrics from a list of SkyPilot job dicts.

    Parameters
    ----------
    jobs : list[dict]
        Job dicts from ``sky jobs queue`` JSON output. Each dict must have
        a ``#RECOVERIES`` key (int). Jobs with ``#RECOVERIES > 0`` count
        as preempted.

    Returns
    -------
    dict
        Keys: ``total_jobs`` (int), ``preempted_count`` (int),
        ``preemption_rate`` (float in [0, 1]),
        ``avg_recovery_seconds`` (float).

    Raises
    ------
    ValueError
        If *jobs* is empty (Rule #25: loud failures, never silent discards).
    """
    if not jobs:
        raise ValueError(
            "jobs list is empty — cannot compute preemption metrics. "
            "Provide at least one job dict from sky jobs queue."
        )

    total_jobs = len(jobs)
    preempted_jobs = [j for j in jobs if j.get("#RECOVERIES", 0) > 0]
    preempted_count = len(preempted_jobs)
    preemption_rate = preempted_count / total_jobs

    if preempted_jobs:
        recovery_times = [
            j.get("recovery_seconds", 0.0) for j in preempted_jobs
        ]
        avg_recovery_seconds = statistics.mean(recovery_times)
    else:
        avg_recovery_seconds = 0.0

    return {
        "total_jobs": int(total_jobs),
        "preempted_count": int(preempted_count),
        "preemption_rate": float(preemption_rate),
        "avg_recovery_seconds": float(avg_recovery_seconds),
    }


def main() -> None:
    """CLI entry point — read jobs JSON, compute and print metrics."""
    parser = argparse.ArgumentParser(
        description="Analyze factorial run preemption metrics.",
    )
    parser.add_argument(
        "--jobs-json",
        type=Path,
        required=True,
        help="Path to JSON file with sky jobs queue output (list of job dicts).",
    )
    args = parser.parse_args()

    jobs_path: Path = args.jobs_json
    if not jobs_path.is_file():
        logger.error("Jobs JSON file not found: %s", jobs_path)
        sys.exit(1)

    jobs_data = json.loads(jobs_path.read_text(encoding="utf-8"))
    if not isinstance(jobs_data, list):
        logger.error("Expected a JSON list of job dicts, got %s", type(jobs_data).__name__)
        sys.exit(1)

    timestamp = datetime.now(UTC).isoformat()
    logger.info("Computing preemption metrics at %s", timestamp)

    metrics = compute_preemption_metrics(jobs_data)

    logger.info("Total jobs: %d", metrics["total_jobs"])
    logger.info("Preempted: %d", metrics["preempted_count"])
    logger.info("Preemption rate: %.2f", metrics["preemption_rate"])
    logger.info("Avg recovery time: %.1f s", metrics["avg_recovery_seconds"])

    # Print as JSON for downstream consumption
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
