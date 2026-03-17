"""Infrastructure timing parser and cost analysis for cloud GPU runs.

Reads shell-generated key=value timestamp files from SkyPilot setup: blocks,
computes durations and cost analysis, and logs everything to MLflow.

Shell timing format (generated in SkyPilot YAML setup: blocks)::

    setup_start = 1710412800.000
    python_install_start = 1710412800.100
    python_install_end = 1710412835.300
    ...
    setup_end = 1710413130.700

Metric prefix taxonomy (slash-prefix, #790):
    setup/  — infrastructure timing (one-time MLflow params)
    cost/   — cost analysis (MLflow metrics at step=0)

Issue: #683
See: docs/planning/profiler-benchmarking-plan-double-check.md
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow

if TYPE_CHECKING:
    from minivess.observability.tracking import ExperimentTracker

logger = logging.getLogger(__name__)

# Operations to extract from timing files.
# Each must have {op}_start and {op}_end lines in the timing file.
_TIMED_OPERATIONS = (
    "python_install",
    "uv_install",
    "uv_sync",
    "dvc_config",
    "dvc_pull",
    "model_weights",
    "verification",
)

_TIMING_FILENAME = "timing_setup.txt"


def parse_setup_timing(timing_file: Path) -> dict[str, float]:
    """Parse a key=value timestamp file and compute durations.

    Parameters
    ----------
    timing_file:
        Path to the timing file. Each line is ``key=value`` where value
        is a Unix timestamp (seconds since epoch, with fractional part).

    Returns
    -------
    Dictionary mapping operation names to durations in seconds.
    Includes ``setup_total`` for the overall setup duration.
    Returns empty dict if file doesn't exist or is empty.
    """
    if not timing_file.exists():
        return {}

    text = timing_file.read_text(encoding="utf-8").strip()
    if not text:
        return {}

    # Parse key=value lines into timestamps dict
    timestamps: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if key and value:
            try:
                timestamps[key] = float(value)
            except ValueError:
                logger.warning("Ignoring non-numeric timestamp: %s=%s", key, value)

    # Compute durations for known operations
    durations: dict[str, float] = {}
    for op in _TIMED_OPERATIONS:
        start_key = f"{op}_start"
        end_key = f"{op}_end"
        if start_key in timestamps and end_key in timestamps:
            durations[op] = timestamps[end_key] - timestamps[start_key]

    # Overall setup duration
    if "setup_start" in timestamps and "setup_end" in timestamps:
        durations["setup_total"] = timestamps["setup_end"] - timestamps["setup_start"]

    return durations


def compute_cost_analysis(
    setup_seconds: float,
    training_seconds: float,
    epoch_count: int,
    hourly_rate_usd: float,
) -> dict[str, float]:
    """Compute cost analysis from timing data.

    Parameters
    ----------
    setup_seconds:
        Total setup phase wall time in seconds.
    training_seconds:
        Total training phase wall time in seconds.
    epoch_count:
        Number of training epochs completed.
    hourly_rate_usd:
        Instance hourly cost in USD (0.0 for local runs).

    Returns
    -------
    Dictionary with all cost_* values.
    """
    total_seconds = setup_seconds + training_seconds
    total_cost = total_seconds / 3600.0 * hourly_rate_usd
    setup_cost = setup_seconds / 3600.0 * hourly_rate_usd
    training_cost = training_seconds / 3600.0 * hourly_rate_usd

    # Effective GPU rate: what you actually pay per hour of GPU work
    if training_seconds > 0:
        effective_rate = total_cost / (training_seconds / 3600.0)
    else:
        effective_rate = -1.0  # Sentinel for no training

    # Setup fraction of total cost
    if total_seconds > 0:
        setup_fraction = setup_seconds / total_seconds
        gpu_utilization = training_seconds / total_seconds
    else:
        setup_fraction = 0.0
        gpu_utilization = 0.0

    # Amortization: epochs needed for setup < 10% of total cost
    # setup / (setup + N * epoch) < 0.10 => N > 9 * setup / epoch
    if epoch_count > 0 and training_seconds > 0:
        seconds_per_epoch = training_seconds / epoch_count
        # Strict inequality N > x: smallest integer is floor(x) + 1
        epochs_to_amortize = (
            int(math.floor(9.0 * setup_seconds / seconds_per_epoch)) + 1
        )
        # Break-even: effective rate < 2x hourly rate
        # (setup + training) / training < 2 => setup < training => N > setup / epoch
        break_even = int(math.floor(setup_seconds / seconds_per_epoch)) + 1
    else:
        epochs_to_amortize = 0
        break_even = 0

    return {
        "cost/total_wall_seconds": round(total_seconds, 1),
        "cost/total_usd": round(total_cost, 4),
        "cost/setup_usd": round(setup_cost, 4),
        "cost/training_usd": round(training_cost, 4),
        "cost/effective_gpu_rate": round(effective_rate, 4),
        "cost/setup_fraction": round(setup_fraction, 4),
        "cost/gpu_utilization_fraction": round(gpu_utilization, 4),
        "cost/epochs_to_amortize_setup": epochs_to_amortize,
        "cost/break_even_epochs": break_even,
    }


def generate_timing_jsonl(
    setup_durations: dict[str, float],
    training_seconds: float,
    epoch_count: int,
    hourly_rate_usd: float,
) -> str:
    """Generate JSONL content for the timing artifact.

    Parameters
    ----------
    setup_durations:
        Operation name -> duration in seconds (from parse_setup_timing).
    training_seconds:
        Total training time in seconds.
    epoch_count:
        Number of epochs completed.
    hourly_rate_usd:
        Instance hourly cost in USD.

    Returns
    -------
    Multi-line JSONL string, one JSON object per line.
    """
    now_str = datetime.now(UTC).isoformat()
    lines: list[str] = []

    # Setup operation entries
    setup_total = setup_durations.get(
        "setup_total", sum(v for k, v in setup_durations.items() if k != "setup_total")
    )
    for op, duration in sorted(setup_durations.items()):
        if op == "setup_total":
            continue
        record: dict[str, Any] = {
            "phase": "setup",
            "operation": op,
            "duration_seconds": round(duration, 2),
            "timestamp_utc": now_str,
        }
        lines.append(json.dumps(record, separators=(",", ":")))

    # Cost summary as last line
    cost = compute_cost_analysis(
        setup_seconds=setup_total,
        training_seconds=training_seconds,
        epoch_count=epoch_count,
        hourly_rate_usd=hourly_rate_usd,
    )
    summary: dict[str, Any] = {
        "phase": "cost",
        "operation": "cost_summary",
        "timestamp_utc": now_str,
        "total_cost_usd": cost["cost/total_usd"],
        "setup_cost_usd": cost["cost/setup_usd"],
        "training_cost_usd": cost["cost/training_usd"],
        "effective_gpu_rate_usd": cost["cost/effective_gpu_rate"],
        "setup_fraction": cost["cost/setup_fraction"],
        "hourly_rate_usd": hourly_rate_usd,
    }
    lines.append(json.dumps(summary, separators=(",", ":")))

    return "\n".join(lines) + "\n"


def estimate_cost_from_first_epoch(
    epoch_seconds: float,
    max_epochs: int,
    num_folds: int,
    hourly_rate_usd: float,
    setup_minutes: float = 5.0,
) -> dict[str, float]:
    """Extrapolate total job cost from a single measured epoch time.

    Called after epoch 0 completes to predict total cost before committing
    to the full training run. Logged to MLflow as estimated_total_cost.

    Parameters
    ----------
    epoch_seconds:
        Wall time for a single epoch in seconds.
    max_epochs:
        Total epochs planned.
    num_folds:
        Number of cross-validation folds.
    hourly_rate_usd:
        Spot instance hourly cost in USD (0.0 for local runs).
    setup_minutes:
        Estimated setup overhead per job in minutes (Docker pull, DVC, etc.).

    Returns
    -------
    Dict with estimated_total_cost, estimated_total_hours, cost_per_epoch,
    epoch_seconds.
    """
    total_epoch_seconds = epoch_seconds * max_epochs * num_folds
    setup_seconds = setup_minutes * 60.0
    total_seconds = total_epoch_seconds + setup_seconds
    total_hours = total_seconds / 3600.0
    total_cost = total_hours * hourly_rate_usd
    cost_per_epoch = (epoch_seconds / 3600.0) * hourly_rate_usd

    return {
        "est/total_cost": round(total_cost, 4),
        "est/total_hours": round(total_hours, 4),
        "est/cost_per_epoch": round(cost_per_epoch, 4),
        "est/epoch_seconds": epoch_seconds,
    }


def get_hourly_rate_usd() -> float:
    """Read INSTANCE_HOURLY_USD from environment.

    Returns 0.0 if not set (local/Docker runs).
    """
    raw = os.environ.get("INSTANCE_HOURLY_USD", "0.0")
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid INSTANCE_HOURLY_USD=%s, using 0.0", raw)
        return 0.0


def log_infrastructure_timing(
    tracker: ExperimentTracker | None = None,
    *,
    timing_dir: Path | None = None,
) -> None:
    """Read setup timing file and log to MLflow.

    Reads ``timing_setup.txt`` from timing_dir (or cwd), parses it,
    and logs all setup_* values as MLflow params. Also logs the timing
    file as an artifact under ``timing/``.

    No-op if the timing file doesn't exist (e.g., local runs).

    Parameters
    ----------
    tracker:
        ExperimentTracker instance with an active MLflow run.
        If None, uses mlflow.log_artifact directly.
    timing_dir:
        Directory containing timing_setup.txt. Defaults to cwd.
    """
    if timing_dir is None:
        timing_dir = Path.cwd()

    timing_file = timing_dir / _TIMING_FILENAME
    durations = parse_setup_timing(timing_file)

    if not durations:
        return

    # Log each duration as an MLflow param (slash-prefix, #790)
    params = {}
    for op, duration in durations.items():
        param_name = (
            f"setup/{op}_seconds" if op != "setup_total" else "setup/total_seconds"
        )
        params[param_name] = round(duration, 1)
    mlflow.log_params(params)

    # Log the raw timing file as artifact
    if timing_file.exists():
        if tracker is not None:
            tracker.log_artifact(timing_file, artifact_path="timing")
        else:
            mlflow.log_artifact(str(timing_file), artifact_path="timing")


def log_cost_analysis(
    tracker: ExperimentTracker | None = None,
    *,
    setup_seconds: float,
    training_seconds: float,
    epoch_count: int,
    hourly_rate_usd: float | None = None,
) -> None:
    """Compute and log cost analysis to MLflow as metrics at step=0.

    Parameters
    ----------
    tracker:
        ExperimentTracker instance with an active MLflow run.
    setup_seconds:
        Total setup phase duration in seconds.
    training_seconds:
        Total training phase duration in seconds.
    epoch_count:
        Number of training epochs completed.
    hourly_rate_usd:
        Instance hourly cost. If None, reads from INSTANCE_HOURLY_USD env var.
    """
    if hourly_rate_usd is None:
        hourly_rate_usd = get_hourly_rate_usd()

    cost = compute_cost_analysis(
        setup_seconds=setup_seconds,
        training_seconds=training_seconds,
        epoch_count=epoch_count,
        hourly_rate_usd=hourly_rate_usd,
    )

    # Log cost values as metrics at step=0
    for key, value in cost.items():
        mlflow.log_metric(key, value, step=0)
