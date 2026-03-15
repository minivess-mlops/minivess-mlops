"""PyTorch profiler integration for training loop instrumentation.

Provides ``build_profiler_context()`` which wraps the epoch loop in
``SegmentationTrainer.fit()`` with ``torch.profiler.profile``. When
profiling is disabled, returns ``contextlib.nullcontext()`` for zero
overhead.

Design decisions (from execution plan, review round 2):
- RC2: schedule(wait=0, warmup=0, active=N) — no warmup at epoch granularity
- RC3: prof.step() called per-EPOCH, not per-batch
- RC4: ProfilingConfig is standalone — not on TrainingConfig
- RC9: Gzip compression + size gate for Chrome traces
- RC11: ProfilingSummary carries pre-computed data (not the profiler object)
- RC15: total_profiled_epochs counted at runtime via callback counter
"""

from __future__ import annotations

import contextlib
import gzip
import logging
import shutil
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from pathlib import Path

    from minivess.config.models import ProfilingConfig

logger = logging.getLogger(__name__)


@dataclass
class ProfilingSummary:
    """Pre-computed profiling results for MLflow logging (RC11).

    Decouples ExperimentTracker from torch.profiler internals — the tracker
    receives this dataclass, not the profiler object.
    """

    trace_paths: list[Path] = field(default_factory=list)
    total_profiled_epochs: int = 0
    validation_profiled: bool = False
    key_averages_text: str = ""
    summary_dict: dict[str, Any] = field(default_factory=dict)


class TraceHandler:
    """on_trace_ready callback for torch.profiler.profile.

    Responsibilities:
    (a) Saves Chrome trace to output_dir/profiling/
    (b) Applies gzip compression if config.compress_traces is True
    (c) Enforces trace_size_limit_mb gate (skip if too large, log warning)
    (d) Increments runtime epoch counter for total_profiled_epochs
    (e) Cleans up uncompressed trace after compression
    """

    def __init__(self, config: ProfilingConfig, output_dir: Path) -> None:
        self.config = config
        self.profiling_dir = output_dir / "profiling"
        self.profiling_dir.mkdir(parents=True, exist_ok=True)
        self.profiled_epoch_count: int = 0
        self.trace_paths: list[Path] = []

    def __call__(self, prof: Any) -> None:
        """Handle a completed profiling step (one epoch)."""
        epoch_idx = self.profiled_epoch_count
        trace_name = f"chrome_trace_epoch_{epoch_idx}"
        raw_path = self.profiling_dir / f"{trace_name}.json"

        # Export Chrome trace to disk
        prof.export_chrome_trace(str(raw_path))

        # Check file size against limit
        if raw_path.exists():
            size_mb = raw_path.stat().st_size / (1024 * 1024)
            if size_mb > self.config.trace_size_limit_mb:
                logger.warning(
                    "Chrome trace epoch %d is %.1f MB (limit: %d MB) — "
                    "skipping upload. Only summary will be logged.",
                    epoch_idx,
                    size_mb,
                    self.config.trace_size_limit_mb,
                )
                raw_path.unlink(missing_ok=True)
                self.profiled_epoch_count += 1
                return

        # Gzip compression
        if self.config.compress_traces and raw_path.exists():
            gz_path = raw_path.with_suffix(".json.gz")
            with open(raw_path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            raw_path.unlink()  # Clean up uncompressed
            self.trace_paths.append(gz_path)
        elif raw_path.exists():
            self.trace_paths.append(raw_path)

        self.profiled_epoch_count += 1


def build_profiler_context(
    config: ProfilingConfig,
    output_dir: Path,
) -> Any:
    """Build a torch.profiler.profile context manager or nullcontext.

    Parameters
    ----------
    config:
        Profiling configuration. When ``enabled=False``, returns
        ``contextlib.nullcontext()`` for zero overhead.
    output_dir:
        Directory for trace files (checkpoint_dir in Docker, tmp_path in tests).

    Returns
    -------
    Context manager — either ``torch.profiler.profile`` or ``nullcontext()``.
    """
    if not config.enabled:
        return contextlib.nullcontext()

    # Map activity strings to torch.profiler.ProfilerActivity enums
    activities = []
    for act in config.activities:
        if act.lower() == "cpu":
            activities.append(torch.profiler.ProfilerActivity.CPU)
        elif act.lower() == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)

    handler = TraceHandler(config=config, output_dir=output_dir)

    # RC2: schedule(wait=0, warmup=0, active=N) — no warmup at epoch level.
    # PyTorch warns about missing warmup, but at epoch granularity warmup
    # is unnecessary (CUDA kernels are JIT-compiled during model instantiation).
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*warmup.*", category=UserWarning)
        profiler_schedule = torch.profiler.schedule(
            wait=0,
            warmup=0,
            active=config.epochs,
        )

    return torch.profiler.profile(
        activities=activities,
        schedule=profiler_schedule,
        on_trace_ready=handler,
        record_shapes=config.record_shapes,
        profile_memory=config.profile_memory,
        with_stack=config.with_stack,
        with_flops=config.with_flops,
    )
