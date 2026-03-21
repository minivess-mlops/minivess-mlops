"""Process-level memory monitoring and guardrails.

Ported from foundation-PLR (streaming_duckdb_export.py:217-284).
Prevents runaway memory allocations that crash the system (62 GB RAM OOM).

Usage::

    from minivess.observability.memory_monitor import MemoryMonitor

    monitor = MemoryMonitor(warning_threshold_gb=4.0, critical_threshold_gb=6.0)

    for i in range(n_iterations):
        # ... heavy computation ...
        if (i + 1) % 50 == 0:
            monitor.enforce()  # GC + log if memory is high

See: .claude/metalearning/2026-03-21-ram-crash-biostatistics-test.md
"""

from __future__ import annotations

import gc
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MemoryMonitor:
    """Monitor process RSS and enforce memory limits.

    Attributes
    ----------
    warning_threshold_gb:
        Log warning when RSS exceeds this (default: 4 GB).
    critical_threshold_gb:
        Force ``gc.collect()`` when RSS exceeds this (default: 6 GB).
    """

    warning_threshold_gb: float = 4.0
    critical_threshold_gb: float = 6.0
    _warned: bool = field(default=False, repr=False)

    def check(self) -> tuple[float, str]:
        """Check current process memory usage.

        Returns
        -------
        (usage_gb, status) where status is ``"ok"``, ``"warning"``,
        ``"critical"``, or ``"unknown"`` (if psutil unavailable).
        """
        try:
            import psutil
        except ImportError:
            return self._check_without_psutil()

        process = psutil.Process(os.getpid())
        mem_gb = process.memory_info().rss / (1024**3)

        if mem_gb >= self.critical_threshold_gb:
            return mem_gb, "critical"
        if mem_gb >= self.warning_threshold_gb:
            return mem_gb, "warning"
        return mem_gb, "ok"

    def _check_without_psutil(self) -> tuple[float, str]:
        """Fallback when psutil is not installed."""
        return 0.0, "unknown"

    def enforce(self) -> None:
        """Check memory and take action if necessary.

        - **warning**: log once
        - **critical**: force ``gc.collect()``, log before/after
        """
        mem_gb, status = self.check()

        if status == "critical":
            logger.warning("CRITICAL: Memory at %.2f GB. Forcing GC...", mem_gb)
            gc.collect()
            mem_after, status_after = self.check()
            if status_after == "critical":
                logger.error(
                    "Memory still at %.2f GB after GC (threshold: %.1f GB)",
                    mem_after,
                    self.critical_threshold_gb,
                )
        elif status == "warning" and not self._warned:
            logger.warning(
                "Memory at %.2f GB (threshold: %.1f GB)",
                mem_gb,
                self.warning_threshold_gb,
            )
            self._warned = True
        elif status == "ok":
            self._warned = False

    def get_usage_gb(self) -> float:
        """Get current memory usage in GB."""
        return self.check()[0]
