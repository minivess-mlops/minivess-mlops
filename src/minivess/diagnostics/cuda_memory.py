"""CUDA memory stats extraction for per-epoch logging.

Extracts peak allocated/reserved memory and allocation retries from
torch.cuda.memory_stats(). Graceful CPU fallback (returns empty dict).

Metrics use prof_cuda_ prefix (RC16) to disambiguate from sys_gpu_*
(nvidia-smi readings) and MLflow system metrics.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_BYTES_TO_MB = 1 / (1024 * 1024)


def extract_memory_stats(
    *,
    device: int = 0,
    alloc_retry_warning_threshold: int = 10,
) -> dict[str, Any]:
    """Extract CUDA memory statistics for the current device.

    Returns dict with prof_cuda_ prefixed metrics, or empty dict on CPU.

    Parameters
    ----------
    device:
        CUDA device index.
    alloc_retry_warning_threshold:
        Log warning if allocation retries exceed this count.
    """
    import torch

    if not torch.cuda.is_available():
        return {}

    try:
        stats = torch.cuda.memory_stats(device=device)
    except RuntimeError:
        logger.warning("Failed to read CUDA memory stats", exc_info=True)
        return {}

    peak_allocated = stats.get("allocated_bytes.all.peak", 0)
    peak_reserved = stats.get("reserved_bytes.all.peak", 0)
    alloc_retries = stats.get("num_alloc_retries", 0)

    result: dict[str, Any] = {
        "prof_cuda_peak_allocated_mb": round(peak_allocated * _BYTES_TO_MB, 1),
        "prof_cuda_peak_reserved_mb": round(peak_reserved * _BYTES_TO_MB, 1),
        "prof_cuda_alloc_retries": alloc_retries,
    }

    if alloc_retries > alloc_retry_warning_threshold:
        logger.warning(
            "CUDA memory fragmentation detected: %d allocation retries "
            "(threshold=%d). Consider reducing batch size or enabling "
            "gradient checkpointing.",
            alloc_retries,
            alloc_retry_warning_threshold,
        )

    return result
