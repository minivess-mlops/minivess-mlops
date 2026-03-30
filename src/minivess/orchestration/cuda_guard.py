"""CUDA availability guard — fail-fast when GPU is unavailable.

Mirrors docker_guard.py: checks escape hatch → checks CUDA → raises RuntimeError.
Prevents silent CPU fallback that wasted 4 hours on 2026-03-29.

Escape hatch: MINIVESS_ALLOW_CPU=1 for pytest ONLY — never in production.

See: .claude/metalearning/2026-03-29-silent-cpu-fallback-no-observability-4h-wasted.md
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class CudaMismatchResult:
    """Result of CUDA version mismatch detection."""

    mismatch_detected: bool
    pytorch_cuda_version: str
    cuda_available: bool
    message: str


def require_cuda_context(flow_name: str) -> None:
    """Raise RuntimeError if CUDA is not available.

    Called at the TOP of GPU flows (train, hpo, post_training, analysis)
    BEFORE any model creation, data loading, or computation.

    Checks:
    1. MINIVESS_ALLOW_CPU=1 (test escape hatch — pytest only)
    2. torch.cuda.is_available()

    Parameters
    ----------
    flow_name:
        Human-readable flow name for the error message.
    """
    if os.environ.get("MINIVESS_ALLOW_CPU") == "1":
        return

    if torch.cuda.is_available():
        return

    pytorch_cuda = getattr(torch.version, "cuda", "unknown")
    raise RuntimeError(
        f"CUDA not available — {flow_name} flow cannot run on CPU.\n\n"
        f"  PyTorch CUDA version: {pytorch_cuda}\n"
        f"  torch.cuda.is_available(): False\n\n"
        f"This usually means the NVIDIA driver is too old for the PyTorch CUDA\n"
        f"version in this Docker image. Check:\n"
        f"  1. Host driver: nvidia-smi (must support CUDA >= {pytorch_cuda})\n"
        f"  2. Container CUDA: docker exec <container> python -c 'import torch; print(torch.version.cuda)'\n"
        f"  3. CDI device mapping: docker run --device nvidia.com/gpu=all ...\n\n"
        f"Escape hatch (pytest ONLY): export MINIVESS_ALLOW_CPU=1"
    )


def detect_cuda_version_mismatch() -> CudaMismatchResult:
    """Advisory check for CUDA toolkit/driver version mismatch.

    Compares PyTorch's compiled CUDA version against runtime availability.
    Does NOT raise — logs a warning if mismatch detected.

    Returns CudaMismatchResult with diagnostic information.
    """
    pytorch_cuda = getattr(torch.version, "cuda", "unknown")
    cuda_available = torch.cuda.is_available()

    if cuda_available:
        return CudaMismatchResult(
            mismatch_detected=False,
            pytorch_cuda_version=str(pytorch_cuda),
            cuda_available=True,
            message=f"CUDA OK: PyTorch compiled with CUDA {pytorch_cuda}, GPU accessible.",
        )

    msg = (
        f"CUDA MISMATCH: PyTorch compiled with CUDA {pytorch_cuda} but "
        f"torch.cuda.is_available() returns False. "
        f"The host NVIDIA driver likely does not support CUDA {pytorch_cuda}."
    )
    logger.warning(msg)

    return CudaMismatchResult(
        mismatch_detected=True,
        pytorch_cuda_version=str(pytorch_cuda),
        cuda_available=False,
        message=msg,
    )
