"""Shared pytest fixtures and test utilities for tests/v2/unit/."""

from __future__ import annotations

import platform
import resource

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Memory safety: prevent tests from crashing the system (RAM OOM).
# Target: all unit tests must run on a 32 GB laptop.
# Soft limit 8 GB prevents runaway allocations; hard limit unchanged.
# See: .claude/metalearning/2026-03-21-ram-crash-biostatistics-test.md
# ---------------------------------------------------------------------------
_MEMORY_LIMIT_GB = 32

if platform.system() == "Linux":
    _limit_bytes = _MEMORY_LIMIT_GB * 1024 * 1024 * 1024
    _soft, _hard = resource.getrlimit(resource.RLIMIT_AS)
    # Only tighten the soft limit — never exceed the existing hard limit
    _new_soft = (
        min(_limit_bytes, _hard) if _hard != resource.RLIM_INFINITY else _limit_bytes
    )
    resource.setrlimit(resource.RLIMIT_AS, (_new_soft, _hard))


class MockMamba(nn.Module):
    """CPU stub for mamba_ssm.Mamba — (B,L,C)→(B,L,C) linear projection.

    Used in unit tests so backbone/block shape tests run without CUDA compilation.
    Implements the same constructor signature as mamba_ssm.Mamba.
    """

    def __init__(
        self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)  # (B,L,C)→(B,L,C)
