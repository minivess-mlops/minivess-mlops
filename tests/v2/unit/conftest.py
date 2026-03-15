"""Shared pytest fixtures and test utilities for tests/v2/unit/."""

from __future__ import annotations

import torch
import torch.nn as nn


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
