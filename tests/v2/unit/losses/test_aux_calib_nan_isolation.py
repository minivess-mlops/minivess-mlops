"""Tests for AuxCalibCompoundLoss NaN isolation.

T3 from double-check plan: if seg_loss or aux_calib produces NaN,
the other component should still provide gradient signal.
"""

from __future__ import annotations

import logging

import pytest
import torch
from torch import nn


class _NaNLoss(nn.Module):
    """Stub loss that always returns NaN."""

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return torch.tensor(float("nan"), device=logits.device, requires_grad=False)


class _FiniteLoss(nn.Module):
    """Stub loss that returns a finite differentiable scalar."""

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return logits.mean() * 0.1


def _make_pair() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    logits = torch.randn(1, 2, 8, 8, 8, requires_grad=True)
    labels = torch.randint(0, 2, (1, 1, 8, 8, 8)).float()
    return logits, labels


class TestAuxCalibNaNIsolation:
    """AuxCalibCompoundLoss must isolate NaN from individual components."""

    def test_seg_nan_returns_finite(self, caplog: pytest.LogCaptureFixture) -> None:
        from minivess.pipeline.loss_functions import AuxCalibCompoundLoss

        loss_fn = AuxCalibCompoundLoss(seg_loss=_NaNLoss())
        logits, labels = _make_pair()
        with caplog.at_level(logging.WARNING):
            result = loss_fn(logits, labels)
        assert torch.isfinite(result), f"Expected finite, got {result.item()}"
        assert any("seg_loss" in r.message.lower() or "nan" in r.message.lower() for r in caplog.records), (
            "Should warn about NaN from seg_loss"
        )

    def test_calib_nan_returns_finite(self, caplog: pytest.LogCaptureFixture) -> None:
        from minivess.pipeline.loss_functions import AuxCalibCompoundLoss

        loss_fn = AuxCalibCompoundLoss(seg_loss=_FiniteLoss())
        # Replace aux_calib with NaN loss
        loss_fn.aux_calib = _NaNLoss()
        logits, labels = _make_pair()
        with caplog.at_level(logging.WARNING):
            result = loss_fn(logits, labels)
        assert torch.isfinite(result), f"Expected finite, got {result.item()}"

    def test_both_nan_raises(self) -> None:
        from minivess.pipeline.loss_functions import AuxCalibCompoundLoss

        loss_fn = AuxCalibCompoundLoss(seg_loss=_NaNLoss())
        loss_fn.aux_calib = _NaNLoss()
        logits, labels = _make_pair()
        with pytest.raises(ValueError, match="Both seg_loss and aux_calib"):
            loss_fn(logits, labels)

    def test_normal_case_unchanged(self) -> None:
        from minivess.pipeline.loss_functions import AuxCalibCompoundLoss

        seg_loss = _FiniteLoss()
        loss_fn = AuxCalibCompoundLoss(seg_loss=seg_loss, aux_calib_weight=0.5)
        logits, labels = _make_pair()
        result = loss_fn(logits, labels)
        assert torch.isfinite(result), f"Normal case should be finite, got {result.item()}"
        result.backward()
        assert logits.grad is not None, "Gradient should flow"
