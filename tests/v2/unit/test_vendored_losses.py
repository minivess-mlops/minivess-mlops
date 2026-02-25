from __future__ import annotations

import pytest
import torch

from minivess.pipeline.loss_functions import build_loss_function
from minivess.pipeline.vendored_losses.cbdice import CenterlineBoundaryDiceLoss
from minivess.pipeline.vendored_losses.centerline_ce import CenterlineCrossEntropyLoss
from minivess.pipeline.vendored_losses.coletra import TopoLoss, WarpLoss

# Shared fixtures
B, C, D, H, W = 2, 2, 8, 8, 8


def _make_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.randn(B, C, D, H, W, requires_grad=True)
    labels = torch.randint(0, C, (B, 1, D, H, W))
    return logits, labels


class TestCenterlineBoundaryDiceLoss:
    def test_forward_returns_scalar(self) -> None:
        logits, labels = _make_inputs()
        loss_fn = CenterlineBoundaryDiceLoss()
        loss = loss_fn(logits, labels)
        assert loss.ndim == 0

    def test_gradient_flows(self) -> None:
        logits, labels = _make_inputs()
        loss_fn = CenterlineBoundaryDiceLoss()
        loss = loss_fn(logits, labels)
        loss.backward()
        assert logits.grad is not None

    def test_no_nan(self) -> None:
        logits, labels = _make_inputs()
        loss_fn = CenterlineBoundaryDiceLoss()
        loss = loss_fn(logits, labels)
        assert not torch.isnan(loss)


class TestCenterlineCrossEntropyLoss:
    def test_forward_returns_scalar(self) -> None:
        logits, labels = _make_inputs()
        loss_fn = CenterlineCrossEntropyLoss()
        loss = loss_fn(logits, labels)
        assert loss.ndim == 0

    def test_gradient_flows(self) -> None:
        logits, labels = _make_inputs()
        loss_fn = CenterlineCrossEntropyLoss()
        loss = loss_fn(logits, labels)
        loss.backward()
        assert logits.grad is not None

    def test_no_nan(self) -> None:
        logits, labels = _make_inputs()
        loss_fn = CenterlineCrossEntropyLoss()
        loss = loss_fn(logits, labels)
        assert not torch.isnan(loss)


class TestWarpLoss:
    def test_forward_returns_scalar(self) -> None:
        logits, labels = _make_inputs()
        loss_fn = WarpLoss()
        loss = loss_fn(logits, labels)
        assert loss.ndim == 0

    def test_gradient_flows(self) -> None:
        logits, labels = _make_inputs()
        loss_fn = WarpLoss()
        loss = loss_fn(logits, labels)
        loss.backward()
        assert logits.grad is not None

    def test_no_nan(self) -> None:
        logits, labels = _make_inputs()
        loss_fn = WarpLoss()
        loss = loss_fn(logits, labels)
        assert not torch.isnan(loss)


class TestTopoLoss:
    def test_forward_returns_scalar(self) -> None:
        logits, labels = _make_inputs()
        loss_fn = TopoLoss()
        loss = loss_fn(logits, labels)
        assert loss.ndim == 0

    def test_gradient_flows(self) -> None:
        logits, labels = _make_inputs()
        loss_fn = TopoLoss()
        loss = loss_fn(logits, labels)
        loss.backward()
        assert logits.grad is not None

    def test_no_nan(self) -> None:
        logits, labels = _make_inputs()
        loss_fn = TopoLoss()
        loss = loss_fn(logits, labels)
        assert not torch.isnan(loss)


class TestFactoryKeys:
    """Test that all new vendored loss keys work in build_loss_function()."""

    @pytest.mark.parametrize("loss_name", ["cbdice", "centerline_ce", "warp", "topo"])
    def test_factory_returns_module(self, loss_name: str) -> None:
        loss_fn = build_loss_function(loss_name)
        assert isinstance(loss_fn, torch.nn.Module)

    @pytest.mark.parametrize("loss_name", ["cbdice", "centerline_ce", "warp", "topo"])
    def test_factory_forward_pass(self, loss_name: str) -> None:
        loss_fn = build_loss_function(loss_name)
        logits, labels = _make_inputs()
        loss = loss_fn(logits, labels)
        assert loss.ndim == 0
        assert not torch.isnan(loss)
