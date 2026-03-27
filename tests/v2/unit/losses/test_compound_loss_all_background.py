"""Tests for compound losses with all-background labels.

T9 from double-check plan: verify all compound losses survive all-background data
where sub-losses (CAPELoss, SkeletonRecallLoss, clDice) degenerate to zero.
"""

from __future__ import annotations

import torch


def _make_all_background(
    batch: int = 1,
    classes: int = 2,
    spatial: tuple[int, int, int] = (16, 16, 8),
) -> tuple[torch.Tensor, torch.Tensor]:
    """All-background: logits random, labels all-zero."""
    torch.manual_seed(42)
    logits = torch.randn(batch, classes, *spatial, requires_grad=True)
    labels = torch.zeros(batch, 1, *spatial)
    return logits, labels


class TestCompoundLossAllBackground:
    """All compound losses must produce finite output with all-background labels."""

    def test_graph_topology_all_background_finite(self) -> None:
        from minivess.pipeline.loss_functions import GraphTopologyLoss

        logits, labels = _make_all_background()
        loss_fn = GraphTopologyLoss()
        loss = loss_fn(logits, labels)
        assert torch.isfinite(loss), f"GraphTopologyLoss all-bg: {loss.item()}"

    def test_graph_topology_all_background_backward(self) -> None:
        from minivess.pipeline.loss_functions import GraphTopologyLoss

        logits, labels = _make_all_background()
        loss_fn = GraphTopologyLoss()
        loss = loss_fn(logits, labels)
        loss.backward()
        assert logits.grad is not None, "Gradient should exist"

    def test_vessel_compound_all_background_finite(self) -> None:
        from minivess.pipeline.loss_functions import VesselCompoundLoss

        logits, labels = _make_all_background()
        loss_fn = VesselCompoundLoss()
        loss = loss_fn(logits, labels)
        assert torch.isfinite(loss), f"VesselCompoundLoss all-bg: {loss.item()}"

    def test_cbdice_cldice_all_background_finite(self) -> None:
        from minivess.pipeline.loss_functions import CbDiceClDiceLoss

        logits, labels = _make_all_background()
        loss_fn = CbDiceClDiceLoss()
        loss = loss_fn(logits, labels)
        assert torch.isfinite(loss), f"CbDiceClDiceLoss all-bg: {loss.item()}"

    def test_topology_compound_all_background_finite(self) -> None:
        from minivess.pipeline.loss_functions import TopologyCompoundLoss

        logits, labels = _make_all_background()
        loss_fn = TopologyCompoundLoss()
        loss = loss_fn(logits, labels)
        assert torch.isfinite(loss), f"TopologyCompoundLoss all-bg: {loss.item()}"

    def test_bce_dice_cldice_all_background_finite(self) -> None:
        from minivess.pipeline.loss_functions import BceDiceClDiceLoss

        logits, labels = _make_all_background()
        loss_fn = BceDiceClDiceLoss()
        loss = loss_fn(logits, labels)
        assert torch.isfinite(loss), f"BceDiceClDiceLoss all-bg: {loss.item()}"
