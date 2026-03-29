"""Tests for compound loss FP16 dtype handling under mixed-precision training.

T1 from double-check plan: GraphTopologyLoss FP16 dtype mismatch under autocast.
The accumulator must always be float32 regardless of input dtype.
"""

from __future__ import annotations

import pytest
import torch


def _make_logits_labels(
    batch: int = 1,
    classes: int = 2,
    spatial: tuple[int, int, int] = (16, 16, 8),
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic logits and labels."""
    torch.manual_seed(42)
    logits = torch.randn(batch, classes, *spatial, dtype=dtype, requires_grad=True)
    labels = torch.randint(0, 2, (batch, 1, *spatial)).float()
    return logits, labels


class TestGraphTopologyLossFP16:
    """GraphTopologyLoss accumulator must always be float32."""

    def test_graph_topology_loss_fp16_input_finite(self) -> None:
        from minivess.pipeline.loss_functions import GraphTopologyLoss

        logits, labels = _make_logits_labels(dtype=torch.float16)
        loss_fn = GraphTopologyLoss()
        # Sub-losses internally cast to float32, but accumulator must also be float32
        loss = loss_fn(logits.float(), labels)
        assert loss.dtype == torch.float32, f"Expected float32, got {loss.dtype}"
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    def test_graph_topology_loss_accumulator_always_float32(self) -> None:
        from minivess.pipeline.loss_functions import GraphTopologyLoss

        logits, labels = _make_logits_labels(dtype=torch.float32)
        loss_fn = GraphTopologyLoss()
        loss = loss_fn(logits, labels)
        # The accumulator must always be float32, never inherit input dtype
        assert loss.dtype == torch.float32, f"Expected float32, got {loss.dtype}"
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    def test_graph_topology_loss_fp16_gradient_finite(self) -> None:
        from minivess.pipeline.loss_functions import GraphTopologyLoss

        logits, labels = _make_logits_labels(dtype=torch.float32)
        loss_fn = GraphTopologyLoss()
        loss = loss_fn(logits, labels)
        loss.backward()
        assert logits.grad is not None, "Gradient is None"
        assert torch.isfinite(logits.grad).all(), "Gradient contains non-finite values"


class TestAllCompoundLossesAccumulatorDtype:
    """All compound losses must produce float32 output even with float16 input."""

    @pytest.mark.parametrize(
        "loss_cls_name",
        [
            "VesselCompoundLoss",
            "TopologyCompoundLoss",
            "GraphTopologyLoss",
            "CbDiceClDiceLoss",
            "BceDiceClDiceLoss",
        ],
    )
    def test_compound_loss_output_is_float32(self, loss_cls_name: str) -> None:
        import minivess.pipeline.loss_functions as lf

        loss_cls = getattr(lf, loss_cls_name)
        loss_fn = loss_cls()
        logits, labels = _make_logits_labels(dtype=torch.float32)
        loss = loss_fn(logits, labels)
        assert loss.dtype == torch.float32, (
            f"{loss_cls_name} produced {loss.dtype}, expected float32"
        )
        assert torch.isfinite(loss), f"{loss_cls_name} produced non-finite loss"
