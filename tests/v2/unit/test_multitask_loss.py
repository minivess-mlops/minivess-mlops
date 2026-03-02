"""Tests for generic multi-task loss composer (T7 — #232)."""

from __future__ import annotations

import torch

from minivess.adapters.base import SegmentationOutput
from minivess.pipeline.multitask_loss import AuxHeadLossConfig, MultiTaskLoss


def _make_output_and_batch(
    batch_size: int = 2,
    spatial: tuple[int, int, int] = (4, 8, 8),
    aux_names: list[str] | None = None,
) -> tuple[SegmentationOutput, dict[str, torch.Tensor]]:
    """Create synthetic SegmentationOutput + batch dict for testing."""
    logits = torch.randn(batch_size, 2, *spatial)
    metadata: dict[str, torch.Tensor] = {}
    batch: dict[str, torch.Tensor] = {
        "label": torch.randint(0, 2, (batch_size, 1, *spatial)).float(),
    }
    for name in aux_names or []:
        metadata[name] = torch.randn(batch_size, 1, *spatial)
        batch[name] = torch.randn(batch_size, 1, *spatial)

    output = SegmentationOutput(
        prediction=torch.softmax(logits, dim=1),
        logits=logits,
        metadata=metadata,
    )
    return output, batch


class TestMultiTaskLoss:
    """Tests for MultiTaskLoss nn.Module."""

    def test_multitask_loss_returns_scalar(self) -> None:
        """Output is scalar tensor."""
        seg_crit = torch.nn.CrossEntropyLoss()
        configs = [
            AuxHeadLossConfig(
                name="sdf", loss_type="smooth_l1", weight=0.25, gt_key="sdf"
            ),
        ]
        loss_fn = MultiTaskLoss(seg_criterion=seg_crit, aux_head_configs=configs)
        output, batch = _make_output_and_batch(aux_names=["sdf"])
        loss = loss_fn(output, batch)
        assert loss.dim() == 0, "Loss should be scalar"

    def test_multitask_loss_upgraded_signature(self) -> None:
        """forward(output, batch) works."""
        seg_crit = torch.nn.CrossEntropyLoss()
        loss_fn = MultiTaskLoss(seg_criterion=seg_crit, aux_head_configs=[])
        output, batch = _make_output_and_batch()
        loss = loss_fn(output, batch)
        assert torch.isfinite(loss)

    def test_multitask_loss_generic_two_heads(self) -> None:
        """Works with 2 arbitrary aux heads."""
        seg_crit = torch.nn.CrossEntropyLoss()
        configs = [
            AuxHeadLossConfig(
                name="head_a", loss_type="smooth_l1", weight=0.2, gt_key="head_a"
            ),
            AuxHeadLossConfig(
                name="head_b", loss_type="mse", weight=0.3, gt_key="head_b"
            ),
        ]
        loss_fn = MultiTaskLoss(seg_criterion=seg_crit, aux_head_configs=configs)
        output, batch = _make_output_and_batch(aux_names=["head_a", "head_b"])
        loss = loss_fn(output, batch)
        assert torch.isfinite(loss)

    def test_multitask_loss_generic_single_head(self) -> None:
        """Works with 1 aux head."""
        seg_crit = torch.nn.CrossEntropyLoss()
        configs = [
            AuxHeadLossConfig(name="aux", loss_type="mse", weight=0.5, gt_key="aux"),
        ]
        loss_fn = MultiTaskLoss(seg_criterion=seg_crit, aux_head_configs=configs)
        output, batch = _make_output_and_batch(aux_names=["aux"])
        loss = loss_fn(output, batch)
        assert torch.isfinite(loss)

    def test_multitask_loss_generic_zero_heads(self) -> None:
        """Seg-only mode (no aux heads)."""
        seg_crit = torch.nn.CrossEntropyLoss()
        loss_fn = MultiTaskLoss(seg_criterion=seg_crit, aux_head_configs=[])
        output, batch = _make_output_and_batch()
        loss = loss_fn(output, batch)
        assert torch.isfinite(loss)

    def test_multitask_loss_gradient_flows(self) -> None:
        """All heads receive gradients."""
        seg_crit = torch.nn.CrossEntropyLoss()
        configs = [
            AuxHeadLossConfig(
                name="sdf", loss_type="smooth_l1", weight=0.25, gt_key="sdf"
            ),
        ]
        loss_fn = MultiTaskLoss(seg_criterion=seg_crit, aux_head_configs=configs)
        output, batch = _make_output_and_batch(aux_names=["sdf"])
        # Make logits require grad
        output.logits.requires_grad_(True)
        output.metadata["sdf"].requires_grad_(True)
        loss = loss_fn(output, batch)
        loss.backward()
        assert output.logits.grad is not None
        assert output.metadata["sdf"].grad is not None

    def test_multitask_loss_weights_respected(self) -> None:
        """Changing weights changes loss."""
        seg_crit = torch.nn.CrossEntropyLoss()
        torch.manual_seed(42)
        output, batch = _make_output_and_batch(aux_names=["sdf"])

        configs_low = [
            AuxHeadLossConfig(name="sdf", loss_type="mse", weight=0.01, gt_key="sdf"),
        ]
        configs_high = [
            AuxHeadLossConfig(name="sdf", loss_type="mse", weight=10.0, gt_key="sdf"),
        ]
        loss_low = MultiTaskLoss(seg_criterion=seg_crit, aux_head_configs=configs_low)(
            output, batch
        )
        loss_high = MultiTaskLoss(
            seg_criterion=seg_crit, aux_head_configs=configs_high
        )(output, batch)
        assert loss_high > loss_low, "Higher weight should give higher loss"

    def test_multitask_loss_masked_regression(self) -> None:
        """Regression loss zero outside vessel when mask_to_foreground=True."""
        seg_crit = torch.nn.CrossEntropyLoss()
        configs = [
            AuxHeadLossConfig(
                name="sdf",
                loss_type="smooth_l1",
                weight=1.0,
                gt_key="sdf",
                mask_to_foreground=True,
            ),
        ]
        loss_fn = MultiTaskLoss(seg_criterion=seg_crit, aux_head_configs=configs)

        # Create batch where label is all zero (no foreground)
        output, batch = _make_output_and_batch(aux_names=["sdf"])
        batch["label"] = torch.zeros_like(batch["label"])

        loss = loss_fn(output, batch)
        assert torch.isfinite(loss)
        # With all-zero foreground mask, aux loss should be zero
        assert loss_fn.component_losses["loss/sdf"] == 0.0

    def test_multitask_loss_unmasked_classification(self) -> None:
        """Classification without fg mask works."""
        seg_crit = torch.nn.CrossEntropyLoss()
        configs = [
            AuxHeadLossConfig(
                name="cls",
                loss_type="bce",
                weight=0.5,
                gt_key="cls",
                mask_to_foreground=False,
            ),
        ]
        loss_fn = MultiTaskLoss(seg_criterion=seg_crit, aux_head_configs=configs)
        output, batch = _make_output_and_batch(aux_names=["cls"])
        # Make GT sigmoid-compatible
        batch["cls"] = torch.sigmoid(batch["cls"])
        loss = loss_fn(output, batch)
        assert torch.isfinite(loss)

    def test_multitask_loss_perfect_pred_zero_aux(self) -> None:
        """Zero aux loss for perfect prediction."""
        seg_crit = torch.nn.CrossEntropyLoss()
        configs = [
            AuxHeadLossConfig(name="sdf", loss_type="mse", weight=1.0, gt_key="sdf"),
        ]
        loss_fn = MultiTaskLoss(seg_criterion=seg_crit, aux_head_configs=configs)
        output, batch = _make_output_and_batch(aux_names=["sdf"])
        # Set prediction = ground truth
        output.metadata["sdf"] = batch["sdf"].clone()
        loss_fn(output, batch)
        # Aux loss should be zero (or near zero)
        assert loss_fn.component_losses["loss/sdf"] < 1e-6

    def test_multitask_loss_empty_foreground_no_nan(self) -> None:
        """No NaN when foreground empty and mask_to_foreground=True."""
        seg_crit = torch.nn.CrossEntropyLoss()
        configs = [
            AuxHeadLossConfig(
                name="sdf",
                loss_type="mse",
                weight=1.0,
                gt_key="sdf",
                mask_to_foreground=True,
            ),
        ]
        loss_fn = MultiTaskLoss(seg_criterion=seg_crit, aux_head_configs=configs)
        output, batch = _make_output_and_batch(aux_names=["sdf"])
        batch["label"] = torch.zeros_like(batch["label"])
        loss = loss_fn(output, batch)
        assert torch.isfinite(loss), "Loss should be finite even with empty foreground"

    def test_multitask_loss_per_component_dict(self) -> None:
        """Stores loss/{name} for each head."""
        seg_crit = torch.nn.CrossEntropyLoss()
        configs = [
            AuxHeadLossConfig(name="sdf", loss_type="mse", weight=0.25, gt_key="sdf"),
            AuxHeadLossConfig(
                name="cl", loss_type="smooth_l1", weight=0.25, gt_key="cl"
            ),
        ]
        loss_fn = MultiTaskLoss(seg_criterion=seg_crit, aux_head_configs=configs)
        output, batch = _make_output_and_batch(aux_names=["sdf", "cl"])
        loss_fn(output, batch)
        assert "loss/seg" in loss_fn.component_losses
        assert "loss/sdf" in loss_fn.component_losses
        assert "loss/cl" in loss_fn.component_losses

    def test_multitask_loss_four_aux_losses(self) -> None:
        """Works with 4 different aux loss types."""
        seg_crit = torch.nn.CrossEntropyLoss()
        configs = [
            AuxHeadLossConfig(name="a", loss_type="smooth_l1", weight=0.1, gt_key="a"),
            AuxHeadLossConfig(name="b", loss_type="mse", weight=0.1, gt_key="b"),
            AuxHeadLossConfig(name="c", loss_type="bce", weight=0.1, gt_key="c"),
            AuxHeadLossConfig(
                name="d", loss_type="cross_entropy", weight=0.1, gt_key="d"
            ),
        ]
        loss_fn = MultiTaskLoss(seg_criterion=seg_crit, aux_head_configs=configs)
        output, batch = _make_output_and_batch(aux_names=["a", "b", "c", "d"])
        # Make BCE-compatible GTs
        batch["c"] = torch.sigmoid(batch["c"])
        loss = loss_fn(output, batch)
        assert torch.isfinite(loss)
