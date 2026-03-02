"""Tests for trainer criterion upgrade for generic multi-task (T9a — #234)."""

from __future__ import annotations

import torch

from minivess.adapters.base import SegmentationOutput
from minivess.adapters.multitask_adapter import AuxHeadConfig, MultiTaskAdapter
from minivess.pipeline.multitask_loss import AuxHeadLossConfig, MultiTaskLoss


class _StubNet(torch.nn.Module):  # type: ignore[misc]
    """Inner network for stub model."""

    def __init__(self) -> None:
        super().__init__()
        self.decoder_conv = torch.nn.Conv3d(1, 16, 3, padding=1)
        self.output_conv = torch.nn.Conv3d(16, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_conv(self.decoder_conv(x))


class _StubModel(torch.nn.Module):  # type: ignore[misc]
    """Minimal stub model for training tests."""

    def __init__(self) -> None:
        super().__init__()
        self.net = _StubNet()

    def forward(self, x: torch.Tensor) -> SegmentationOutput:
        logits = self.net(x)
        return SegmentationOutput(
            prediction=torch.softmax(logits, dim=1),
            logits=logits,
            metadata={},
        )


def _make_batch(
    batch_size: int = 1,
    spatial: tuple[int, int, int] = (4, 8, 8),
) -> dict[str, torch.Tensor]:
    """Create a batch dict with image, label, and aux GT keys."""
    return {
        "image": torch.randn(batch_size, 1, *spatial),
        "label": torch.randint(0, 2, (batch_size, 1, *spatial)).float(),
        "sdf": torch.randn(batch_size, 1, *spatial),
    }


class TestTrainerCriterionUpgrade:
    """Tests for trainer criterion interface upgrade (T9a)."""

    def test_trainer_multitask_criterion_receives_output(self) -> None:
        """MultiTaskLoss gets SegmentationOutput, not raw logits."""
        seg_crit = torch.nn.CrossEntropyLoss()
        configs = [
            AuxHeadLossConfig(name="sdf", loss_type="mse", weight=0.25, gt_key="sdf"),
        ]
        loss_fn = MultiTaskLoss(seg_criterion=seg_crit, aux_head_configs=configs)

        # Create a multitask adapter
        base = _StubModel()
        aux_configs = [
            AuxHeadConfig(name="sdf", head_type="regression", out_channels=1)
        ]
        adapter = MultiTaskAdapter(base_model=base, aux_head_configs=aux_configs)

        batch = _make_batch()
        output = adapter(batch["image"])

        # MultiTaskLoss should accept (output, batch)
        loss = loss_fn(output, batch)
        assert torch.isfinite(loss)

    def test_trainer_multitask_criterion_receives_batch(self) -> None:
        """MultiTaskLoss gets full batch dict with GT keys."""
        seg_crit = torch.nn.CrossEntropyLoss()
        configs = [
            AuxHeadLossConfig(name="sdf", loss_type="mse", weight=0.25, gt_key="sdf"),
        ]
        loss_fn = MultiTaskLoss(seg_criterion=seg_crit, aux_head_configs=configs)

        base = _StubModel()
        aux_configs = [
            AuxHeadConfig(name="sdf", head_type="regression", out_channels=1)
        ]
        adapter = MultiTaskAdapter(base_model=base, aux_head_configs=aux_configs)

        batch = _make_batch()
        output = adapter(batch["image"])
        loss_fn(output, batch)

        # Per-component losses should be populated
        assert "loss/seg" in loss_fn.component_losses
        assert "loss/sdf" in loss_fn.component_losses

    def test_trainer_standard_criterion_unchanged(self) -> None:
        """Non-multitask criterion still works with (logits, labels)."""
        criterion = torch.nn.CrossEntropyLoss()
        base = _StubModel()
        batch = _make_batch()
        output = base(batch["image"])

        # Standard criterion: (logits, labels)
        label = batch["label"].squeeze(1).long()
        loss = criterion(output.logits, label)
        assert torch.isfinite(loss)

    def test_multitask_training_step(self) -> None:
        """One training step with MultiTaskAdapter + MultiTaskLoss completes."""
        base = _StubModel()
        aux_configs = [
            AuxHeadConfig(name="sdf", head_type="regression", out_channels=1)
        ]
        adapter = MultiTaskAdapter(base_model=base, aux_head_configs=aux_configs)

        seg_crit = torch.nn.CrossEntropyLoss()
        loss_configs = [
            AuxHeadLossConfig(name="sdf", loss_type="mse", weight=0.25, gt_key="sdf"),
        ]
        loss_fn = MultiTaskLoss(seg_criterion=seg_crit, aux_head_configs=loss_configs)

        optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)
        batch = _make_batch()

        # Forward
        output = adapter(batch["image"])
        loss = loss_fn(output, batch)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss)

    def test_multitask_backward_updates_all_heads(self) -> None:
        """All three task heads updated after one training step."""
        base = _StubModel()
        aux_configs = [
            AuxHeadConfig(name="sdf", head_type="regression", out_channels=1),
        ]
        adapter = MultiTaskAdapter(base_model=base, aux_head_configs=aux_configs)

        seg_crit = torch.nn.CrossEntropyLoss()
        loss_configs = [
            AuxHeadLossConfig(name="sdf", loss_type="mse", weight=0.25, gt_key="sdf"),
        ]
        loss_fn = MultiTaskLoss(seg_criterion=seg_crit, aux_head_configs=loss_configs)

        batch = _make_batch()
        output = adapter(batch["image"])
        loss = loss_fn(output, batch)
        loss.backward()

        # Check gradients on both base model and aux head
        for name, p in adapter.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_trainer_standard_model_no_regression(self) -> None:
        """Standard model ignores aux keys in batch."""
        criterion = torch.nn.CrossEntropyLoss()
        base = _StubModel()
        batch = _make_batch()

        # Standard model doesn't produce aux outputs
        output = base(batch["image"])
        assert "sdf" not in output.metadata

        # Standard loss works fine
        label = batch["label"].squeeze(1).long()
        loss = criterion(output.logits, label)
        assert torch.isfinite(loss)

    def test_is_multitask_loss_detection(self) -> None:
        """isinstance check correctly identifies MultiTaskLoss."""
        seg_crit = torch.nn.CrossEntropyLoss()
        multitask_loss = MultiTaskLoss(seg_criterion=seg_crit, aux_head_configs=[])

        assert isinstance(multitask_loss, MultiTaskLoss)
        assert not isinstance(seg_crit, MultiTaskLoss)
