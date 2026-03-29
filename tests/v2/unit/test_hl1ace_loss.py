"""Tests for Hard L1-ACE auxiliary calibration loss.

Ported from cai4cai/Average-Calibration-Losses (CC BY 4.0).
Tests validate the hard-binned L1 Average Calibration Error (hL1-ACE)
loss for binary segmentation calibration.

Reference: Barfoot et al. (2025). "Average Calibration Losses." IEEE TMI.
"""

from __future__ import annotations

import torch


def _make_binary_seg_pair(
    batch: int = 2,
    channels: int = 2,
    spatial: tuple[int, ...] = (8, 8, 8),
    *,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create random (logits, labels) pair for binary segmentation."""
    logits = torch.randn(batch, channels, *spatial, device=device, requires_grad=True)
    labels = torch.randint(0, channels, (batch, 1, *spatial), device=device)
    return logits, labels


class TestHL1ACELossReturnsScalar:
    """T1: Loss returns finite scalar for random inputs."""

    def test_hl1ace_loss_returns_scalar(self) -> None:
        from minivess.pipeline.calibration_losses import HL1ACELoss

        loss_fn = HL1ACELoss()
        logits, labels = _make_binary_seg_pair()
        result = loss_fn(logits, labels)

        assert result.ndim == 0, "Loss should be a scalar"
        assert torch.isfinite(result), "Loss should be finite"
        assert result.dtype == torch.float32


class TestHL1ACELossGradientFlows:
    """T1: Gradient flows through the loss."""

    def test_hl1ace_loss_gradient_flows(self) -> None:
        from minivess.pipeline.calibration_losses import HL1ACELoss

        loss_fn = HL1ACELoss()
        logits, labels = _make_binary_seg_pair()
        result = loss_fn(logits, labels)
        result.backward()

        assert logits.grad is not None, "Gradient should flow to logits"
        assert torch.isfinite(logits.grad).all(), "Gradients should be finite"


class TestHL1ACELossPerfectCalibration:
    """T1: Near-perfectly calibrated predictions produce near-zero loss."""

    def test_hl1ace_loss_perfect_calibration_near_zero(self) -> None:
        from minivess.pipeline.calibration_losses import HL1ACELoss

        loss_fn = HL1ACELoss(n_bins=10)
        # Create perfectly calibrated predictions:
        # Where label=1, predict high confidence; where label=0, predict low confidence.
        batch, channels, d, h, w = 2, 2, 8, 8, 8
        labels = torch.zeros(batch, 1, d, h, w, dtype=torch.long)
        labels[:, :, :4, :, :] = 1  # First half is foreground

        # Create logits that produce near-perfect softmax probabilities
        # High logit for the correct class
        logits = torch.zeros(batch, channels, d, h, w)
        logits[:, 1, :4, :, :] = 10.0  # Foreground class high where label=1
        logits[:, 0, 4:, :, :] = 10.0  # Background class high where label=0
        logits.requires_grad_(True)

        result = loss_fn(logits, labels)
        assert result.item() < 0.05, (
            f"Perfect calibration should give near-zero loss, got {result.item()}"
        )


class TestHL1ACELossOverconfident:
    """T1: Overconfident predictions produce positive loss."""

    def test_hl1ace_loss_overconfident_positive(self) -> None:
        from minivess.pipeline.calibration_losses import HL1ACELoss

        loss_fn = HL1ACELoss()
        batch, channels, d, h, w = 2, 2, 8, 8, 8
        labels = torch.randint(0, 2, (batch, 1, d, h, w))

        # Overconfident: always predict class 0 with high confidence
        logits = torch.zeros(batch, channels, d, h, w, requires_grad=True)
        logits_data = logits.detach().clone()
        logits_data[:, 0, :, :, :] = 10.0  # Always confident class 0
        logits = logits_data.requires_grad_(True)

        result = loss_fn(logits, labels)
        assert result.item() > 0.0, (
            "Overconfident predictions should produce positive loss"
        )


class TestHL1ACELossNBinsConfigurable:
    """T1: n_bins parameter is configurable."""

    def test_hl1ace_loss_n_bins_configurable(self) -> None:
        from minivess.pipeline.calibration_losses import HL1ACELoss

        loss_5 = HL1ACELoss(n_bins=5)
        loss_15 = HL1ACELoss(n_bins=15)
        loss_20 = HL1ACELoss(n_bins=20)

        assert loss_5.n_bins == 5
        assert loss_15.n_bins == 15
        assert loss_20.n_bins == 20

        # Default should be 15 per Barfoot et al.
        loss_default = HL1ACELoss()
        assert loss_default.n_bins == 15

        # All should produce finite output
        logits, labels = _make_binary_seg_pair()
        for loss_fn in [loss_5, loss_15, loss_20]:
            result = loss_fn(logits, labels)
            assert torch.isfinite(result), (
                f"Loss with n_bins={loss_fn.n_bins} should be finite"
            )


class TestHL1ACELossBatchIndependence:
    """T1: Loss is computed per-batch then averaged."""

    def test_hl1ace_loss_batch_independence(self) -> None:
        from minivess.pipeline.calibration_losses import HL1ACELoss

        loss_fn = HL1ACELoss()
        torch.manual_seed(42)

        # Single sample
        logits_1, labels_1 = _make_binary_seg_pair(batch=1)
        result_1 = loss_fn(logits_1, labels_1)

        # Different single sample
        logits_2, labels_2 = _make_binary_seg_pair(batch=1)
        result_2 = loss_fn(logits_2, labels_2)

        # Both should be finite scalars
        assert torch.isfinite(result_1)
        assert torch.isfinite(result_2)


class TestHL1ACELossAllEmptyBins:
    """T2 double-check: HL1ACE must not return NaN on extreme logits (all bins empty)."""

    def test_hl1ace_extreme_logits_one_channel_all_empty_bins(self) -> None:
        from minivess.pipeline.calibration_losses import HL1ACELoss

        # Channel 0 logits very high → softmax channel 1 ≈ 0.0
        # All channel-1 voxels land in bin 0 → other bins are NaN
        # Target has NO foreground (all-zero) → non_empty for channel 1 is False
        # ace_per_bc[b,1] is NaN (from nanmean) × 0.0 (non_empty) = NaN (IEEE 754)
        # This NaN poisons .mean() at line 100
        logits = torch.zeros(1, 2, 8, 8, 8, requires_grad=True)
        logits_data = logits.detach().clone()
        logits_data[:, 0, :, :, :] = 100.0  # Channel 0 dominates
        logits = logits_data.requires_grad_(True)
        labels = torch.zeros(1, 1, 8, 8, 8)  # All background
        loss_fn = HL1ACELoss()
        result = loss_fn(logits, labels)
        assert torch.isfinite(result), (
            f"HL1ACE should not return NaN on extreme logits, got {result.item()}"
        )

    def test_hl1ace_all_same_prediction_finite(self) -> None:
        from minivess.pipeline.calibration_losses import HL1ACELoss

        # Uniform softmax (0.5) — some bins may be empty
        logits = torch.zeros(1, 2, 8, 8, 8, requires_grad=True)
        labels = torch.zeros(1, 1, 8, 8, 8)
        result = HL1ACELoss()(logits, labels)
        assert torch.isfinite(result), (
            f"HL1ACE should be finite for uniform predictions, got {result.item()}"
        )


class TestHL1ACELossBinarySegmentation:
    """T1: Works correctly with binary segmentation (C=2 channels)."""

    def test_hl1ace_loss_binary_segmentation(self) -> None:
        from minivess.pipeline.calibration_losses import HL1ACELoss

        loss_fn = HL1ACELoss()
        # Binary segmentation: 2 channels (background + foreground)
        logits, labels = _make_binary_seg_pair(channels=2)
        result = loss_fn(logits, labels)

        assert result.ndim == 0
        assert torch.isfinite(result)
        assert result.item() >= 0.0, "Calibration loss should be non-negative"
