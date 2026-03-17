"""Tests for auxiliary calibration loss integration into compound loss factory.

Validates the with_aux_calib flag that wraps any segmentation loss with hL1-ACE.
"""

from __future__ import annotations

import pytest
import torch


def _make_binary_seg_pair(
    batch: int = 2,
    channels: int = 2,
    spatial: tuple[int, ...] = (8, 8, 8),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create random (logits, labels) pair for binary segmentation."""
    logits = torch.randn(batch, channels, *spatial, requires_grad=True)
    labels = torch.randint(0, channels, (batch, 1, *spatial))
    return logits, labels


class TestCompoundLossWithoutAuxCalib:
    """T2: with_aux_calib=False produces identical loss to baseline."""

    def test_compound_loss_without_aux_calib(self) -> None:
        from minivess.pipeline.loss_functions import build_loss_function

        # Build baseline loss
        baseline = build_loss_function("dice_ce")

        # Build same loss with aux_calib disabled (should be identical)
        compound = build_loss_function("dice_ce", with_aux_calib=False)

        logits, labels = _make_binary_seg_pair()
        with torch.no_grad():
            baseline_val = baseline(logits, labels)
            compound_val = compound(logits, labels)

        assert torch.allclose(baseline_val, compound_val, atol=1e-6), (
            f"with_aux_calib=False should produce identical loss: "
            f"baseline={baseline_val.item()}, compound={compound_val.item()}"
        )


class TestCompoundLossWithAuxCalib:
    """T2: with_aux_calib=True adds hL1-ACE term."""

    def test_compound_loss_with_aux_calib(self) -> None:
        from minivess.pipeline.loss_functions import build_loss_function

        baseline = build_loss_function("dice_ce")
        compound = build_loss_function("dice_ce", with_aux_calib=True)

        logits, labels = _make_binary_seg_pair()
        with torch.no_grad():
            baseline_val = baseline(logits, labels)
            compound_val = compound(logits, labels)

        # Compound with aux_calib should be different (adds calibration term)
        assert not torch.allclose(baseline_val, compound_val, atol=1e-6), (
            "with_aux_calib=True should produce different loss than baseline"
        )
        # Compound should be >= baseline (adds non-negative calibration loss)
        # Note: hL1-ACE is non-negative, so total >= seg_loss
        assert compound_val >= baseline_val - 1e-6


class TestCompoundLossAuxCalibWeight:
    """T2: aux_calib_weight scales the auxiliary term."""

    def test_compound_loss_aux_calib_weight(self) -> None:
        from minivess.pipeline.loss_functions import build_loss_function

        baseline = build_loss_function("dice_ce")
        compound_w1 = build_loss_function(
            "dice_ce", with_aux_calib=True, aux_calib_weight=1.0
        )
        compound_w01 = build_loss_function(
            "dice_ce", with_aux_calib=True, aux_calib_weight=0.1
        )

        logits, labels = _make_binary_seg_pair()
        with torch.no_grad():
            base_val = baseline(logits, labels)
            w1_val = compound_w1(logits, labels)
            w01_val = compound_w01(logits, labels)

        # Weight=0.1 should produce a smaller aux term than weight=1.0
        aux_w1 = w1_val - base_val
        aux_w01 = w01_val - base_val
        if aux_w1.item() > 0.001:  # Only check ratio if aux term is non-trivial
            assert aux_w01 < aux_w1, (
                "Lower aux_calib_weight should produce smaller calibration term"
            )


class TestCompoundLossAuxCalibGradient:
    """T2: Gradient flows through the compound loss with aux_calib."""

    def test_compound_loss_aux_calib_gradient(self) -> None:
        from minivess.pipeline.loss_functions import build_loss_function

        compound = build_loss_function("dice_ce", with_aux_calib=True)
        logits, labels = _make_binary_seg_pair()
        result = compound(logits, labels)
        result.backward()

        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()


class TestLossFactoryAuxCalibFlag:
    """T2: Loss factory correctly creates compound loss from config."""

    def test_loss_factory_aux_calib_flag(self) -> None:
        from minivess.pipeline.loss_functions import build_loss_function

        # Default: no aux_calib
        loss_default = build_loss_function("cbdice_cldice")

        # With aux_calib
        loss_aux = build_loss_function("cbdice_cldice", with_aux_calib=True)

        # The aux version should be a wrapper type
        # The default should be the original loss type
        assert type(loss_default).__name__ == "CbDiceClDiceLoss"

        logits, labels = _make_binary_seg_pair()
        result = loss_aux(logits, labels)
        assert torch.isfinite(result)


class TestLossFactoryAllLossesWithAuxCalib:
    """T2: All 3 main segmentation losses work with aux_calib."""

    @pytest.mark.parametrize(
        "loss_name",
        ["dice_ce", "cbdice_cldice", "dice_ce_cldice"],
    )
    def test_loss_factory_all_losses_with_aux_calib(self, loss_name: str) -> None:
        from minivess.pipeline.loss_functions import build_loss_function

        loss_fn = build_loss_function(loss_name, with_aux_calib=True)
        logits, labels = _make_binary_seg_pair()
        result = loss_fn(logits, labels)

        assert torch.isfinite(result), (
            f"{loss_name} with aux_calib should produce finite loss"
        )
        assert result.item() >= 0.0, (
            f"{loss_name} with aux_calib should be non-negative"
        )
