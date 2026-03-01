"""Regression tests for verified loss function bugs.

These tests capture the 5 bugs discovered during pre-PR code review of the
graph-constrained-models branch. Each test should FAIL before its corresponding
fix is applied (RED phase), then PASS after (GREEN phase).

See docs/planning/debug-training-all-losses-plan.md for the full bug inventory.
"""

from __future__ import annotations

import pytest
import torch

from minivess.pipeline.loss_functions import build_loss_function


class TestStandaloneCldicePreprocessing:
    """Bug #1: Standalone cldice receives raw logits + 1-ch labels without
    softmax/one-hot preprocessing (unlike compound losses that wrap it)."""

    def test_standalone_cldice_perfect_pred_near_zero(self) -> None:
        """Perfect prediction should give near-zero loss for standalone cldice."""
        torch.manual_seed(42)
        labels = torch.zeros(1, 1, 8, 16, 16)
        labels[:, :, 2:6, 4:12, 4:12] = 1.0

        # Build logits that perfectly match labels
        logits = torch.zeros(1, 2, 8, 16, 16)
        labels_int = labels.squeeze(1).long()
        for c in range(2):
            logits[:, c] = (labels_int == c).float() * 10.0 - 5.0
        logits.requires_grad_(True)

        loss_fn = build_loss_function("cldice")
        loss = loss_fn(logits, labels)
        assert torch.isfinite(loss), f"cldice loss is not finite: {loss.item()}"
        assert loss.item() < 0.1, (
            f"cldice loss for perfect prediction should be near zero, got {loss.item()}"
        )

    def test_standalone_cldice_has_gradient(self) -> None:
        """Standalone cldice must produce valid gradients."""
        torch.manual_seed(42)
        logits = torch.randn(1, 2, 8, 16, 16, requires_grad=True)
        labels = torch.zeros(1, 1, 8, 16, 16)
        labels[:, :, 2:6, 4:12, 4:12] = 1.0

        loss_fn = build_loss_function("cldice")
        loss = loss_fn(logits, labels)
        loss.backward()
        assert logits.grad is not None, "No gradient from standalone cldice"
        assert not torch.any(torch.isnan(logits.grad)), "NaN in cldice gradient"


class TestCbdiceSmallSpatialDims:
    """Bug #2: cbdice avg_pool3d(kernel_size=5) crashes when spatial dim < 5."""

    @pytest.mark.parametrize(
        "spatial_size",
        [(5, 16, 16), (4, 16, 16), (3, 8, 8)],
        ids=["z5", "z4", "z3"],
    )
    def test_cbdice_small_z_no_crash(self, spatial_size: tuple[int, ...]) -> None:
        """cbdice must not crash on small spatial dimensions."""
        torch.manual_seed(42)
        d, h, w = spatial_size
        logits = torch.randn(1, 2, d, h, w, requires_grad=True)
        labels = torch.zeros(1, 1, d, h, w)
        labels[:, :, d // 4 : d // 2, h // 4 : h // 2, w // 4 : w // 2] = 1.0

        loss_fn = build_loss_function("cbdice")
        loss = loss_fn(logits, labels)
        assert torch.isfinite(loss), (
            f"cbdice not finite at {spatial_size}: {loss.item()}"
        )
        loss.backward()
        assert logits.grad is not None

    def test_cbdice_cldice_small_z_no_crash(self) -> None:
        """Compound cbdice_cldice must also survive small Z."""
        torch.manual_seed(42)
        logits = torch.randn(1, 2, 5, 16, 16, requires_grad=True)
        labels = torch.zeros(1, 1, 5, 16, 16)
        labels[:, :, 1:3, 4:12, 4:12] = 1.0

        loss_fn = build_loss_function("cbdice_cldice")
        loss = loss_fn(logits, labels)
        assert torch.isfinite(loss), f"cbdice_cldice not finite at Z=5: {loss.item()}"


class TestTopoLossSmallZ:
    """Bug #3: TopoLoss NaN at Z<=7 due to avg_pool3d producing empty tensors."""

    @pytest.mark.parametrize(
        "spatial_size",
        [(5, 16, 16), (7, 16, 16), (3, 8, 8)],
        ids=["z5", "z7", "z3"],
    )
    def test_topoloss_small_z_no_nan(self, spatial_size: tuple[int, ...]) -> None:
        """TopoLoss must produce finite values at small Z dimensions."""
        torch.manual_seed(42)
        d, h, w = spatial_size
        logits = torch.randn(1, 2, d, h, w, requires_grad=True)
        labels = torch.zeros(1, 1, d, h, w)
        labels[:, :, d // 4 : d // 2, h // 4 : h // 2, w // 4 : w // 2] = 1.0

        loss_fn = build_loss_function("topo")
        loss = loss_fn(logits, labels)
        assert torch.isfinite(loss), (
            f"topo loss not finite at {spatial_size}: {loss.item()}"
        )
        loss.backward()
        assert logits.grad is not None
        assert not torch.any(torch.isnan(logits.grad)), "NaN in topo gradient"

    def test_full_topo_small_z_no_nan(self) -> None:
        """full_topo (compound with TopoLoss-like BettiLoss) at small Z."""
        torch.manual_seed(42)
        logits = torch.randn(1, 2, 5, 16, 16, requires_grad=True)
        labels = torch.zeros(1, 1, 5, 16, 16)
        labels[:, :, 1:3, 4:12, 4:12] = 1.0

        loss_fn = build_loss_function("full_topo")
        loss = loss_fn(logits, labels)
        assert torch.isfinite(loss), f"full_topo not finite at Z=5: {loss.item()}"


class TestTopoSegLossGradientChain:
    """Bug #4: TopoSegLoss returns torch.tensor(0.0) without requires_grad
    when no critical points are found, breaking the gradient chain."""

    def test_toposegloss_zero_critical_has_grad(self) -> None:
        """TopoSegLoss must preserve gradient chain even with zero critical points."""
        torch.manual_seed(42)
        # Create logits that perfectly match labels (minimal disagreement)
        # to trigger the zero-critical-points path
        labels = torch.zeros(1, 1, 8, 16, 16)
        labels[:, :, 2:6, 4:12, 4:12] = 1.0

        logits = torch.zeros(1, 2, 8, 16, 16)
        labels_int = labels.squeeze(1).long()
        for c in range(2):
            logits[:, c] = (labels_int == c).float() * 10.0 - 5.0
        logits.requires_grad_(True)

        loss_fn = build_loss_function("toposeg")
        loss = loss_fn(logits, labels)
        assert loss.requires_grad, (
            "TopoSegLoss must return a tensor with requires_grad=True, "
            f"got requires_grad={loss.requires_grad}"
        )
        loss.backward()
        assert logits.grad is not None, "No gradient from TopoSegLoss"


class TestBettiMatchingScale:
    """Bug #5: betti_matching returns ~84 while all other losses return 0.3-0.8,
    causing it to dominate any compound loss."""

    def test_betti_matching_magnitude_bounded(self) -> None:
        """betti_matching loss should be in a comparable range to other losses."""
        torch.manual_seed(42)
        logits = torch.randn(1, 2, 8, 16, 16, requires_grad=True)
        labels = torch.zeros(1, 1, 8, 16, 16)
        labels[:, :, 2:6, 4:12, 4:12] = 1.0

        loss_fn = build_loss_function("betti_matching")
        loss = loss_fn(logits, labels)
        assert torch.isfinite(loss), f"betti_matching not finite: {loss.item()}"
        assert loss.item() < 20.0, (
            f"betti_matching loss too large ({loss.item():.1f}), "
            "should be normalized to comparable scale as other losses"
        )
