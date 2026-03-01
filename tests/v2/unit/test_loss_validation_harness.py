"""Production-grade loss validation test harness.

Validates every graph-topology sweep loss function against:
  1. Gradient sanity — finite forward, non-zero backward, no NaN edge cases
  2. Value sanity — perfect < random, non-negative, empty mask handling
  3. Convergence smoke — overfit single sample in 20 gradient steps

All tests run CPU-only with synthetic tensors (no GPU or dataset required).

See docs/planning/novel-loss-debugging-plan.xml for full validation plan.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from minivess.pipeline.loss_functions import build_loss_function

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# The 5 losses in the graph-topology sweep
SWEEP_LOSSES = [
    "dice_ce",
    "cbdice_cldice",
    "graph_topology",
    "skeleton_recall",
    "betti_matching",
]

# Spatial size must be large enough for skeleton operations
SPATIAL = (8, 16, 16)
B, C = 1, 2

# Multi-patch-size parametrization for production edge cases
SPATIAL_SIZES = [
    (8, 16, 16),  # Default
    (5, 16, 16),  # MiniVess minimum Z dimension (critical edge case)
    (16, 16, 16),  # Comfortable positive control
]


def _random_logits() -> torch.Tensor:
    """Random logits (B, C, D, H, W) with requires_grad."""
    return torch.randn(B, C, *SPATIAL, requires_grad=True)


def _random_labels() -> torch.Tensor:
    """Random binary labels (B, 1, D, H, W)."""
    return torch.randint(0, 2, (B, 1, *SPATIAL)).float()


def _perfect_logits(labels: torch.Tensor) -> torch.Tensor:
    """Logits that perfectly match labels (high confidence)."""
    labels_int = labels.squeeze(1).long()
    logits = torch.zeros(B, C, *SPATIAL)
    for c in range(C):
        logits[:, c] = (labels_int == c).float() * 10.0 - 5.0
    logits.requires_grad_(True)
    return logits


def _all_zeros_logits() -> torch.Tensor:
    """Logits that produce near-zero foreground probability."""
    logits = torch.zeros(B, C, *SPATIAL)
    logits[:, 0] = 10.0  # High background confidence
    logits[:, 1] = -10.0
    logits.requires_grad_(True)
    return logits


def _all_ones_logits() -> torch.Tensor:
    """Logits that produce near-one foreground probability."""
    logits = torch.zeros(B, C, *SPATIAL)
    logits[:, 0] = -10.0
    logits[:, 1] = 10.0  # High foreground confidence
    logits.requires_grad_(True)
    return logits


def _tube_labels() -> torch.Tensor:
    """Labels with a simple tube structure (good for skeleton/topology tests)."""
    labels = torch.zeros(B, 1, *SPATIAL)
    # Create a tube along the D axis in the center
    d, h, w = SPATIAL
    labels[:, 0, :, h // 2 - 1 : h // 2 + 1, w // 2 - 1 : w // 2 + 1] = 1.0
    return labels


# ---------------------------------------------------------------------------
# Category 1: Gradient Sanity
# ---------------------------------------------------------------------------


class TestGradientSanity:
    """Every loss must produce finite, non-zero gradients."""

    @pytest.mark.parametrize("loss_name", SWEEP_LOSSES)
    def test_forward_finite(self, loss_name: str) -> None:
        """Forward pass produces finite scalar loss for random input."""
        loss_fn = build_loss_function(loss_name)
        logits = _random_logits()
        labels = _tube_labels()
        loss = loss_fn(logits, labels)
        assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    @pytest.mark.parametrize("loss_name", SWEEP_LOSSES)
    def test_backward_non_zero_grads(self, loss_name: str) -> None:
        """Backward pass produces non-zero gradients."""
        loss_fn = build_loss_function(loss_name)
        logits = _random_logits()
        labels = _tube_labels()
        loss = loss_fn(logits, labels)
        loss.backward()
        assert logits.grad is not None, "No gradient computed"
        assert logits.grad.abs().sum() > 0, "All gradients are zero"

    @pytest.mark.parametrize("loss_name", SWEEP_LOSSES)
    def test_no_nan_on_all_zeros_pred(self, loss_name: str) -> None:
        """Loss does not NaN when prediction is all zeros."""
        loss_fn = build_loss_function(loss_name)
        logits = _all_zeros_logits()
        labels = _tube_labels()
        loss = loss_fn(logits, labels)
        assert not torch.isnan(loss), f"NaN loss on all-zeros pred: {loss.item()}"

    @pytest.mark.parametrize("loss_name", SWEEP_LOSSES)
    def test_no_nan_on_all_ones_pred(self, loss_name: str) -> None:
        """Loss does not NaN when prediction is all ones."""
        loss_fn = build_loss_function(loss_name)
        logits = _all_ones_logits()
        labels = _tube_labels()
        loss = loss_fn(logits, labels)
        assert not torch.isnan(loss), f"NaN loss on all-ones pred: {loss.item()}"

    @pytest.mark.parametrize("loss_name", SWEEP_LOSSES)
    def test_no_nan_on_perfect_pred(self, loss_name: str) -> None:
        """Loss does not NaN when prediction exactly matches GT."""
        loss_fn = build_loss_function(loss_name)
        labels = _tube_labels()
        logits = _perfect_logits(labels)
        loss = loss_fn(logits, labels)
        assert not torch.isnan(loss), f"NaN loss on perfect pred: {loss.item()}"


# ---------------------------------------------------------------------------
# Category 2: Value Sanity
# ---------------------------------------------------------------------------


class TestValueSanity:
    """Loss values must be bounded and sensible."""

    @pytest.mark.parametrize("loss_name", SWEEP_LOSSES)
    def test_perfect_pred_lower_than_random(self, loss_name: str) -> None:
        """Loss for perfect prediction < loss for random prediction."""
        loss_fn = build_loss_function(loss_name)
        labels = _tube_labels()

        # Perfect prediction
        perfect_logits = _perfect_logits(labels)
        perfect_loss = loss_fn(perfect_logits, labels).item()

        # Random prediction (average over 3 seeds for stability)
        random_losses = []
        for seed in range(3):
            torch.manual_seed(seed + 42)
            rand_logits = _random_logits()
            random_losses.append(loss_fn(rand_logits, labels).item())
        avg_random_loss = sum(random_losses) / len(random_losses)

        assert perfect_loss < avg_random_loss, (
            f"Perfect loss ({perfect_loss:.4f}) >= avg random loss ({avg_random_loss:.4f})"
        )

    @pytest.mark.parametrize("loss_name", SWEEP_LOSSES)
    def test_loss_is_non_negative(self, loss_name: str) -> None:
        """Loss value is >= 0."""
        loss_fn = build_loss_function(loss_name)
        logits = _random_logits()
        labels = _tube_labels()
        loss = loss_fn(logits, labels).item()
        assert loss >= -1e-6, f"Loss is negative: {loss}"

    @pytest.mark.parametrize("loss_name", SWEEP_LOSSES)
    def test_empty_mask_handled(self, loss_name: str) -> None:
        """Empty GT mask (all background) does not cause crash or NaN."""
        loss_fn = build_loss_function(loss_name)
        logits = _random_logits()
        labels = torch.zeros(B, 1, *SPATIAL)  # All background
        loss = loss_fn(logits, labels)
        assert not torch.isnan(loss), f"NaN on empty mask: {loss.item()}"
        assert torch.isfinite(loss), f"Inf on empty mask: {loss.item()}"


# ---------------------------------------------------------------------------
# Category 3: Convergence Smoke Test
# ---------------------------------------------------------------------------


class TestConvergenceSmoke:
    """Loss should decrease on a trivially overfittable task."""

    @pytest.mark.parametrize("loss_name", SWEEP_LOSSES)
    def test_overfit_single_sample(self, loss_name: str) -> None:
        """Train a tiny model for 20 steps; loss at step 20 < loss at step 1."""

        class TinyModel(nn.Module):  # type: ignore[misc]
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv3d(1, 8, 3, padding=1)
                self.relu = nn.ReLU()
                self.conv2 = nn.Conv3d(8, C, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.conv2(self.relu(self.conv1(x)))

        torch.manual_seed(42)
        model = TinyModel()
        loss_fn = build_loss_function(loss_name)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Synthetic input: single-channel volume
        x = torch.randn(B, 1, *SPATIAL)
        labels = _tube_labels()

        losses: list[float] = []
        for step in range(20):
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            # Safety: bail if NaN
            if math.isnan(loss.item()):
                pytest.fail(f"NaN at step {step}")

        # Loss should decrease: step 20 < step 1
        # Use max of first 3 steps vs min of last 3 for robustness
        early_loss = max(losses[:3])
        late_loss = min(losses[-3:])
        assert late_loss < early_loss, (
            f"No convergence: early={early_loss:.4f}, late={late_loss:.4f}"
        )


# ---------------------------------------------------------------------------
# Category 4: Warning System
# ---------------------------------------------------------------------------


class TestWarningSystem:
    """Verify that experimental losses emit warnings."""

    def test_experimental_loss_emits_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """EXPERIMENTAL losses emit a warning on first build."""
        from minivess.pipeline.loss_functions import _WARNED_LOSSES

        _WARNED_LOSSES.discard("betti")  # Reset for this test
        import logging

        with caplog.at_level(
            logging.WARNING, logger="minivess.pipeline.loss_functions"
        ):
            build_loss_function("betti")
        assert any("EXPERIMENTAL" in r.message for r in caplog.records)

    def test_library_loss_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """LIBRARY losses do NOT emit warnings."""
        import logging

        with caplog.at_level(
            logging.WARNING, logger="minivess.pipeline.loss_functions"
        ):
            build_loss_function("dice_ce")
        assert not any(
            "EXPERIMENTAL" in r.message or "HYBRID" in r.message for r in caplog.records
        )


# ---------------------------------------------------------------------------
# Category 5: Multi-Patch-Size Validation
# ---------------------------------------------------------------------------


class TestMultiPatchSizeFinite:
    """Verify all sweep losses produce finite values across patch sizes.

    Tests three spatial sizes based on production reality:
    - (8, 16, 16): current default
    - (5, 16, 16): MiniVess minimum Z dimension (critical edge case)
    - (16, 16, 16): comfortable positive control
    """

    @pytest.mark.parametrize("loss_name", SWEEP_LOSSES)
    @pytest.mark.parametrize(
        "spatial",
        SPATIAL_SIZES,
        ids=["8x16x16", "5x16x16", "16x16x16"],
    )
    def test_finite_across_patch_sizes(
        self, loss_name: str, spatial: tuple[int, int, int]
    ) -> None:
        """Loss must be finite across all production patch sizes."""
        torch.manual_seed(42)
        logits = torch.randn(B, C, *spatial, requires_grad=True)
        labels = torch.zeros(B, 1, *spatial)
        d, h, w = spatial
        labels[:, :, d // 4 : d // 2, h // 4 : h // 2, w // 4 : w // 2] = 1.0

        loss_fn = build_loss_function(loss_name)
        loss = loss_fn(logits, labels)
        assert torch.isfinite(loss), (
            f"{loss_name} not finite at spatial={spatial}: {loss.item()}"
        )
        loss.backward()
        assert logits.grad is not None, f"{loss_name}: no grad at spatial={spatial}"
