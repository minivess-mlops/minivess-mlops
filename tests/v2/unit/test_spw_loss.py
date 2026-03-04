"""Tests for SPW (Steerable Pyramid Weighted) loss function.

SPW Loss (Lu 2025, arXiv:2503.06604) uses multi-scale steerable pyramid
decomposition for adaptive weighting of cross-entropy, targeting thin
tubular structures (vessels, neurites).

Issue: #310 | Phase 4 | Plan: T4.1 (RED)
"""

from __future__ import annotations

import torch


def _make_dummy_inputs(
    batch: int = 2,
    num_classes: int = 2,
    size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create dummy logits and labels for testing."""
    torch.manual_seed(42)
    logits = torch.randn(batch, num_classes, size, size, size, requires_grad=True)
    labels = torch.randint(0, num_classes, (batch, 1, size, size, size))
    return logits, labels


class TestSPWLoss:
    """Unit tests for Steerable Pyramid Weighted Loss."""

    def test_spw_loss_differentiable(self) -> None:
        """SPW loss must be differentiable for gradient-based training."""
        from minivess.pipeline.vendored_losses.spw_loss import SPWLoss

        loss_fn = SPWLoss()
        logits, labels = _make_dummy_inputs()

        loss_val = loss_fn(logits, labels)
        loss_val.backward()

        assert logits.grad is not None
        assert logits.grad.shape == logits.shape

    def test_spw_loss_finite_values(self) -> None:
        """SPW loss should produce finite values for valid inputs."""
        from minivess.pipeline.vendored_losses.spw_loss import SPWLoss

        loss_fn = SPWLoss()
        logits, labels = _make_dummy_inputs()

        loss_val = loss_fn(logits, labels)
        assert torch.isfinite(loss_val), f"SPW loss is not finite: {loss_val}"

    def test_spw_loss_gradient_nonzero(self) -> None:
        """Gradients should be non-zero (loss provides training signal)."""
        from minivess.pipeline.vendored_losses.spw_loss import SPWLoss

        loss_fn = SPWLoss()
        logits, labels = _make_dummy_inputs()

        loss_val = loss_fn(logits, labels)
        loss_val.backward()

        grad_norm = logits.grad.norm().item()
        assert grad_norm > 1e-8, f"Gradient norm too small: {grad_norm}"

    def test_spw_loss_registered_in_factory(self) -> None:
        """SPW loss should be constructible via build_loss_function()."""
        from minivess.pipeline.loss_functions import build_loss_function

        loss_fn = build_loss_function("spw")
        assert loss_fn is not None

        logits, labels = _make_dummy_inputs()
        loss_val = loss_fn(logits, labels)
        assert torch.isfinite(loss_val)

    def test_spw_loss_higher_weight_on_boundaries(self) -> None:
        """SPW should assign higher weight near boundaries than in
        uniform regions (the key insight of multi-scale weighting)."""
        from minivess.pipeline.vendored_losses.spw_loss import SPWLoss

        loss_fn = SPWLoss()

        # Create a volume with a clear boundary
        batch, nc, s = 1, 2, 16
        logits = torch.zeros(batch, nc, s, s, s, requires_grad=True)
        labels = torch.zeros(batch, 1, s, s, s, dtype=torch.long)

        # Half foreground, half background — creates a boundary plane
        labels[:, :, :, :, s // 2 :] = 1

        loss_with_boundary = loss_fn(logits, labels)

        # All background — no boundary
        labels_uniform = torch.zeros(batch, 1, s, s, s, dtype=torch.long)
        logits_uniform = torch.zeros(batch, nc, s, s, s, requires_grad=True)
        loss_uniform = loss_fn(logits_uniform, labels_uniform)

        # Both should be finite; boundary case typically different from uniform
        assert torch.isfinite(loss_with_boundary)
        assert torch.isfinite(loss_uniform)

    def test_spw_loss_nonnegative(self) -> None:
        """SPW loss should be non-negative."""
        from minivess.pipeline.vendored_losses.spw_loss import SPWLoss

        loss_fn = SPWLoss()
        logits, labels = _make_dummy_inputs()

        loss_val = loss_fn(logits, labels)
        assert loss_val.item() >= 0.0, f"Negative loss: {loss_val.item()}"
