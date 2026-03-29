"""Tests for SkeletonRecallLoss memory behavior.

T12 from double-check plan: verify skeleton is detached, no memory leak
across multiple forward passes, batch size scaling.
"""

from __future__ import annotations

import gc

import torch


def _make_pair(
    batch: int = 1,
    spatial: tuple[int, int, int] = (16, 16, 8),
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    logits = torch.randn(batch, 2, *spatial, requires_grad=True)
    labels = torch.randint(0, 2, (batch, 1, *spatial)).float()
    return logits, labels


class TestSkeletonRecallMemory:
    """SkeletonRecallLoss CPU-GPU transfer must not leak memory."""

    def test_skeleton_detached_no_graph_accumulation(self) -> None:
        from minivess.pipeline.vendored_losses.skeleton_recall import SkeletonRecallLoss

        loss_fn = SkeletonRecallLoss()
        logits, labels = _make_pair()

        # First forward + backward
        loss1 = loss_fn(logits, labels)
        loss1.backward()

        # Count tensors in gc
        gc.collect()
        tensors_after_first = sum(
            1 for obj in gc.get_objects() if isinstance(obj, torch.Tensor)
        )

        # Create fresh logits for second pass
        logits2 = torch.randn_like(logits.detach()).requires_grad_(True)
        loss2 = loss_fn(logits2, labels)
        loss2.backward()

        gc.collect()
        tensors_after_second = sum(
            1 for obj in gc.get_objects() if isinstance(obj, torch.Tensor)
        )

        # Tensor count should not grow significantly (allow small variance)
        growth = tensors_after_second - tensors_after_first
        assert growth < 50, (
            f"Tensor count grew by {growth} between forward passes — potential leak"
        )

    def test_multiple_forward_no_crash(self) -> None:
        from minivess.pipeline.vendored_losses.skeleton_recall import SkeletonRecallLoss

        loss_fn = SkeletonRecallLoss()
        for i in range(5):
            logits, labels = _make_pair()
            loss = loss_fn(logits, labels)
            assert torch.isfinite(loss), f"Iteration {i}: loss not finite"
            loss.backward()
            assert logits.grad is not None

    def test_batch_size_scaling(self) -> None:
        from minivess.pipeline.vendored_losses.skeleton_recall import SkeletonRecallLoss

        loss_fn = SkeletonRecallLoss()

        logits1, labels1 = _make_pair(batch=1)
        loss1 = loss_fn(logits1, labels1)
        assert torch.isfinite(loss1), f"Batch=1: {loss1.item()}"

        logits4, labels4 = _make_pair(batch=4)
        loss4 = loss_fn(logits4, labels4)
        assert torch.isfinite(loss4), f"Batch=4: {loss4.item()}"
