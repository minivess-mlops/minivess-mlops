"""Tests for SkeletonRecallLoss edge cases.

Verifies that the loss warns when skeleton degenerates and falls back
to full foreground mask (changing semantics from skeleton-recall to
plain-recall).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import pytest


class TestSkeletonRecallEdgeCases:
    """SkeletonRecallLoss must warn on degenerate skeleton."""

    def test_thin_structure_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """2x2-wide tube produces empty 3D skeleton → must log warning."""
        from minivess.pipeline.vendored_losses.skeleton_recall import SkeletonRecallLoss

        loss_fn = SkeletonRecallLoss(softmax=True)

        # 2x2x16 tube: skeletonize returns empty in 3D (Lee94 collapses it)
        logits = torch.randn(1, 2, 16, 16, 16)
        labels = torch.zeros(1, 1, 16, 16, 16)
        labels[:, :, :, 7:9, 7:9] = 1.0  # 2x2 cross-section tube

        with caplog.at_level(logging.WARNING):
            loss = loss_fn(logits, labels)

        assert torch.isfinite(loss), f"Loss is not finite: {loss}"
        assert any(
            "skeleton" in r.message.lower() and "falling back" in r.message.lower()
            for r in caplog.records
        ), (
            "SkeletonRecallLoss must warn when skeleton is empty and falls back "
            f"to full foreground mask. Got logs: {[r.message for r in caplog.records]}"
        )

    def test_normal_structure_no_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Thick cylinder should not trigger the warning."""
        from minivess.pipeline.vendored_losses.skeleton_recall import SkeletonRecallLoss

        loss_fn = SkeletonRecallLoss(softmax=True)

        # Create a thick cylinder (radius 5 voxels) that has a clear skeleton
        logits = torch.randn(1, 2, 32, 32, 32)
        labels = torch.zeros(1, 1, 32, 32, 32)
        for d in range(32):
            for h in range(32):
                for w in range(32):
                    if (h - 16) ** 2 + (w - 16) ** 2 < 25:  # r=5
                        labels[:, :, d, h, w] = 1.0

        with caplog.at_level(logging.WARNING):
            loss = loss_fn(logits, labels)

        assert torch.isfinite(loss)
        skeleton_warnings = [
            r
            for r in caplog.records
            if "skeleton" in r.message.lower() and "falling back" in r.message.lower()
        ]
        assert len(skeleton_warnings) == 0, (
            f"No skeleton fallback warning expected for thick structure, got: "
            f"{[r.message for r in skeleton_warnings]}"
        )
