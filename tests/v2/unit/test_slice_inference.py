"""Tests for slice-by-slice inference utility (SAM-02).

Validates slice iteration, SAM-compatible resizing, and volume reassembly.
"""

from __future__ import annotations

import torch
from torch import nn

# ---------------------------------------------------------------------------
# Test: resize_for_sam / unresize_from_sam
# ---------------------------------------------------------------------------


class TestSamResizing:
    """SAM expects 1024x1024; verify resize/unresize round-trip."""

    def test_resize_for_sam_output_size(self) -> None:
        from minivess.adapters.slice_inference import resize_for_sam

        image_2d = torch.randn(1, 1, 512, 512)
        resized = resize_for_sam(image_2d, target_size=1024)
        assert resized.shape == (1, 1, 1024, 1024)

    def test_unresize_from_sam_restores_shape(self) -> None:
        from minivess.adapters.slice_inference import unresize_from_sam

        sam_output = torch.randn(1, 2, 1024, 1024)
        restored = unresize_from_sam(sam_output, original_h=512, original_w=512)
        assert restored.shape == (1, 2, 512, 512)

    def test_resize_unresize_roundtrip(self) -> None:
        from minivess.adapters.slice_inference import resize_for_sam, unresize_from_sam

        image = torch.randn(1, 1, 256, 256)
        resize_for_sam(image, target_size=1024)  # verify no errors
        # Simulate model output with 2 classes
        model_out = torch.randn(1, 2, 1024, 1024)
        restored = unresize_from_sam(model_out, original_h=256, original_w=256)
        assert restored.shape == (1, 2, 256, 256)

    def test_resize_non_square_input(self) -> None:
        from minivess.adapters.slice_inference import resize_for_sam

        image_2d = torch.randn(1, 1, 300, 400)
        resized = resize_for_sam(image_2d, target_size=1024)
        assert resized.shape == (1, 1, 1024, 1024)

    def test_resize_for_sam3_1008(self) -> None:
        """SAM3 uses 1008x1008 input (patch_size=14 → 72×72 features)."""
        from minivess.adapters.slice_inference import resize_for_sam

        image_2d = torch.randn(1, 1, 512, 512)
        resized = resize_for_sam(image_2d, target_size=1008)
        assert resized.shape == (1, 1, 1008, 1008)


# ---------------------------------------------------------------------------
# Test: slice_by_slice_forward
# ---------------------------------------------------------------------------


class _DummyModel2D(nn.Module):  # type: ignore[misc]
    """Minimal 2D model that outputs (B, 2, H, W)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        return torch.randn(b, 2, h, w)


class TestSliceBySliceForward:
    """slice_by_slice_forward iterates Z-slices through a 2D model."""

    def test_output_shape_matches_volume(self) -> None:
        from minivess.adapters.slice_inference import slice_by_slice_forward

        model_2d = _DummyModel2D()
        volume = torch.randn(1, 1, 8, 64, 64)  # (B, C, D, H, W)
        output = slice_by_slice_forward(model_2d, volume)
        assert output.shape == (1, 2, 8, 64, 64)

    def test_each_slice_processed_independently(self) -> None:
        from minivess.adapters.slice_inference import slice_by_slice_forward

        call_count = {"n": 0}

        class _CountingModel(nn.Module):  # type: ignore[misc]
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                call_count["n"] += 1
                b, c, h, w = x.shape
                return torch.zeros(b, 2, h, w)

        model = _CountingModel()
        volume = torch.randn(1, 1, 5, 32, 32)
        slice_by_slice_forward(model, volume)
        assert call_count["n"] == 5  # One call per Z-slice

    def test_gradient_flows_through_slices(self) -> None:
        from minivess.adapters.slice_inference import slice_by_slice_forward

        class _LearnableModel(nn.Module):  # type: ignore[misc]
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.ones(1))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * self.weight

        model = _LearnableModel()
        volume = torch.randn(1, 1, 3, 16, 16, requires_grad=True)
        output = slice_by_slice_forward(model, volume)
        loss = output.sum()
        loss.backward()
        assert model.weight.grad is not None

    def test_batch_size_preserved(self) -> None:
        from minivess.adapters.slice_inference import slice_by_slice_forward

        model_2d = _DummyModel2D()
        volume = torch.randn(2, 1, 4, 32, 32)
        output = slice_by_slice_forward(model_2d, volume)
        assert output.shape[0] == 2
