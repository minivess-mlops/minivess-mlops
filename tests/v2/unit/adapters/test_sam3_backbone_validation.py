"""Tests for sam3_backbone input shape validation.

T0.2: Verify _preprocess() rejects non-4D input with helpful error.
"""

from __future__ import annotations

import pytest
import torch

# Mock the SAM3 import since we don't have weights in test env
_sam3_skip = pytest.mark.skipif(
    True,  # Always skip actual backbone tests — need HF weights
    reason="Sam3Backbone requires HF weights; test validation logic only",
)


class TestPreprocessInputValidation:
    """_preprocess() should validate input is exactly 4D (B,C,H,W)."""

    def test_rejects_3d_input(self) -> None:
        """3D tensor (missing batch dim) should raise ValueError."""
        from minivess.adapters.sam3_backbone import Sam3Backbone

        # Create a mock backbone to test _preprocess without HF weights
        backbone = object.__new__(Sam3Backbone)
        backbone._input_size = 1008

        x = torch.randn(3, 64, 64)  # 3D — wrong
        with pytest.raises(ValueError, match="Expected 4D"):
            backbone._preprocess(x)

    def test_rejects_5d_input(self) -> None:
        """5D tensor (full 3D volume) should raise ValueError."""
        from minivess.adapters.sam3_backbone import Sam3Backbone

        backbone = object.__new__(Sam3Backbone)
        backbone._input_size = 1008

        x = torch.randn(1, 1, 3, 64, 64)  # 5D — wrong
        with pytest.raises(ValueError, match="Expected 4D"):
            backbone._preprocess(x)

    def test_accepts_4d_input(self) -> None:
        """4D tensor (B,C,H,W) should pass validation."""
        from minivess.adapters.sam3_backbone import Sam3Backbone

        backbone = object.__new__(Sam3Backbone)
        backbone._input_size = 64  # Small for test speed

        x = torch.randn(1, 1, 64, 64)  # 4D — correct
        # Should not raise — but will resize and normalize
        result = backbone._preprocess(x)
        assert result.ndim == 4
