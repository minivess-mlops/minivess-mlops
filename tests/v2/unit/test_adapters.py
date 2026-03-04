from __future__ import annotations

import torch

from minivess.adapters.base import SegmentationOutput


class TestSegmentationOutput:
    """Test SegmentationOutput dataclass."""

    def test_creation(self) -> None:
        out = SegmentationOutput(
            prediction=torch.randn(1, 2, 8, 8, 4),
            logits=torch.randn(1, 2, 8, 8, 4),
        )
        assert out.prediction.shape == (1, 2, 8, 8, 4)
        assert out.metadata == {}

    def test_with_metadata(self) -> None:
        out = SegmentationOutput(
            prediction=torch.randn(1, 2, 8, 8, 4),
            logits=torch.randn(1, 2, 8, 8, 4),
            metadata={"architecture": "test"},
        )
        assert out.metadata["architecture"] == "test"

    def test_logits_shape_matches_prediction(self) -> None:
        shape = (2, 3, 16, 16, 8)
        out = SegmentationOutput(
            prediction=torch.randn(*shape),
            logits=torch.randn(*shape),
        )
        assert out.logits.shape == out.prediction.shape

    def test_empty_metadata_is_independent_per_instance(self) -> None:
        """Default metadata dicts should not be shared across instances."""
        out1 = SegmentationOutput(
            prediction=torch.zeros(1, 1, 1, 1, 1),
            logits=torch.zeros(1, 1, 1, 1, 1),
        )
        out2 = SegmentationOutput(
            prediction=torch.zeros(1, 1, 1, 1, 1),
            logits=torch.zeros(1, 1, 1, 1, 1),
        )
        out1.metadata["key"] = "value"
        assert "key" not in out2.metadata
