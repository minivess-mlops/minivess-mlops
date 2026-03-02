"""Tests for SAM3 mask decoder wrapper (T3).

Validates Sam3MaskDecoder: null prompt mode, concept prompt mode,
binary-to-2class conversion, gradient flow.
"""

from __future__ import annotations

import pytest
import torch

from minivess.config.models import ModelConfig, ModelFamily


@pytest.fixture()
def sam3_config() -> ModelConfig:
    """Minimal SAM3 config for testing."""
    return ModelConfig(
        family=ModelFamily.SAM3_VANILLA,
        name="sam3-decoder-test",
        in_channels=1,
        out_channels=2,
    )


class TestSam3MaskDecoder:
    """Sam3MaskDecoder wraps SAM3 decoder head for mask prediction."""

    def test_decoder_creates(self, sam3_config: ModelConfig) -> None:
        from minivess.adapters.sam3_decoder import Sam3MaskDecoder

        decoder = Sam3MaskDecoder(config=sam3_config, use_stub=True)
        assert decoder is not None

    def test_decoder_output_shape_matches_input(self, sam3_config: ModelConfig) -> None:
        """Output spatial dims match input (after resize back)."""
        from minivess.adapters.sam3_decoder import Sam3MaskDecoder

        decoder = Sam3MaskDecoder(config=sam3_config, use_stub=True)
        # FPN features: (B, 256, H_feat, W_feat)
        features = torch.randn(1, 256, 72, 72)
        output = decoder(features)
        # Output should be (B, 1, H_feat, W_feat) for binary mask
        assert output.shape[0] == 1
        assert output.shape[1] == 1  # single-channel binary logits
        assert output.shape[2] == 72
        assert output.shape[3] == 72

    def test_decoder_gradient_flows(self, sam3_config: ModelConfig) -> None:
        """Gradients flow through decoder during backward pass."""
        from minivess.adapters.sam3_decoder import Sam3MaskDecoder

        decoder = Sam3MaskDecoder(config=sam3_config, use_stub=True)
        features = torch.randn(1, 256, 72, 72)
        output = decoder(features)
        loss = output.sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in decoder.parameters()
        )
        assert has_grad, "Gradients must flow through decoder"

    def test_decoder_null_prompt_mode(self, sam3_config: ModelConfig) -> None:
        """Works with null embeddings for automatic segmentation."""
        from minivess.adapters.sam3_decoder import Sam3MaskDecoder

        decoder = Sam3MaskDecoder(config=sam3_config, use_stub=True)
        features = torch.randn(1, 256, 72, 72)
        # Null prompt = no prompt embeddings, automatic mode
        output = decoder(features, prompt_embedding=None)
        assert output.shape[0] == 1

    def test_decoder_concept_prompt_mode(self, sam3_config: ModelConfig) -> None:
        """Works with concept text prompt embedding."""
        from minivess.adapters.sam3_decoder import Sam3MaskDecoder

        decoder = Sam3MaskDecoder(config=sam3_config, use_stub=True)
        features = torch.randn(1, 256, 72, 72)
        # Simulate concept prompt embedding (e.g., "microvasculature")
        prompt = torch.randn(1, 256)
        output = decoder(features, prompt_embedding=prompt)
        assert output.shape[0] == 1


class TestBinaryTo2Class:
    """binary_to_2class() converts single-channel logits to 2-class."""

    def test_binary_to_2class_shape(self, sam3_config: ModelConfig) -> None:
        from minivess.adapters.sam3_decoder import Sam3MaskDecoder

        decoder = Sam3MaskDecoder(config=sam3_config, use_stub=True)
        logits = torch.randn(1, 1, 72, 72)
        two_class = decoder.binary_to_2class(logits)
        assert two_class.shape == (1, 2, 72, 72)

    def test_binary_to_2class_symmetry(self, sam3_config: ModelConfig) -> None:
        """Output is [-logits, logits] concatenated along channel dim."""
        from minivess.adapters.sam3_decoder import Sam3MaskDecoder

        decoder = Sam3MaskDecoder(config=sam3_config, use_stub=True)
        logits = torch.randn(1, 1, 8, 8)
        two_class = decoder.binary_to_2class(logits)
        # Channel 0 should be -logits, channel 1 should be logits
        assert torch.allclose(two_class[:, 0:1], -logits, atol=1e-6)
        assert torch.allclose(two_class[:, 1:2], logits, atol=1e-6)
