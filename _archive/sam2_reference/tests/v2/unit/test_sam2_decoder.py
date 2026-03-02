"""Tests for SAM2 mask decoder wrapper (SAM-03).

Validates null-prompt mask decoding and binary-to-2class conversion.
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Test: Sam2MaskDecoder
# ---------------------------------------------------------------------------


class TestSam2MaskDecoder:
    """Sam2MaskDecoder wraps SAM2 decoder with null prompts."""

    def test_decoder_creates_without_error(self) -> None:
        from minivess.adapters.sam2_decoder import Sam2MaskDecoder

        decoder = Sam2MaskDecoder(embed_dim=96)
        assert decoder is not None

    def test_decoder_output_shape(self) -> None:
        from minivess.adapters.sam2_decoder import Sam2MaskDecoder

        decoder = Sam2MaskDecoder(embed_dim=96)
        # Simulate encoder features
        features = torch.randn(1, 96, 64, 64)
        with torch.no_grad():
            logits = decoder(features)
        # Output should be (B, 2, H_out, W_out) — 2 classes
        assert logits.ndim == 4
        assert logits.shape[0] == 1
        assert logits.shape[1] == 2

    def test_decoder_trainable_params_nonzero(self) -> None:
        from minivess.adapters.sam2_decoder import Sam2MaskDecoder

        decoder = Sam2MaskDecoder(embed_dim=96)
        n_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        assert n_params > 0

    def test_decoder_gradient_flow(self) -> None:
        from minivess.adapters.sam2_decoder import Sam2MaskDecoder

        decoder = Sam2MaskDecoder(embed_dim=96)
        features = torch.randn(1, 96, 32, 32, requires_grad=True)
        logits = decoder(features)
        loss = logits.sum()
        loss.backward()
        assert features.grad is not None


# ---------------------------------------------------------------------------
# Test: binary_to_2class
# ---------------------------------------------------------------------------


class TestBinaryTo2Class:
    """Convert binary SAM output to 2-class format."""

    def test_binary_to_2class_shape(self) -> None:
        from minivess.adapters.sam2_decoder import binary_to_2class

        binary_logits = torch.randn(2, 1, 64, 64)
        two_class = binary_to_2class(binary_logits)
        assert two_class.shape == (2, 2, 64, 64)

    def test_binary_to_2class_values(self) -> None:
        from minivess.adapters.sam2_decoder import binary_to_2class

        binary_logits = torch.tensor([[[[1.0, -1.0]]]])  # (1,1,1,2)
        two_class = binary_to_2class(binary_logits)
        # Background should be -logits, foreground should be +logits
        assert torch.allclose(two_class[0, 0], -binary_logits[0, 0])
        assert torch.allclose(two_class[0, 1], binary_logits[0, 0])
