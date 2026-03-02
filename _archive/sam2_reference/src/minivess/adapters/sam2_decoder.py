"""SAM2 mask decoder wrapper with null-prompt mode.

Provides a lightweight mask decoder that takes encoder features and
produces binary segmentation logits. Uses null (learned) prompt
embeddings for fully automatic segmentation (no user prompts).

The decoder uses transposed convolutions to upsample from encoder
feature resolution back to a target spatial size.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def binary_to_2class(binary_logits: Tensor) -> Tensor:
    """Convert single-channel binary logits to 2-class format.

    Follows the VesselFMAdapter pattern: ``torch.cat([-logits, logits], dim=1)``.

    Parameters
    ----------
    binary_logits:
        Tensor of shape (B, 1, ...) with raw logits.

    Returns
    -------
    Tensor of shape (B, 2, ...) where channel 0 is background (-logits)
    and channel 1 is foreground (+logits).
    """
    return torch.cat([-binary_logits, binary_logits], dim=1)


class Sam2MaskDecoder(nn.Module):
    """Lightweight mask decoder with null prompt embeddings.

    Takes encoder features of shape (B, embed_dim, H', W') and
    produces 2-class logits of shape (B, 2, H_out, W_out).

    Parameters
    ----------
    embed_dim:
        Number of channels in the encoder features.
    hidden_dim:
        Hidden dimension for decoder layers.
    num_upsample:
        Number of 2x transposed-conv upsample layers.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 128,
        num_upsample: int = 2,
    ) -> None:
        super().__init__()

        # Learned null prompt embedding (replaces user click/box prompts)
        self.null_prompt = nn.Parameter(torch.randn(1, embed_dim, 1, 1) * 0.02)

        # Projection from encoder features + prompt to hidden space
        self.input_proj = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
        )

        # Upsample path
        upsample_layers: list[nn.Module] = []
        for _ in range(num_upsample):
            upsample_layers.extend(
                [
                    nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2),
                    nn.GELU(),
                ]
            )
        self.upsample = nn.Sequential(*upsample_layers)

        # Final 1x1 conv to binary logit
        self.head = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, features: Tensor) -> Tensor:
        """Decode encoder features into 2-class segmentation logits.

        Parameters
        ----------
        features:
            Encoder output of shape (B, embed_dim, H', W').

        Returns
        -------
        Logits tensor of shape (B, 2, H_out, W_out) where
        H_out = H' * 2^num_upsample, W_out = W' * 2^num_upsample.
        """
        # Add null prompt embedding (broadcast over spatial dims)
        b = features.shape[0]
        prompt = self.null_prompt.expand(b, -1, features.shape[2], features.shape[3])
        x = features + prompt

        x = self.input_proj(x)
        x = self.upsample(x)
        binary_logits = self.head(x)

        return binary_to_2class(binary_logits)
