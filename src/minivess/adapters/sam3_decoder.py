"""SAM3 mask decoder wrapper.

Wraps SAM3's mask prediction head for binary segmentation.
Supports two prompt modes:
    - Null: fully automatic segmentation (no prompts)
    - Concept: text embedding for concept-driven segmentation

The decoder takes 256-dim FPN features and predicts a single-channel
binary mask. ``binary_to_2class()`` converts to 2-class format for
cross-entropy loss compatibility.

References:
    - Ravi et al. (2025). "SAM 3." arXiv:2511.16719
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from minivess.config.models import ModelConfig

logger = logging.getLogger(__name__)

# Decoder input dimension (matches FPN neck output)
SAM3_DECODER_IN_DIM: int = 256


class _StubSam3Decoder(nn.Module):
    """Lightweight stub mimicking SAM3 mask decoder output.

    Used for testing and CI where the real SAM3 package is not installed.
    """

    def __init__(self, in_channels: int = SAM3_DECODER_IN_DIM) -> None:
        super().__init__()
        # Simple conv stack to produce single-channel mask logits
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, features: Tensor, prompt: Tensor | None = None) -> Tensor:
        """Predict mask logits from FPN features.

        Parameters
        ----------
        features:
            FPN features of shape (B, 256, H, W).
        prompt:
            Optional prompt embedding (ignored in stub).

        Returns
        -------
        Mask logits of shape (B, 1, H, W).
        """
        if prompt is not None:
            # Broadcast prompt and add to features as channel-wise bias
            prompt_bias = prompt.unsqueeze(-1).unsqueeze(-1)  # (B, 256, 1, 1)
            # Project prompt to match feature channels if needed
            if prompt_bias.shape[1] != features.shape[1]:
                prompt_bias = prompt_bias[:, : features.shape[1]]
            features = features + prompt_bias

        result: Tensor = self.decoder(features)
        return result


class Sam3MaskDecoder(nn.Module):
    """SAM3 mask decoder wrapper for binary segmentation.

    Parameters
    ----------
    config:
        Model configuration.
    use_stub:
        If True, use ``_StubSam3Decoder`` instead of real SAM3 decoder.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        use_stub: bool = False,
    ) -> None:
        super().__init__()
        self._config = config

        if use_stub:
            self.decoder: nn.Module = _StubSam3Decoder()
        else:
            self.decoder = self._load_sam3_decoder()

    def _load_sam3_decoder(self) -> nn.Module:
        """Load real SAM3 mask decoder.

        Raises
        ------
        ImportError
            If SAM3 package is not available.
        """
        try:
            from sam3.model.model_builder import build_sam3_image_model

            logger.info("Loading SAM3 decoder via native sam3 package")
            model = build_sam3_image_model(
                device="cpu",
                eval_mode=True,
                load_from_HF=True,
            )
            decoder: nn.Module = model.detector.head
            return decoder
        except ImportError:
            pass

        msg = (
            "SAM3 package not available for decoder loading. Install via:\n"
            "  git clone https://github.com/facebookresearch/sam3.git\n"
            "  cd sam3 && pip install -e ."
        )
        raise ImportError(msg)

    def forward(
        self,
        features: Tensor,
        prompt_embedding: Tensor | None = None,
    ) -> Tensor:
        """Predict binary mask logits from FPN features.

        Parameters
        ----------
        features:
            FPN features of shape (B, 256, H, W).
        prompt_embedding:
            Optional concept prompt embedding of shape (B, dim).
            None = null prompt (automatic mode).

        Returns
        -------
        Binary mask logits of shape (B, 1, H, W).
        """
        result: Tensor = self.decoder(features, prompt_embedding)
        return result

    @staticmethod
    def binary_to_2class(logits: Tensor) -> Tensor:
        """Convert single-channel binary logits to 2-class format.

        Follows VesselFMAdapter pattern: ``torch.cat([-logits, logits], dim=1)``.

        Parameters
        ----------
        logits:
            Binary mask logits of shape (B, 1, H, W).

        Returns
        -------
        Two-class logits of shape (B, 2, H, W) where channel 0 is background
        and channel 1 is foreground.
        """
        return torch.cat([-logits, logits], dim=1)
