"""SAM3 mask decoder wrapper.

Wraps SAM3's mask prediction head for binary segmentation.
Supports two prompt modes:
    - Null: fully automatic segmentation (no prompts)
    - Concept: text embedding for concept-driven segmentation

The decoder takes 256-dim FPN features and predicts a single-channel
binary mask. ``binary_to_2class()`` converts to 2-class format for
cross-entropy loss compatibility.

IMPORTANT: Real pretrained SAM3 is required. No stub/fallback mode exists.
If SAM3 is not installed, RuntimeError is raised with install instructions.

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


class Sam3MaskDecoder(nn.Module):
    """SAM3 mask decoder wrapper for binary segmentation.

    Parameters
    ----------
    config:
        Model configuration.
    """

    def __init__(
        self,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self._config = config
        self.decoder: nn.Module = self._load_sam3_decoder()

    def _load_sam3_decoder(self) -> nn.Module:
        """Load real SAM3 mask decoder.

        Tries native sam3 package first.

        Raises
        ------
        RuntimeError
            If SAM3 package is not installed.
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
            logger.debug("Native sam3 package not found for decoder")

        msg = (
            "SAM3 package not installed — cannot load mask decoder.\n"
            "Install via:\n"
            "  git clone https://github.com/facebookresearch/sam3.git\n"
            "  cd sam3 && pip install -e .\n"
            "Or: uv add 'transformers>=4.50' for HuggingFace support."
        )
        raise RuntimeError(msg)

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
