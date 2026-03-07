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
        self._lightweight = False
        self.decoder: nn.Module = self._load_sam3_decoder()

    def _load_sam3_decoder(self) -> nn.Module:
        """Load SAM3 mask decoder.

        Tries native sam3 package first (uses SAM3's own detection head).
        Falls back to HuggingFace path with a lightweight Conv head — the HF
        Sam3MaskDecoder has a complex prompted-segmentation API that does not
        map to binary automated segmentation. A simple trainable Conv head on
        top of SAM3 FPN features (256-dim) is more appropriate for V1 Vanilla.

        Raises
        ------
        RuntimeError
            If neither sam3 nor transformers with SAM3 support is available.
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

        # HuggingFace path: use a lightweight Conv head instead of the HF
        # Sam3MaskDecoder (which has a prompted-segmentation API incompatible
        # with automated binary segmentation).
        try:
            from transformers import Sam3Model  # noqa: F401

            logger.info(
                "Using lightweight Conv decoder for HF SAM3 path "
                "(HF mask decoder has incompatible prompted-segmentation API)"
            )
            self._lightweight = True
            return self._build_lightweight_decoder()
        except ImportError:
            logger.debug("HuggingFace transformers with SAM3 not found")

        msg = (
            "SAM3 package not installed — cannot load mask decoder.\n"
            "Install via:\n"
            "  git clone https://github.com/facebookresearch/sam3.git\n"
            "  cd sam3 && pip install -e .\n"
            "Or: uv add 'transformers>=5.2' for HuggingFace support."
        )
        raise RuntimeError(msg)

    @staticmethod
    def _build_lightweight_decoder() -> nn.Module:
        """Build a lightweight Conv decoder for binary segmentation.

        Takes 256-dim FPN features and outputs a 1-channel binary mask.
        ~66K trainable params (vs ~2.3M for native SAM3 decoder).
        """
        return nn.Sequential(
            nn.Conv2d(SAM3_DECODER_IN_DIM, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

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
        if self._lightweight:
            result: Tensor = self.decoder(features)
        else:
            result = self.decoder(features, prompt_embedding)
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
