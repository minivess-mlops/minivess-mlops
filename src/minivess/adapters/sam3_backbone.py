"""SAM3 backbone wrapper — ViT-32L Perception Encoder.

Wraps Meta's SAM3 (Segment Anything Model 3) perception encoder for use
as a frozen feature extractor in the MiniVess segmentation pipeline.

SAM3 Architecture:
    - Backbone: ViT-32L (1024-dim embeddings, 32 transformer blocks)
    - Input: 1008x1008 RGB, normalized mean=0.5, std=0.5
    - Feature output: 1024-dim from ViT, 256-dim from FPN neck
    - Patch size: 14 → 72×72 feature maps
    - Total params: ~848M (perception encoder ~648M)

Two loading modes:
    1. Native: ``build_sam3_image_model()`` from sam3 package (git clone)
    2. HuggingFace: ``Sam3Model.from_pretrained("facebook/sam3")``

Provides ``_StubSam3Encoder`` for testing without SAM3 installed.

References:
    - Ravi et al. (2025). "SAM 3." arXiv:2511.16719
    - github.com/facebookresearch/sam3
    - huggingface.co/facebook/sam3 (gated)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor, nn

if TYPE_CHECKING:
    from pathlib import Path

    from minivess.config.models import ModelConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SAM3 constants
# ---------------------------------------------------------------------------
SAM3_INPUT_SIZE: int = 1008
SAM3_HF_MODEL_ID: str = "facebook/sam3"
SAM3_CKPT_NAME: str = "sam3.pt"
SAM3_IMAGE_MEAN: tuple[float, float, float] = (0.5, 0.5, 0.5)
SAM3_IMAGE_STD: tuple[float, float, float] = (0.5, 0.5, 0.5)
SAM3_EMBED_DIM: int = 1024
SAM3_FPN_DIM: int = 256
SAM3_PATCH_SIZE: int = 14
SAM3_FEATURE_MAP_SIZE: int = SAM3_INPUT_SIZE // SAM3_PATCH_SIZE  # 72


# ---------------------------------------------------------------------------
# Stub encoder for testing without SAM3 package
# ---------------------------------------------------------------------------
class _StubSam3Encoder(nn.Module):  # type: ignore[misc]
    """Lightweight stub mimicking SAM3 ViT-32L output shape.

    Used for testing and CI where the real SAM3 package is not installed.
    Produces random features with the correct output dimensions.
    """

    def __init__(self, embed_dim: int = SAM3_EMBED_DIM) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.feature_map_size = SAM3_FEATURE_MAP_SIZE
        # Minimal conv to produce features with correct shape
        self.proj = nn.Conv2d(
            3, embed_dim, kernel_size=SAM3_PATCH_SIZE, stride=SAM3_PATCH_SIZE
        )

    def forward(self, x: Tensor) -> Tensor:
        """Extract stub features.

        Parameters
        ----------
        x:
            Input tensor. Accepts (B, 1, H, W) or (B, 3, H, W).
            Non-1008 inputs are resized.

        Returns
        -------
        Feature tensor of shape (B, embed_dim, H_feat, W_feat).
        """
        # Grayscale → 3-channel
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)

        # Resize to SAM3 input size if needed
        if x.shape[2] != SAM3_INPUT_SIZE or x.shape[3] != SAM3_INPUT_SIZE:
            x = F.interpolate(
                x,
                size=(SAM3_INPUT_SIZE, SAM3_INPUT_SIZE),
                mode="bilinear",
                align_corners=False,
            )

        # Project to feature space via patch embedding
        result: Tensor = self.proj(x)
        return result


# ---------------------------------------------------------------------------
# Stub FPN neck for testing
# ---------------------------------------------------------------------------
class _StubFPNNeck(nn.Module):  # type: ignore[misc]
    """Lightweight stub mimicking SAM3 FPN neck output."""

    def __init__(
        self,
        in_channels: int = SAM3_EMBED_DIM,
        out_channels: int = SAM3_FPN_DIM,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Project ViT features to FPN dimension."""
        result: Tensor = self.proj(x)
        return result


# ---------------------------------------------------------------------------
# Sam3Backbone
# ---------------------------------------------------------------------------
class Sam3Backbone(nn.Module):  # type: ignore[misc]
    """SAM3 perception encoder wrapper for feature extraction.

    Wraps either the real SAM3 ViT-32L encoder or a stub for testing.
    Provides methods for:
    - Single-image feature extraction (2D)
    - Volume feature extraction (slice-by-slice, 3D)
    - Feature caching to/from disk

    Parameters
    ----------
    config:
        Model configuration.
    use_stub:
        If True, use ``_StubSam3Encoder`` instead of real SAM3.
    freeze:
        If True (default), freeze all encoder parameters.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        use_stub: bool = False,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self._config = config
        self._embed_dim = SAM3_EMBED_DIM
        self._fpn_dim = SAM3_FPN_DIM

        self.encoder: nn.Module
        self.fpn_neck: nn.Module

        if use_stub:
            self.encoder = _StubSam3Encoder(embed_dim=self._embed_dim)
            self.fpn_neck = _StubFPNNeck(
                in_channels=self._embed_dim, out_channels=self._fpn_dim
            )
        else:
            self.encoder, self.fpn_neck = self._load_sam3_encoder()

        if freeze:
            self._freeze_encoder()

    @property
    def out_channels(self) -> int:
        """ViT backbone output dimension (1024)."""
        return self._embed_dim

    @property
    def fpn_channels(self) -> int:
        """FPN neck output dimension (256)."""
        return self._fpn_dim

    def _freeze_encoder(self) -> None:
        """Freeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.fpn_neck.parameters():
            param.requires_grad = False
        logger.info(
            "SAM3 encoder frozen (%d params)",
            sum(p.numel() for p in self.encoder.parameters()),
        )

    def _load_sam3_encoder(self) -> tuple[nn.Module, nn.Module]:
        """Load the real SAM3 perception encoder.

        Tries native sam3 package first, falls back to HuggingFace transformers.

        Returns
        -------
        Tuple of (encoder, fpn_neck) modules.

        Raises
        ------
        ImportError
            If neither sam3 nor transformers with SAM3 support is available.
        """
        try:
            from sam3.model.model_builder import build_sam3_image_model

            logger.info("Loading SAM3 via native sam3 package")
            model = build_sam3_image_model(
                device="cpu",
                eval_mode=True,
                load_from_HF=True,
            )
            # Extract the perception encoder (ViT backbone) from the full model
            # SAM3 model has detector.backbone.visual as the ViT
            encoder = model.detector.backbone.visual
            fpn_neck = model.detector.backbone  # neck is part of backbone
            return encoder, fpn_neck
        except ImportError:
            pass

        try:
            from transformers import Sam3Model

            logger.info("Loading SAM3 via HuggingFace transformers")
            model = Sam3Model.from_pretrained(SAM3_HF_MODEL_ID)
            # Extract perception encoder from HF model
            encoder = model.vision_encoder
            fpn_neck = model.neck if hasattr(model, "neck") else _StubFPNNeck()
            return encoder, fpn_neck
        except ImportError:
            pass

        msg = (
            "SAM3 package not available. Install via:\n"
            "  git clone https://github.com/facebookresearch/sam3.git\n"
            "  cd sam3 && pip install -e .\n"
            "Or install transformers>=4.48 with SAM3 support."
        )
        raise ImportError(msg)

    def _preprocess(self, x: Tensor) -> Tensor:
        """Preprocess input for SAM3.

        Handles grayscale→3ch expansion, resize to 1008, and normalization.
        """
        # Grayscale → 3-channel
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)

        # Resize to SAM3 input size
        if x.shape[2] != SAM3_INPUT_SIZE or x.shape[3] != SAM3_INPUT_SIZE:
            x = F.interpolate(
                x,
                size=(SAM3_INPUT_SIZE, SAM3_INPUT_SIZE),
                mode="bilinear",
                align_corners=False,
            )

        # Normalize: mean=0.5, std=0.5 per channel
        mean = torch.tensor(SAM3_IMAGE_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(SAM3_IMAGE_STD, device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    def extract_features(self, x: Tensor) -> Tensor:
        """Extract ViT backbone features (1024-dim).

        Parameters
        ----------
        x:
            Input tensor of shape (B, C, H, W) where C is 1 or 3.

        Returns
        -------
        Feature tensor of shape (B, 1024, H_feat, W_feat).
        """
        x = self._preprocess(x)
        with torch.no_grad():
            result: Tensor = self.encoder(x)
            return result

    def extract_fpn_features(self, x: Tensor) -> Tensor:
        """Extract FPN neck features (256-dim).

        Parameters
        ----------
        x:
            Input tensor of shape (B, C, H, W).

        Returns
        -------
        Feature tensor of shape (B, 256, H_feat, W_feat).
        """
        vit_features = self.extract_features(x)
        with torch.no_grad():
            result: Tensor = self.fpn_neck(vit_features)
            return result

    def get_volume_embeddings(self, volume: Tensor) -> Tensor:
        """Extract features for all Z-slices of a 3D volume.

        Parameters
        ----------
        volume:
            3D volume of shape (B, C, D, H, W).

        Returns
        -------
        Features of shape (B, embed_dim, D, H_feat, W_feat).
        """
        b, c, d, h, w = volume.shape
        slice_features: list[Tensor] = []

        for z_idx in range(d):
            slice_2d = volume[:, :, z_idx, :, :]  # (B, C, H, W)
            features = self.extract_features(slice_2d)  # (B, 1024, H_f, W_f)
            slice_features.append(features)

        # Stack along depth: (B, 1024, D, H_f, W_f)
        return torch.stack(slice_features, dim=2)

    @staticmethod
    def save_cached_features(features: Tensor, path: Path) -> None:
        """Save extracted features to disk.

        Parameters
        ----------
        features:
            Feature tensor to cache.
        path:
            Destination file path (.pt).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(features, path)
        logger.debug(
            "Saved cached features to %s (%.1f MB)", path, path.stat().st_size / 1e6
        )

    @staticmethod
    def load_cached_features(path: Path) -> Tensor:
        """Load cached features from disk.

        Parameters
        ----------
        path:
            Source file path (.pt).

        Returns
        -------
        Loaded feature tensor.
        """
        result: Tensor = torch.load(path, weights_only=True)
        return result
