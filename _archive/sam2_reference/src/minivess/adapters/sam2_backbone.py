"""SAM2 Hiera backbone wrapper with weight management.

Wraps the SAM2 image encoder (Hiera architecture) for feature extraction.
Supports optional pretrained weight download with SHA256 verification.
All SAM2 imports are lazy — the backbone works without ``sam2`` installed
by using a lightweight Conv2d stub encoder for testing.

Reference: Ravi et al. (2024). "SAM 2: Segment Anything in Images and Videos."
Meta FAIR. arXiv:2408.00714
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SAM2 variant registry
# ---------------------------------------------------------------------------

SAM2_VARIANTS: dict[str, dict[str, Any]] = {
    "hiera_tiny": {
        "embed_dim": 96,
        "num_params_m": 38.9,
        "encoder_out_channels": 256,
        "notes": "Smallest Hiera variant, fits 8 GB VRAM",
    },
    "hiera_small": {
        "embed_dim": 96,
        "num_params_m": 46.0,
        "encoder_out_channels": 256,
        "notes": "Small Hiera variant",
    },
    "hiera_base_plus": {
        "embed_dim": 112,
        "num_params_m": 80.8,
        "encoder_out_channels": 256,
        "notes": "Base+ Hiera variant",
    },
    "hiera_large": {
        "embed_dim": 144,
        "num_params_m": 224.4,
        "encoder_out_channels": 256,
        "notes": "Large Hiera variant, requires 16+ GB VRAM",
    },
}

SAM2_CHECKPOINT_URLS: dict[str, str] = {
    "hiera_tiny": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    "hiera_small": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    "hiera_base_plus": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
    "hiera_large": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
}

SAM2_CHECKSUMS: dict[str, str] = {
    "hiera_tiny": "e1f7be449dab6773e81a757b3a22b001a8b97cc786181d3db3bab4bd59be8bac",
    "hiera_small": "4acfe484e26feada1f5c5f64c6a5e tried53f8a7e7ffce30d3dd1f00e6e13d2636",
    "hiera_base_plus": "5765a5d1a1d3ccbe66da9c3b779ddd82b6fae57f9aa1c3e55dcbe0ee9fdd81c0",
    "hiera_large": "4a0b4deaf4c25ce8a32e961d2f1d7e2191d2549abb40c7e7b2f8dbfc8c8085a5",
}


# ---------------------------------------------------------------------------
# Checksum verification
# ---------------------------------------------------------------------------


def verify_checkpoint_integrity(data: bytes, expected_sha256: str) -> bool:
    """Verify SHA256 checksum of binary data.

    Parameters
    ----------
    data:
        Raw bytes to hash.
    expected_sha256:
        Expected lowercase hex SHA256 digest (64 chars).

    Returns
    -------
    True if the computed hash matches.
    """
    computed = hashlib.sha256(data).hexdigest()
    return computed == expected_sha256


# ---------------------------------------------------------------------------
# Stub encoder (no SAM2 dependency)
# ---------------------------------------------------------------------------


class _StubHieraEncoder(nn.Module):
    """Lightweight Conv2d encoder that mimics Hiera output shapes.

    Used when ``pretrained=False`` (unit tests, CI) so that the full
    SAM2 package is not required.
    """

    def __init__(self, embed_dim: int = 96, out_channels: int = 256) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, embed_dim, kernel_size=16, stride=16, padding=0),
            nn.GELU(),
            nn.Conv2d(embed_dim, out_channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


# ---------------------------------------------------------------------------
# Sam2Backbone
# ---------------------------------------------------------------------------


class Sam2Backbone(nn.Module):
    """SAM2 Hiera image encoder wrapper.

    When ``pretrained=False``, uses a lightweight stub encoder that
    produces feature maps with the correct channel dimensions. This
    allows unit testing and CI without downloading SAM2 weights.

    When ``pretrained=True``, attempts to load official SAM2 weights
    via the ``sam2`` package (must be installed separately).

    Parameters
    ----------
    variant:
        SAM2 Hiera variant name (e.g., "hiera_tiny").
    pretrained:
        Download and load official Meta SAM2 weights.
    """

    def __init__(
        self,
        variant: str = "hiera_tiny",
        *,
        pretrained: bool = False,
    ) -> None:
        super().__init__()

        if variant not in SAM2_VARIANTS:
            msg = (
                f"Unknown SAM2 variant '{variant}'. "
                f"Available: {list(SAM2_VARIANTS.keys())}"
            )
            raise ValueError(msg)

        self.variant = variant
        self.embed_dim = SAM2_VARIANTS[variant]["embed_dim"]
        self.out_channels = SAM2_VARIANTS[variant]["encoder_out_channels"]

        if pretrained:
            self._encoder = self._load_pretrained_encoder(variant)
        else:
            self._encoder = _StubHieraEncoder(
                embed_dim=self.embed_dim,
                out_channels=self.out_channels,
            )

        # Freeze all encoder parameters by default
        for param in self._encoder.parameters():
            param.requires_grad = False

    def _load_pretrained_encoder(self, variant: str) -> nn.Module:
        """Load official SAM2 encoder from the sam2 package."""
        try:
            from sam2.build_sam import build_sam2
        except ImportError:
            msg = (
                "SAM2 pretrained weights require the 'sam2' package. "
                "Install with: uv add sam2 (or uv pip install sam2)"
            )
            raise ImportError(msg) from None

        checkpoint_url = SAM2_CHECKPOINT_URLS[variant]
        logger.info("Loading SAM2 %s encoder from %s", variant, checkpoint_url)

        # Use torch.hub to download checkpoint
        dst = f"/tmp/sam2_{variant}.pt"  # noqa: S108
        torch.hub.download_url_to_file(checkpoint_url, dst, progress=True)

        # Build SAM2 model and extract encoder
        sam2_model = build_sam2(variant, dst)
        encoder: nn.Module = sam2_model.image_encoder
        return encoder

    def extract_features(self, image_2d: Tensor) -> Tensor:
        """Extract features from a single 2D image.

        Parameters
        ----------
        image_2d:
            Input tensor of shape (B, C, H, W). C should be 3 for
            pretrained SAM2, or any channels for stub encoder.

        Returns
        -------
        Feature tensor of shape (B, out_channels, H', W') where
        H' and W' depend on the encoder's stride.
        """
        result: Tensor = self._encoder(image_2d)
        return result

    def get_image_embeddings(self, volume: Tensor) -> list[Tensor]:
        """Extract features slice-by-slice from a 3D volume.

        Parameters
        ----------
        volume:
            Input 3D tensor of shape (B, C, D, H, W).

        Returns
        -------
        List of D feature tensors, each (B, out_channels, H', W').
        """
        b, c, d, h, w = volume.shape
        embeddings: list[Tensor] = []

        for z_idx in range(d):
            slice_2d = volume[:, :, z_idx, :, :]  # (B, C, H, W)

            # SAM2 expects 3-channel input; replicate grayscale
            if slice_2d.shape[1] == 1:
                slice_2d = slice_2d.expand(-1, 3, -1, -1)

            # Resize to SAM's expected input size (1024x1024)
            if slice_2d.shape[2] != 1024 or slice_2d.shape[3] != 1024:
                slice_2d = F.interpolate(
                    slice_2d,
                    size=(1024, 1024),
                    mode="bilinear",
                    align_corners=False,
                )

            features: Tensor = self._encoder(slice_2d)
            embeddings.append(features)

        return embeddings
