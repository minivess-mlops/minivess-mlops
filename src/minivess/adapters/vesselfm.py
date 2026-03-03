"""vesselFM foundation model adapter for 3D vessel segmentation.

Wraps the vesselFM pre-trained DynUNet (Wittmann et al., 2024) with
HuggingFace Hub checkpoint download. vesselFM is the first foundation
model specifically for 3D blood vessel segmentation.

Reference: Wittmann et al. (2024). "vesselFM: A Foundation Model for
Universal 3D Blood Vessel Segmentation." CVPR 2025. arxiv:2411.17386
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, Any

import torch
from huggingface_hub import hf_hub_download
from monai.networks.nets import DynUNet
from torch import Tensor

from minivess.adapters.base import AdapterConfigInfo, ModelAdapter, SegmentationOutput

if TYPE_CHECKING:
    from minivess.config.models import ModelConfig

logger = logging.getLogger(__name__)

# vesselFM architecture constants (from dyn_unet_base.yaml)
VESSELFM_FILTERS: list[int] = [32, 64, 128, 256, 320, 320]
VESSELFM_HF_REPO = "bwittmann/vesselFM"
VESSELFM_HF_FILENAME = "vesselFM_base.pt"

# SHA256 checksum for the official vesselFM_base.pt weights.
# Set to None until verified from an actual download. The adapter
# skips checksum verification when this is None.
VESSELFM_WEIGHT_SHA256: str | None = None


def verify_checksum(data: bytes, expected_sha256: str | None) -> bool:
    """Verify SHA256 checksum of binary data.

    Parameters
    ----------
    data:
        Raw bytes to hash (e.g., file content).
    expected_sha256:
        Expected lowercase hex SHA256 digest (64 chars).
        If None, verification is skipped (returns True).

    Returns
    -------
    True if the computed hash matches the expected hash,
    or if expected_sha256 is None (skip verification).
    """
    if expected_sha256 is None:
        return True
    computed = hashlib.sha256(data).hexdigest()
    return computed == expected_sha256


def _strip_state_dict_prefix(
    state_dict: dict[str, Any],
    prefixes: tuple[str, ...] = ("module.", "model.", "conv."),
) -> dict[str, Any]:
    """Remove common prefixes from state dict keys.

    Parameters
    ----------
    state_dict:
        Original state dict with potentially prefixed keys.
    prefixes:
        Tuple of prefix strings to strip (tried in order).

    Returns
    -------
    New state dict with prefixes removed.
    """
    cleaned: dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
                break
        cleaned[new_key] = value
    return cleaned


class VesselFMAdapter(ModelAdapter):
    """vesselFM foundation model adapter.

    Uses a DynUNet backbone with [32, 64, 128, 256, 320, 320] encoder
    filters. The original vesselFM outputs a single binary channel;
    this adapter converts to 2-class output for API compatibility.

    Parameters
    ----------
    config:
        ModelConfig with VESSEL_FM family.
    pretrained:
        Download pre-trained weights from HuggingFace Hub.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.config = config

        n_levels = len(VESSELFM_FILTERS)
        kernel_size = [[3, 3, 3]] * n_levels
        strides = [[1, 1, 1]] + [[2, 2, 2]] * (n_levels - 1)
        upsample_kernel_size = [[2, 2, 2]] * (n_levels - 1)

        # vesselFM outputs 1 channel (binary segmentation)
        # We use out_channels=1 internally, then expand to 2 in forward()
        self.net = DynUNet(
            spatial_dims=3,
            in_channels=config.in_channels,
            out_channels=1,
            kernel_size=kernel_size,
            strides=strides,
            upsample_kernel_size=upsample_kernel_size,
            filters=VESSELFM_FILTERS,
            norm_name="instance",
            deep_supervision=False,
            res_block=True,
        )

        if pretrained:
            logger.warning(
                "vesselFM was pre-trained on MiniVess (1 of 17 datasets). "
                "Zero-shot results are NOT independent. "
                "Tag runs with data_leakage=pretrained_includes_minivess."
            )
            self._load_pretrained()

    def _load_pretrained(self) -> None:
        """Download and load vesselFM pre-trained weights."""
        logger.info(
            "Downloading vesselFM checkpoint from %s/%s",
            VESSELFM_HF_REPO,
            VESSELFM_HF_FILENAME,
        )
        ckpt_path = hf_hub_download(
            repo_id=VESSELFM_HF_REPO,
            filename=VESSELFM_HF_FILENAME,
        )
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # Handle potential prefix mismatches (module., model., etc.)
        state_dict = _strip_state_dict_prefix(state_dict)
        self.net.load_state_dict(state_dict, strict=False)
        logger.info("Loaded vesselFM pre-trained weights")

    def forward(self, images: Tensor, **kwargs: Any) -> SegmentationOutput:
        """Run vesselFM inference.

        Parameters
        ----------
        images:
            Input tensor (B, 1, D, H, W).

        Returns
        -------
        SegmentationOutput with 2-class softmax predictions.

        Note: Uses custom logic (binary-to-2-class conversion) rather
        than the standard _build_output helper.
        """
        # vesselFM outputs (B, 1, D, H, W) binary logits
        binary_logits = self.net(images)

        # Convert to 2-class: [background, foreground]
        logits = torch.cat([-binary_logits, binary_logits], dim=1)
        return self._build_output(logits, "vesselfm")

    def get_config(self) -> AdapterConfigInfo:
        return self._build_config(
            architecture="vesselfm",
            filters=VESSELFM_FILTERS,
        )

    # load_checkpoint, save_checkpoint, trainable_parameters, export_onnx
    # inherited from ModelAdapter base class (uses self.net)
