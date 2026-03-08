"""AttentionUnet adapter — U-Net with attention gates for 3D segmentation.

Wraps MONAI's AttentionUnet (~5M params with default channels).
Lightweight, attention-augmented U-Net; good lightweight comparison point.

Reference: Oktay et al. (2018). "Attention U-Net: Learning where to look
for the pancreas." MIDL 2018.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from monai.networks.nets import (  # type: ignore[attr-defined]
    AttentionUnet as MonaiAttentionUnet,
)

from minivess.adapters.base import AdapterConfigInfo, ModelAdapter, SegmentationOutput

if TYPE_CHECKING:
    from torch import Tensor

    from minivess.config.models import ModelConfig

# Default channels (encoder levels) and matching strides
_DEFAULT_CHANNELS = (16, 32, 64, 128, 256)
_DEFAULT_STRIDES = (2, 2, 2, 2)


class AttentionUnetAdapter(ModelAdapter):
    """MONAI AttentionUnet adapter for 3D biomedical segmentation.

    Parameters
    ----------
    config:
        ModelConfig with MONAI_ATTENTIONUNET family.
        Optional ``architecture_params``:
        - ``channels`` (list[int]): Feature channels per level
          (default: (16, 32, 64, 128, 256)).
        - ``strides`` (list[int]): Downsampling strides; length must be
          ``len(channels) - 1`` (default: (2, 2, 2, 2)).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        arch = config.architecture_params

        channels_raw = arch.get("channels", _DEFAULT_CHANNELS)
        strides_raw = arch.get("strides", _DEFAULT_STRIDES)
        self._channels: tuple[int, ...] = tuple(int(c) for c in channels_raw)
        self._strides: tuple[int, ...] = tuple(int(s) for s in strides_raw)

        self.net = MonaiAttentionUnet(
            spatial_dims=3,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            channels=self._channels,
            strides=self._strides,
        )

    def forward(self, images: Tensor, **kwargs: Any) -> SegmentationOutput:
        """Run AttentionUnet inference.

        Parameters
        ----------
        images:
            Input tensor (B, C, D, H, W).

        Returns
        -------
        SegmentationOutput with softmax predictions and raw logits.
        """
        logits = self.net(images)
        return self._build_output(logits, "attentionunet")

    def get_config(self) -> AdapterConfigInfo:
        return self._build_config(
            channels=list(self._channels),
            strides=list(self._strides),
        )
