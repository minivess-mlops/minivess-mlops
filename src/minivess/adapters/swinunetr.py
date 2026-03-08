"""SwinUNETR adapter — Swin Transformer + UNETR decoder for 3D segmentation.

Wraps MONAI's SwinUNETR (~62M params with default feature_size=48).
Transformer-based; strong on datasets where long-range context matters.

Reference: Tang et al. (2022). "Self-supervised pre-training of Swin transformers
for 3D medical image analysis." CVPR 2022.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from monai.networks.nets import (  # type: ignore[attr-defined]
    SwinUNETR as MonaiSwinUNETR,
)

from minivess.adapters.base import AdapterConfigInfo, ModelAdapter, SegmentationOutput

if TYPE_CHECKING:
    from torch import Tensor

    from minivess.config.models import ModelConfig

_DEFAULT_FEATURE_SIZE = 24
_DEFAULT_IMG_SIZE: tuple[int, int, int] = (96, 96, 96)


class SwinUNETRAdapter(ModelAdapter):
    """MONAI SwinUNETR adapter for 3D biomedical segmentation.

    Parameters
    ----------
    config:
        ModelConfig with MONAI_SWINUNETR family.
        Optional ``architecture_params``:
        - ``feature_size`` (int): Embedding feature size (default: 24).
        - ``img_size`` (tuple[int,int,int]): Expected input spatial size
          used for position embedding initialisation (default: (96,96,96)).
          Must match the actual patch/volume size used during forward.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        arch = config.architecture_params

        self._feature_size: int = int(arch.get("feature_size", _DEFAULT_FEATURE_SIZE))
        img_raw = arch.get("img_size", _DEFAULT_IMG_SIZE)
        self._img_size: tuple[int, int, int] = (
            int(img_raw[0]),
            int(img_raw[1]),
            int(img_raw[2]),
        )

        self.net = MonaiSwinUNETR(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            feature_size=self._feature_size,
            spatial_dims=3,
        )

    def forward(self, images: Tensor, **kwargs: Any) -> SegmentationOutput:
        """Run SwinUNETR inference.

        Parameters
        ----------
        images:
            Input tensor (B, C, D, H, W).

        Returns
        -------
        SegmentationOutput with softmax predictions and raw logits.
        """
        logits = self.net(images)
        return self._build_output(logits, "swinunetr")

    def get_config(self) -> AdapterConfigInfo:
        return self._build_config(
            feature_size=self._feature_size,
            img_size=list(self._img_size),
        )
