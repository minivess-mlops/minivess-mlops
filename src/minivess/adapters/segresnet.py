"""SegResNet adapter — lightweight ResNet encoder-decoder for 3D segmentation.

Wraps MONAI's SegResNet (~4.7M params with default init_filters=16).
Good for memory-constrained scenarios and as a fast baseline model.

Reference: Myronenko (2018). "3D MRI brain tumor segmentation using
autoencoder regularization." BrainLes@MICCAI 2018.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from monai.networks.nets import (  # type: ignore[attr-defined]
    SegResNet as MonaiSegResNet,
)

from minivess.adapters.base import AdapterConfigInfo, ModelAdapter, SegmentationOutput

if TYPE_CHECKING:
    from torch import Tensor

    from minivess.config.models import ModelConfig

_DEFAULT_INIT_FILTERS = 16


class SegResNetAdapter(ModelAdapter):
    """MONAI SegResNet adapter for 3D biomedical segmentation.

    Parameters
    ----------
    config:
        ModelConfig with MONAI_SEGRESNET family.
        Optional ``architecture_params``:
        - ``init_filters`` (int): Initial number of conv filters (default: 16).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        arch = config.architecture_params

        self._init_filters: int = int(arch.get("init_filters", _DEFAULT_INIT_FILTERS))

        self.net = MonaiSegResNet(
            spatial_dims=3,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            init_filters=self._init_filters,
        )

    def forward(self, images: Tensor, **kwargs: Any) -> SegmentationOutput:
        """Run SegResNet inference.

        Parameters
        ----------
        images:
            Input tensor (B, C, D, H, W).

        Returns
        -------
        SegmentationOutput with softmax predictions and raw logits.
        """
        logits = self.net(images)
        return self._build_output(logits, "segresnet")

    def get_config(self) -> AdapterConfigInfo:
        return self._build_config(init_filters=self._init_filters)
