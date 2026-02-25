from __future__ import annotations

from typing import Any

import torch
from monai.networks.nets import SegResNet as MonaiSegResNet
from torch import Tensor

from minivess.adapters.base import AdapterConfigInfo, ModelAdapter, SegmentationOutput
from minivess.config.models import ModelConfig


class SegResNetAdapter(ModelAdapter):
    """MONAI SegResNet adapter for 3D biomedical segmentation."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.net = MonaiSegResNet(
            spatial_dims=3,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            init_filters=32,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            dropout_prob=0.2,
        )

    def forward(self, images: Tensor, **kwargs: Any) -> SegmentationOutput:
        logits = self.net(images)
        prediction = torch.softmax(logits, dim=1)
        return SegmentationOutput(
            prediction=prediction,
            logits=logits,
            metadata={"architecture": "segresnet"},
        )

    def get_config(self) -> AdapterConfigInfo:
        return AdapterConfigInfo(
            family=self.config.family.value,
            name=self.config.name,
            in_channels=self.config.in_channels,
            out_channels=self.config.out_channels,
            trainable_params=self.trainable_parameters(),
            extras={
                "init_filters": 32,
                "blocks_down": (1, 2, 2, 4),
                "blocks_up": (1, 1, 1),
            },
        )

    # load_checkpoint, save_checkpoint, trainable_parameters, export_onnx
    # inherited from ModelAdapter base class (uses self.net)
