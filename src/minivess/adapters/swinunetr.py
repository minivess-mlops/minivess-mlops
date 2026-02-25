from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from monai.networks.nets import SwinUNETR as MonaiSwinUNETR
from torch import Tensor

from minivess.adapters.base import AdapterConfigInfo, ModelAdapter, SegmentationOutput
from minivess.config.models import ModelConfig


class SwinUNETRAdapter(ModelAdapter):
    """MONAI SwinUNETR adapter for transformer-based 3D segmentation."""

    def __init__(
        self,
        config: ModelConfig,
        feature_size: int = 48,
    ) -> None:
        super().__init__()
        self.config = config
        self._feature_size = feature_size
        self.net = MonaiSwinUNETR(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            feature_size=feature_size,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            norm_name="instance",
            spatial_dims=3,
        )

    def forward(self, images: Tensor, **kwargs: Any) -> SegmentationOutput:
        logits = self.net(images)
        prediction = torch.softmax(logits, dim=1)
        return SegmentationOutput(
            prediction=prediction,
            logits=logits,
            metadata={"architecture": "swinunetr"},
        )

    def get_config(self) -> AdapterConfigInfo:
        return AdapterConfigInfo(
            family=self.config.family.value,
            name=self.config.name,
            in_channels=self.config.in_channels,
            out_channels=self.config.out_channels,
            trainable_params=self.trainable_parameters(),
            extras={
                "feature_size": self._feature_size,
                "depths": (2, 2, 2, 2),
                "num_heads": (3, 6, 12, 24),
            },
        )

    # load_checkpoint, save_checkpoint, trainable_parameters inherited from base.
    # Custom export_onnx with dynamic_axes for SwinUNETR.

    def export_onnx(self, path: Path, example_input: Tensor) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            self.net,
            example_input,
            str(path),
            input_names=["images"],
            output_names=["logits"],
            dynamic_axes={
                "images": {0: "batch", 2: "depth", 3: "height", 4: "width"},
                "logits": {0: "batch", 2: "depth", 3: "height", 4: "width"},
            },
            opset_version=17,
        )
