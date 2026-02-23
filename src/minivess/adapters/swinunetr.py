from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from monai.networks.nets import SwinUNETR as MonaiSwinUNETR
from torch import Tensor

from minivess.adapters.base import ModelAdapter, SegmentationOutput
from minivess.config.models import ModelConfig


class SwinUNETRAdapter(ModelAdapter):
    """MONAI SwinUNETR adapter for transformer-based 3D segmentation."""

    def __init__(
        self,
        config: ModelConfig,
        img_size: tuple[int, int, int] = (128, 128, 32),
    ) -> None:
        super().__init__()
        self.config = config
        self._img_size = img_size
        self.net = MonaiSwinUNETR(
            img_size=img_size,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            feature_size=48,
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

    def get_config(self) -> dict[str, Any]:
        return {
            "family": self.config.family.value,
            "name": self.config.name,
            "in_channels": self.config.in_channels,
            "out_channels": self.config.out_channels,
            "img_size": self._img_size,
            "feature_size": 48,
            "depths": (2, 2, 2, 2),
            "num_heads": (3, 6, 12, 24),
            "trainable_params": self.trainable_parameters(),
        }

    def load_checkpoint(self, path: Path) -> None:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        self.net.load_state_dict(state_dict)

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path)

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

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
