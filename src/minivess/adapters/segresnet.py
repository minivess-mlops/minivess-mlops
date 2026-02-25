from __future__ import annotations

from pathlib import Path
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

    def load_checkpoint(self, path: Path) -> None:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        self.net.load_state_dict(state_dict)

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path)

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def export_onnx(self, path: Path, example_input: Tensor) -> None:
        """Export model to ONNX format.

        Uses the new dynamo-based exporter (PyTorch 2.5+) with fallback
        to the legacy TorchScript exporter for compatibility.
        """
        import warnings

        path.parent.mkdir(parents=True, exist_ok=True)
        self.net.eval()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # Try dynamo exporter first (PyTorch 2.5+)
                onnx_program = torch.onnx.export(
                    self.net,
                    example_input,
                    dynamo=True,
                )
                onnx_program.save(str(path))
            except Exception:
                # Fallback: legacy TorchScript exporter
                torch.onnx.export(
                    self.net,
                    example_input,
                    str(path),
                    input_names=["images"],
                    output_names=["logits"],
                    opset_version=17,
                    dynamo=False,
                )
