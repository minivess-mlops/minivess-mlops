"""VISTA-3D model adapter using MONAI's SegResNetDS2 backbone.

VISTA-3D (He et al., 2024) is a foundation model for 3D interactive
segmentation. This adapter uses the auto-segmentation backbone
(SegResNetDS2) without interactive prompting for automated pipelines.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import torch
from monai.networks.nets import SegResNetDS2
from torch import Tensor

from minivess.adapters.base import AdapterConfigInfo, ModelAdapter, SegmentationOutput

if TYPE_CHECKING:
    from pathlib import Path

    from minivess.config.models import ModelConfig


class Vista3dAdapter(ModelAdapter):
    """MONAI SegResNetDS2 adapter (VISTA-3D backbone) for 3D segmentation."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.net = SegResNetDS2(
            spatial_dims=3,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            init_filters=32,
            blocks_down=(1, 2, 2, 4),
            dsdepth=1,
        )

    def forward(self, images: Tensor, **kwargs: Any) -> SegmentationOutput:
        """Run forward pass.

        Parameters
        ----------
        images:
            Input tensor (B, C, D, H, W).

        Returns
        -------
        SegmentationOutput with predictions and logits.
        """
        output = self.net(images)
        # SegResNetDS2 with dsdepth=1 returns a list of [main_output]
        logits = output[0] if isinstance(output, (list, tuple)) else output
        prediction = torch.softmax(logits, dim=1)
        return SegmentationOutput(
            prediction=prediction,
            logits=logits,
            metadata={"architecture": "vista3d"},
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
        """Export model to ONNX format."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.net.eval()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                onnx_program = torch.onnx.export(
                    self.net,
                    example_input,
                    dynamo=True,
                )
                onnx_program.save(str(path))
            except Exception:
                torch.onnx.export(
                    self.net,
                    example_input,
                    str(path),
                    input_names=["images"],
                    output_names=["logits"],
                    opset_version=17,
                    dynamo=False,
                )
