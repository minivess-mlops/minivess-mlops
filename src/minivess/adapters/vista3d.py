"""VISTA-3D model adapter using MONAI's SegResNetDS2 backbone.

VISTA-3D (He et al., 2024) is a foundation model for 3D interactive
segmentation. This adapter uses the auto-segmentation backbone
(SegResNetDS2) without interactive prompting for automated pipelines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from monai.networks.nets import SegResNetDS2
from torch import Tensor

from minivess.adapters.base import AdapterConfigInfo, ModelAdapter, SegmentationOutput

if TYPE_CHECKING:

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

    # load_checkpoint, save_checkpoint, trainable_parameters, export_onnx
    # inherited from ModelAdapter base class (uses self.net)
