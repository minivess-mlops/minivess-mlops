from __future__ import annotations

from typing import TYPE_CHECKING, Any

from monai.networks.nets import SegResNet as MonaiSegResNet

from minivess.adapters.base import AdapterConfigInfo, ModelAdapter, SegmentationOutput

if TYPE_CHECKING:
    from torch import Tensor

    from minivess.config.models import ModelConfig


class SegResNetAdapter(ModelAdapter):
    """MONAI SegResNet adapter for 3D biomedical segmentation."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        arch = config.architecture_params

        self._init_filters = arch.get("init_filters", 32)
        self._blocks_down = arch.get("blocks_down", (1, 2, 2, 4))
        self._blocks_up = arch.get("blocks_up", (1, 1, 1))

        self.net = MonaiSegResNet(
            spatial_dims=3,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            init_filters=self._init_filters,
            blocks_down=self._blocks_down,
            blocks_up=self._blocks_up,
            dropout_prob=0.2,
        )

    def forward(self, images: Tensor, **kwargs: Any) -> SegmentationOutput:
        logits = self.net(images)
        return self._build_output(logits, "segresnet")

    def get_config(self) -> AdapterConfigInfo:
        return self._build_config(
            init_filters=self._init_filters,
            blocks_down=self._blocks_down,
            blocks_up=self._blocks_up,
        )

    # load_checkpoint, save_checkpoint, trainable_parameters, export_onnx
    # inherited from ModelAdapter base class (uses self.net)
