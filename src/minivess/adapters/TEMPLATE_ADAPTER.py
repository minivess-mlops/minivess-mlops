"""TEMPLATE: How to wrap any monai.networks.nets.* model in ~5 minutes.

Copy this file, rename it, and follow the 5-step instructions below.
All lines marked "REPLACE THIS" must be changed.

Steps:
  1. Copy this file: cp TEMPLATE_ADAPTER.py mymodeladapter.py
  2. Replace YOURMODEL with your MONAI class (e.g., SegResNet, SwinUNETR)
  3. Add a ModelFamily enum entry in src/minivess/config/models.py
  4. Register in src/minivess/adapters/model_builder.py (_populate_registry)
  5. Write tests in tests/v2/unit/test_YOURMODEL_adapter.py (see test_segresnet_adapter.py)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# REPLACE THIS: import your MONAI model class
# from monai.networks.nets import YOURMODEL as MonaiYourModel
from monai.networks.nets import (  # type: ignore[attr-defined]
    SegResNet as MonaiYourModel,
)

from minivess.adapters.base import AdapterConfigInfo, ModelAdapter, SegmentationOutput

if TYPE_CHECKING:
    from torch import Tensor

    from minivess.config.models import ModelConfig


# REPLACE THIS: set any architecture-specific defaults here
_DEFAULT_PARAM_NAME = 16  # e.g., default init_filters


class YourModelAdapter(ModelAdapter):  # REPLACE THIS: rename the class
    """MONAI YOURMODEL adapter for 3D biomedical segmentation.

    Parameters
    ----------
    config:
        ModelConfig with MONAI_YOURMODEL family.  # REPLACE THIS
        Optional ``architecture_params``:
        - ``param_name`` (int): Description (default: 16).  # REPLACE THIS
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        arch = config.architecture_params

        # REPLACE THIS: extract your architecture params from config
        self._param_name: int = int(arch.get("param_name", _DEFAULT_PARAM_NAME))

        # REPLACE THIS: construct your MONAI model
        # Always use spatial_dims=3, in_channels, out_channels from config.
        # Do NOT hardcode these — the adapter must be model-agnostic.
        self.net = MonaiYourModel(
            spatial_dims=3,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            # YOUR MODEL PARAMS HERE
        )

    def forward(self, images: Tensor, **kwargs: Any) -> SegmentationOutput:
        """Run YOUR MODEL inference.

        Parameters
        ----------
        images:
            Input tensor (B, C, D, H, W).

        Returns
        -------
        SegmentationOutput with softmax predictions and raw logits.
        """
        logits = self.net(images)

        # If your model has deep supervision (returns list/tuple), extract the main output:
        # if isinstance(logits, (list, tuple)):
        #     logits = logits[0]

        # REPLACE THIS: use your model's architecture name in the string
        return self._build_output(logits, "yourmodel")

    def get_config(self) -> AdapterConfigInfo:
        # REPLACE THIS: add any model-specific extras you want in the config snapshot
        return self._build_config(param_name=self._param_name)

    # -----------------------------------------------------------------------
    # Inherited from ModelAdapter (no changes needed unless your model differs):
    # - load_checkpoint()   — loads self.net state dict
    # - save_checkpoint()   — saves self.net state dict
    # - trainable_parameters() — counts self.net params with requires_grad=True
    # - export_onnx()        — exports self.net with dynamo/fallback
    # -----------------------------------------------------------------------
