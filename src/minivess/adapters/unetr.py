"""UNETR adapter — ViT encoder + CNN decoder for 3D segmentation.

Wraps MONAI's UNETR (~92M params with default ViT-Base config).
Pure transformer encoder; requires img_size at construction time.

Reference: Hatamizadeh et al. (2022). "UNETR: Transformers for 3D medical
image segmentation." WACV 2022.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from monai.networks.nets import UNETR as MonaiUNETR  # type: ignore[attr-defined]

from minivess.adapters.base import AdapterConfigInfo, ModelAdapter, SegmentationOutput

if TYPE_CHECKING:
    from torch import Tensor

    from minivess.config.models import ModelConfig

_DEFAULT_IMG_SIZE: tuple[int, int, int] = (96, 96, 96)
_DEFAULT_HIDDEN_SIZE = 768
_DEFAULT_FEATURE_SIZE = 16


class UNETRAdapter(ModelAdapter):
    """MONAI UNETR adapter for 3D biomedical segmentation.

    Parameters
    ----------
    config:
        ModelConfig with MONAI_UNETR family.
        Optional ``architecture_params``:
        - ``img_size`` (tuple[int,int,int]): Input spatial size — must match
          actual patch/volume size (default: (96,96,96)).
        - ``hidden_size`` (int): ViT embedding dimension (default: 768).
        - ``feature_size`` (int): Feature map channels in CNN decoder (default: 16).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        arch = config.architecture_params

        img_raw = arch.get("img_size", _DEFAULT_IMG_SIZE)
        self._img_size: tuple[int, int, int] = (
            int(img_raw[0]),
            int(img_raw[1]),
            int(img_raw[2]),
        )
        self._hidden_size: int = int(arch.get("hidden_size", _DEFAULT_HIDDEN_SIZE))
        self._feature_size: int = int(arch.get("feature_size", _DEFAULT_FEATURE_SIZE))

        self.net = MonaiUNETR(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            img_size=self._img_size,
            hidden_size=self._hidden_size,
            feature_size=self._feature_size,
            spatial_dims=3,
        )

    def forward(self, images: Tensor, **kwargs: Any) -> SegmentationOutput:
        """Run UNETR inference.

        Parameters
        ----------
        images:
            Input tensor (B, C, D, H, W). Spatial dims must match ``img_size``.

        Returns
        -------
        SegmentationOutput with softmax predictions and raw logits.
        """
        logits = self.net(images)
        return self._build_output(logits, "unetr")

    def get_config(self) -> AdapterConfigInfo:
        return self._build_config(
            img_size=list(self._img_size),
            hidden_size=self._hidden_size,
            feature_size=self._feature_size,
        )
