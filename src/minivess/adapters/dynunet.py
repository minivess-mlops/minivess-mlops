"""DynUNet adapter for nnU-Net-style 3D segmentation.

Wraps MONAI's DynUNet with configurable filter widths for
width ablation studies. Supports deep supervision.

Reference: Isensee et al. (2024). "nnU-Net Revisited." MICCAI 2024.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from monai.networks.nets import DynUNet as MonaiDynUNet
from torch import Tensor

from minivess.adapters.base import ModelAdapter, SegmentationOutput

if TYPE_CHECKING:
    from pathlib import Path

    from minivess.config.models import ModelConfig


class DynUNetAdapter(ModelAdapter):
    """MONAI DynUNet adapter with configurable width scaling.

    Parameters
    ----------
    config:
        ModelConfig with MONAI_DYNUNET family.
    filters:
        Channel widths per encoder level (e.g., [32, 64, 128, 256]).
    deep_supervision:
        Enable intermediate supervision heads.
    res_block:
        Use residual blocks (nnU-Net-v2 style).
    """

    def __init__(
        self,
        config: ModelConfig,
        filters: list[int] | None = None,
        *,
        deep_supervision: bool = False,
        res_block: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.filters = filters or [32, 64, 128, 256]
        self._deep_supervision = deep_supervision

        n_levels = len(self.filters)

        # Kernel sizes: 3×3×3 at each level
        kernel_size = [[3, 3, 3]] * n_levels

        # Strides: first is identity, rest are 2×2×2 downsampling
        strides = [[1, 1, 1]] + [[2, 2, 2]] * (n_levels - 1)

        # Upsample kernels match downsampling strides (skip first)
        upsample_kernel_size = [[2, 2, 2]] * (n_levels - 1)

        self.net = MonaiDynUNet(
            spatial_dims=3,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            kernel_size=kernel_size,
            strides=strides,
            upsample_kernel_size=upsample_kernel_size,
            filters=self.filters,
            norm_name="instance",
            deep_supervision=deep_supervision,
            res_block=res_block,
        )

    def forward(self, images: Tensor, **kwargs: Any) -> SegmentationOutput:
        """Run DynUNet inference.

        Parameters
        ----------
        images:
            Input tensor (B, C, D, H, W).

        Returns
        -------
        SegmentationOutput with softmax predictions and raw logits.
        """
        output = self.net(images)

        # Deep supervision returns stacked outputs; take the first (main)
        if self._deep_supervision and isinstance(output, (list, tuple)):
            logits = output[0]
        else:
            logits = output

        prediction = torch.softmax(logits, dim=1)

        return SegmentationOutput(
            prediction=prediction,
            logits=logits,
            metadata={"architecture": "dynunet"},
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "family": self.config.family.value,
            "name": self.config.name,
            "in_channels": self.config.in_channels,
            "out_channels": self.config.out_channels,
            "filters": self.filters,
            "deep_supervision": self._deep_supervision,
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
        """Export model to ONNX format."""
        import warnings

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
