"""SDF regression head for multi-task segmentation models.

Lightweight head that predicts signed distance fields from decoder features.
No final activation — SDF values are unbounded real numbers.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class SDFHead(nn.Module):  # type: ignore[misc]
    """Lightweight 3D regression head for SDF prediction.

    Architecture: Conv3d(C, C//4, 1) -> InstanceNorm3d -> GELU -> Conv3d(C//4, 1, 1)
    No activation on final layer (SDF can be any real value).

    Args:
        in_channels: Number of input feature channels from decoder.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        mid_channels = max(in_channels // 4, 1)
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(mid_channels),
            nn.GELU(),
            nn.Conv3d(mid_channels, 1, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: [B, C, D, H, W] -> [B, 1, D, H, W]."""
        return self.net(x)
