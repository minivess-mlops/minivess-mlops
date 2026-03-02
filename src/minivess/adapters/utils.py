"""Shared adapter utilities factored from centreline_head.py."""

from __future__ import annotations

from torch import nn


def get_output_channels(module: nn.Module) -> int:
    """Get output channels from a module by inspecting its Conv3d layers.

    Falls back to BatchNorm3d/InstanceNorm3d if no Conv3d found.

    Parameters
    ----------
    module:
        Any nn.Module containing Conv3d or normalization layers.

    Returns
    -------
    Number of output channels, or 0 if none found.
    """
    last_channels = 0
    for m in module.modules():
        if isinstance(m, nn.Conv3d):
            last_channels = m.out_channels
    if last_channels == 0:
        for m in module.modules():
            if isinstance(m, nn.BatchNorm3d | nn.InstanceNorm3d):
                last_channels = m.num_features
    return last_channels
