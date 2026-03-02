"""Slice-by-slice inference utility for 2D models on 3D volumes.

Provides helpers to iterate Z-slices through a 2D model (e.g., SAM2)
and reassemble the output into a 3D volume. Includes SAM-compatible
resizing (1024x1024) and inverse transform.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def resize_for_sam(image_2d: Tensor, target_size: int = 1024) -> Tensor:
    """Resize a 2D image to SAM's expected input size.

    Parameters
    ----------
    image_2d:
        Input tensor of shape (B, C, H, W).
    target_size:
        Target spatial size (square). SAM expects 1024.

    Returns
    -------
    Resized tensor of shape (B, C, target_size, target_size).
    """
    return F.interpolate(
        image_2d,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )


def unresize_from_sam(sam_output: Tensor, original_h: int, original_w: int) -> Tensor:
    """Resize SAM output back to original spatial dimensions.

    Parameters
    ----------
    sam_output:
        SAM model output of shape (B, C, H_sam, W_sam).
    original_h:
        Original height before resize.
    original_w:
        Original width before resize.

    Returns
    -------
    Tensor of shape (B, C, original_h, original_w).
    """
    return F.interpolate(
        sam_output,
        size=(original_h, original_w),
        mode="bilinear",
        align_corners=False,
    )


def slice_by_slice_forward(
    model_2d: nn.Module,
    volume: Tensor,
    *,
    resize_to: int | None = None,
) -> Tensor:
    """Run a 2D model slice-by-slice on a 3D volume.

    Iterates along the depth (Z) axis, passing each 2D slice through
    the model, then stacks results back into a 3D volume.

    Parameters
    ----------
    model_2d:
        Any ``nn.Module`` that accepts (B, C, H, W) and returns (B, C', H, W).
    volume:
        Input 3D tensor of shape (B, C, D, H, W).
    resize_to:
        If set, resize each slice to this size before model and unresize after.

    Returns
    -------
    Output tensor of shape (B, C', D, H, W).
    """
    b, c, d, h, w = volume.shape
    slices_out: list[Tensor] = []

    for z_idx in range(d):
        slice_2d = volume[:, :, z_idx, :, :]  # (B, C, H, W)

        if resize_to is not None:
            slice_2d = resize_for_sam(slice_2d, target_size=resize_to)

        out_2d = model_2d(slice_2d)  # (B, C', H', W')

        if resize_to is not None:
            out_2d = unresize_from_sam(out_2d, original_h=h, original_w=w)

        slices_out.append(out_2d)

    # Stack along depth: (B, C', D, H, W)
    return torch.stack(slices_out, dim=2)
