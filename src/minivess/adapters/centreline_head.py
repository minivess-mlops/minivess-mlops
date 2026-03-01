"""Centreline prediction head — multi-task adapter.

Wraps any ModelAdapter and adds a 1x1x1 conv head that predicts a
centreline distance map alongside the segmentation output. Uses the
composition pattern (like LoRA/TFFM): holds ``_base_model``.

GT centreline distance maps are computed via:
  - skimage.morphology.skeletonize (Lee94) for skeleton extraction
  - scipy.ndimage.distance_transform_edt for distance computation

See CLAUDE.md Critical Rule #3 — uses established libraries for
all non-differentiable GT processing.

References:
  - Kirchhoff et al. (2024) — Skeleton recall, ECCV 2024
  - Shit et al. (2021) — clDice, CVPR 2021
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor, nn

from minivess.adapters.base import AdapterConfigInfo, ModelAdapter, SegmentationOutput

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class CentrelineHeadAdapter(ModelAdapter):  # type: ignore[misc]
    """Multi-task adapter: segmentation + centreline distance map.

    Wraps any ModelAdapter and adds a regression head that predicts
    distance-to-centreline for each voxel. Uses a forward hook on the
    last decoder layer to capture features, then runs a lightweight
    1x1x1 conv head.

    Parameters
    ----------
    base_model:
        An existing ModelAdapter to wrap.
    seg_weight:
        Weight for segmentation loss in multi-task training.
    centreline_weight:
        Weight for centreline regression loss.
    enabled:
        If False, skip centreline head entirely (backward compatibility).
    """

    def __init__(
        self,
        base_model: ModelAdapter,
        seg_weight: float = 0.5,
        centreline_weight: float = 0.5,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self._base_model = base_model
        self.config = base_model.config
        self._seg_weight = seg_weight
        self._centreline_weight = centreline_weight
        self._enabled = enabled

        if enabled:
            # Find the layer before the output head to capture decoder features
            target_layer, n_channels = self._find_last_decoder_layer(base_model)

            # Centreline distance head: conv → ReLU (distances are non-negative)
            self.centreline_head = nn.Sequential(
                nn.Conv3d(n_channels, 1, kernel_size=1),
                nn.ReLU(),
            )

            # Captured features storage
            self._captured_features: Tensor | None = None
            self._hook_handle = target_layer.register_forward_hook(self._capture_hook)

            head_params = sum(p.numel() for p in self.centreline_head.parameters())
            logger.info(
                "Centreline head added: %d input channels, %d params",
                n_channels,
                head_params,
            )

    def _capture_hook(
        self,
        module: nn.Module,
        input: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        """Forward hook: capture decoder features (no modification)."""
        self._captured_features = output

    @staticmethod
    def _find_last_decoder_layer(model: ModelAdapter) -> tuple[nn.Module, int]:
        """Find the last decoder layer before the output head.

        Returns
        -------
        Tuple of (target_module, n_channels).
        """
        net: nn.Module = model.net

        # Strategy 1: MONAI DynUNet — hook the last upsampling block
        if hasattr(net, "upsamples") and len(net.upsamples) > 0:
            last_up: nn.Module = net.upsamples[-1]
            n_channels = _get_output_channels(last_up)
            if n_channels > 0:
                return last_up, n_channels

        # Strategy 2: Look for output_block and hook the layer before it
        if hasattr(net, "output_block"):
            # Find the first Conv3d in the output block to get its in_channels
            for m in net.output_block.modules():
                if isinstance(m, nn.Conv3d):
                    return net.output_block, m.in_channels

        # Strategy 3: Generic fallback — find second-to-last Conv3d
        convs: list[tuple[nn.Module, int]] = []
        for module in net.modules():
            if isinstance(module, nn.Conv3d):
                convs.append((module, module.out_channels))

        if len(convs) >= 2:
            target, n_ch = convs[-2]
            logger.warning(
                "No known decoder structure; hooking second-to-last Conv3d (%d ch)",
                n_ch,
            )
            return target, n_ch

        if len(convs) >= 1:
            target, n_ch = convs[-1]
            return target, n_ch

        msg = "Cannot find decoder layer in base model"
        raise ValueError(msg)

    def forward(self, images: Tensor, **kwargs: Any) -> SegmentationOutput:
        """Run forward pass with optional centreline prediction.

        The capture hook stores decoder features during the base model's
        forward pass. Then the centreline head runs on those features.
        """
        self._captured_features = None
        result: SegmentationOutput = self._base_model(images, **kwargs)

        if not self._enabled:
            result.metadata["centreline_head_enabled"] = False
            result.metadata["centreline_map"] = None
            return result

        # Run centreline head on captured decoder features
        if self._captured_features is not None:
            features = self._captured_features
            # Ensure spatial dims match output
            if features.shape[2:] != result.logits.shape[2:]:
                features = nn.functional.interpolate(
                    features,
                    size=result.logits.shape[2:],
                    mode="trilinear",
                    align_corners=False,
                )
            centreline_map: Tensor = self.centreline_head(features)
        else:
            # Fallback: run head on logits if hook didn't fire
            centreline_map = self.centreline_head(result.logits)

        result.metadata["centreline_map"] = centreline_map
        result.metadata["centreline_head_enabled"] = True
        return result

    def get_config(self) -> AdapterConfigInfo:
        base_config = self._base_model.get_config()
        base_config.extras.update(
            {
                "centreline_head_enabled": self._enabled,
                "seg_weight": self._seg_weight,
                "centreline_weight": self._centreline_weight,
            }
        )
        return base_config

    def load_checkpoint(self, path: Path) -> None:
        """Load full wrapper state (base model + centreline head)."""
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        self.load_state_dict(state_dict, strict=False)

    def save_checkpoint(self, path: Path) -> None:
        """Save full wrapper state (base model + centreline head)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def trainable_parameters(self) -> int:
        """Return total trainable params (base + centreline head)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def export_onnx(self, path: Path, example_input: Tensor) -> None:
        self._base_model.export_onnx(path, example_input)

    def __del__(self) -> None:
        if hasattr(self, "_hook_handle"):
            self._hook_handle.remove()


def compute_centreline_distance_map(mask: np.ndarray) -> np.ndarray:
    """Compute distance-to-centreline map from a binary 3D mask.

    Uses established libraries:
      - skimage.morphology.skeletonize (Lee94) for skeleton extraction
      - scipy.ndimage.distance_transform_edt for distance computation

    Parameters
    ----------
    mask:
        Binary 3D mask (D, H, W).

    Returns
    -------
    Distance map where each voxel contains its Euclidean distance to
    the nearest centreline voxel. Background voxels get distance 0.
    """
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize

    mask_bin = np.asarray(mask, dtype=bool)

    if not mask_bin.any():
        return np.zeros_like(mask_bin, dtype=np.float32)

    # Step 1: Extract skeleton via skimage (Lee94)
    skeleton = skeletonize(mask_bin)

    if not skeleton.any():
        # Thin structure: skeleton is empty, return all zeros
        return np.zeros_like(mask_bin, dtype=np.float32)

    # Step 2: Distance to nearest skeleton voxel via scipy EDT
    # Invert skeleton: EDT computes distance to nearest True voxel
    dist_map = distance_transform_edt(~skeleton).astype(np.float32)

    return dist_map


def _get_output_channels(module: nn.Module) -> int:
    """Get output channels from a module by inspecting its Conv3d layers."""
    last_channels = 0
    for m in module.modules():
        if isinstance(m, nn.Conv3d):
            last_channels = m.out_channels
    if last_channels == 0:
        for m in module.modules():
            if isinstance(m, nn.BatchNorm3d | nn.InstanceNorm3d):
                last_channels = m.num_features
    return last_channels
