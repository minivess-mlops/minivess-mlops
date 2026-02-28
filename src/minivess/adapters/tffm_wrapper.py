"""TFFM (Topology Feature Fusion Module) wrapper adapter.

Composition pattern: TFFMWrapper(DynUNetAdapter(config)) — wraps any
ModelAdapter and applies graph attention-based feature fusion via a
forward hook on the bottleneck layer.

Uses dense graph attention (no torch_geometric dependency). Graph is
constructed via cosine-similarity kNN on pooled spatial grid cells.

Reference: Ahmed et al. (2026). "TFFM: Topology-Aware Feature Fusion Module
via Latent Graph Reasoning for Retinal Vessel Segmentation." WACV 2026.
Ported from https://github.com/tffm-module/tffm-code (2D → 3D).

See CLAUDE.md Critical Rule #3 — uses PyTorch built-in ops (adaptive_avg_pool3d,
cosine_similarity, softmax) for all graph construction and attention.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from minivess.adapters.base import AdapterConfigInfo, ModelAdapter, SegmentationOutput
from minivess.adapters.graph_modules import TFFMBlock3D

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class TFFMWrapper(ModelAdapter):  # type: ignore[misc]
    """TFFM wrapper that applies graph attention fusion to any ModelAdapter.

    Uses the composition pattern (like LoRA): holds ``_base_model`` and
    registers a forward hook on the bottleneck layer to apply the TFFM
    block during inference.

    Parameters
    ----------
    base_model:
        An existing ModelAdapter to wrap (e.g., DynUNetAdapter).
    grid_size:
        Spatial grid size for graph construction (G → G^3 nodes).
    hidden_dim:
        Hidden feature dimension for graph attention.
    n_heads:
        Number of attention heads in GAT layers.
    k_neighbors:
        Number of neighbors in kNN graph.
    dropout:
        Dropout probability for attention.
    """

    def __init__(
        self,
        base_model: ModelAdapter,
        grid_size: int = 8,
        hidden_dim: int = 32,
        n_heads: int = 4,
        k_neighbors: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self._base_model = base_model
        self.config = base_model.config
        self._grid_size = grid_size
        self._hidden_dim = hidden_dim
        self._n_heads = n_heads

        # Find bottleneck layer and get its channel count
        target_layer, n_channels = self._find_bottleneck(base_model)

        # Create TFFM block matching bottleneck channels
        self.tffm_block = TFFMBlock3D(
            in_channels=n_channels,
            grid_size=grid_size,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            k_neighbors=k_neighbors,
            dropout=dropout,
        )

        # Register forward hook on bottleneck
        self._hook_handle = target_layer.register_forward_hook(self._tffm_hook)

        tffm_params = sum(p.numel() for p in self.tffm_block.parameters())
        logger.info(
            "TFFM applied to bottleneck (%d channels): %d params, grid=%d, heads=%d",
            n_channels,
            tffm_params,
            grid_size,
            n_heads,
        )

    def _tffm_hook(
        self,
        module: nn.Module,
        input: tuple[Tensor, ...],
        output: Tensor,
    ) -> Tensor:
        """Forward hook: apply TFFM block to bottleneck output."""
        result: Tensor = self.tffm_block(output)
        return result

    @staticmethod
    def _find_bottleneck(model: ModelAdapter) -> tuple[nn.Module, int]:
        """Find the bottleneck (deepest encoder) layer in the base model.

        Searches for the deepest convolutional module in the network
        by examining the model's structure.

        Returns
        -------
        Tuple of (target_module, n_channels).
        """
        net: nn.Module = model.net

        # Strategy 1: MONAI DynUNet — bottleneck is the last module before upsamples
        if hasattr(net, "bottleneck"):
            bottleneck: nn.Module = net.bottleneck
            n_channels = _get_output_channels(bottleneck)
            return bottleneck, n_channels

        # Strategy 2: MONAI SegResNet / SegResNetDS2 — look for down_layers[-1]
        if hasattr(net, "down_layers"):
            last_down: nn.Module = net.down_layers[-1]
            n_channels = _get_output_channels(last_down)
            return last_down, n_channels

        # Strategy 3: Generic fallback — find the deepest Conv3d
        last_conv: nn.Module | None = None
        last_channels = 0
        for module in net.modules():
            if isinstance(module, nn.Conv3d):
                last_conv = module
                last_channels = module.out_channels

        if last_conv is not None:
            logger.warning(
                "No known bottleneck structure found; hooking last Conv3d (%d ch)",
                last_channels,
            )
            return last_conv, last_channels

        msg = "Cannot find bottleneck layer in base model"
        raise ValueError(msg)

    def forward(self, images: Tensor, **kwargs: Any) -> SegmentationOutput:
        """Run forward pass through TFFM-enhanced model.

        The hook modifies bottleneck features in-place during the base
        model's forward pass, so no additional logic is needed here.
        """
        result: SegmentationOutput = self._base_model(images, **kwargs)
        # Add TFFM metadata
        result.metadata["tffm_applied"] = True
        return result

    def get_config(self) -> AdapterConfigInfo:
        base_config = self._base_model.get_config()
        base_config.extras.update(
            {
                "tffm_grid_size": self._grid_size,
                "tffm_hidden_dim": self._hidden_dim,
                "tffm_n_heads": self._n_heads,
                "tffm_params": sum(p.numel() for p in self.tffm_block.parameters()),
            }
        )
        return base_config

    def load_checkpoint(self, path: Path) -> None:
        """Load full wrapper state (base model + TFFM block)."""
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        self.load_state_dict(state_dict, strict=False)

    def save_checkpoint(self, path: Path) -> None:
        """Save full wrapper state (base model + TFFM block)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def trainable_parameters(self) -> int:
        """Return total trainable params (base + TFFM)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def export_onnx(self, path: Path, example_input: Tensor) -> None:
        """Export TFFM-enhanced model to ONNX."""
        self._base_model.export_onnx(path, example_input)

    def __del__(self) -> None:
        """Remove the forward hook on cleanup."""
        if hasattr(self, "_hook_handle"):
            self._hook_handle.remove()


def _get_output_channels(module: nn.Module) -> int:
    """Get output channels from a module by inspecting its Conv3d layers."""
    last_channels = 0
    for m in module.modules():
        if isinstance(m, nn.Conv3d):
            last_channels = m.out_channels
    if last_channels == 0:
        # Fallback: look for BatchNorm/InstanceNorm
        for m in module.modules():
            if isinstance(m, nn.BatchNorm3d | nn.InstanceNorm3d):
                last_channels = m.num_features
    return last_channels
