"""Generic multi-task adapter with config-driven auxiliary heads.

Wraps any ModelAdapter and adds N auxiliary heads defined by config.
Task-agnostic — it knows about head TYPES (regression, classification,
segmentation) but NOT about specific TASKS (SDF, centerline, etc.).

Uses forward hooks to capture decoder features and runs lightweight
heads on them. New tasks require only config changes, not code changes.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from minivess.adapters.base import AdapterConfigInfo, SegmentationOutput
from minivess.adapters.utils import get_output_channels

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class AuxHeadConfig:
    """Configuration for one auxiliary head.

    Args:
        name: Head name (key in SegmentationOutput.metadata).
        head_type: One of "regression", "classification", "segmentation".
        out_channels: Number of output channels for this head.
        gt_key: Key in batch dict for this head's ground truth. Defaults to name.
    """

    name: str
    head_type: str
    out_channels: int
    gt_key: str = ""

    def __post_init__(self) -> None:
        if not self.gt_key:
            self.gt_key = self.name


def _build_head(head_type: str, in_channels: int, out_channels: int) -> nn.Module:
    """Build an auxiliary head module from config.

    Parameters
    ----------
    head_type:
        One of "regression", "classification", "segmentation".
    in_channels:
        Number of input channels from decoder features.
    out_channels:
        Number of output channels for this head.

    Returns
    -------
    nn.Module implementing the head.
    """
    mid_channels = max(in_channels // 4, 1)

    if head_type not in ("regression", "classification", "segmentation"):
        msg = f"Unknown head type: {head_type}. Supported: regression, classification, segmentation"
        raise ValueError(msg)

    # All head types share the same architecture (1x1 conv → norm → GELU → 1x1 conv).
    # No final activation: regression outputs are unbounded, classification/segmentation
    # outputs are raw logits (activation applied downstream by the loss function).
    return nn.Sequential(
        nn.Conv3d(in_channels, mid_channels, kernel_size=1, bias=False),
        nn.InstanceNorm3d(mid_channels),
        nn.GELU(),
        nn.Conv3d(mid_channels, out_channels, kernel_size=1),
    )


class MultiTaskAdapter(nn.Module):  # type: ignore[misc]
    """Generic multi-task adapter wrapping any base model.

    Adds N auxiliary heads defined by AuxHeadConfig. Uses a forward hook
    on the last decoder layer to capture features, then runs lightweight
    heads on those features.

    Parameters
    ----------
    base_model:
        The base model to wrap (must have .net, decoder_conv, or Conv3d layers).
    aux_head_configs:
        List of AuxHeadConfig defining auxiliary heads.
    """

    def __init__(
        self,
        base_model: nn.Module,
        aux_head_configs: list[AuxHeadConfig] | None = None,
    ) -> None:
        super().__init__()
        self._base_model = base_model
        self.aux_head_configs = aux_head_configs or []

        # Find decoder layer and channel count
        target_layer, n_channels = self._find_last_decoder_layer(base_model)

        # Build aux heads as a ModuleDict (registered as submodules)
        self.aux_heads = nn.ModuleDict()
        for config in self.aux_head_configs:
            self.aux_heads[config.name] = _build_head(
                config.head_type, n_channels, config.out_channels
            )
            head_params = sum(
                p.numel() for p in self.aux_heads[config.name].parameters()
            )
            logger.info(
                "Aux head '%s' (%s, %d out_ch): %d params from %d decoder channels",
                config.name,
                config.head_type,
                config.out_channels,
                head_params,
                n_channels,
            )

        # Hook to capture decoder features
        self._captured_features: Tensor | None = None
        self._hook_handle = target_layer.register_forward_hook(self._capture_hook)

    def _capture_hook(
        self,
        module: nn.Module,
        input: tuple[Tensor, ...],  # noqa: A002
        output: Tensor,
    ) -> None:
        """Forward hook: capture decoder features."""
        self._captured_features = output

    @staticmethod
    def _find_last_decoder_layer(model: nn.Module) -> tuple[nn.Module, int]:
        """Find the last decoder layer before the output head.

        Returns
        -------
        Tuple of (target_module, n_channels).
        """
        net: Any = getattr(model, "net", model)

        # Strategy 1: MONAI DynUNet — hook the last upsampling block
        if hasattr(net, "upsamples") and len(net.upsamples) > 0:
            last_up: nn.Module = net.upsamples[-1]
            n_channels = get_output_channels(last_up)
            if n_channels > 0:
                return last_up, n_channels

        # Strategy 2: Look for output_block and hook the layer before it
        if hasattr(net, "output_block"):
            for m in net.output_block.modules():
                if isinstance(m, nn.Conv3d):
                    return net.output_block, m.in_channels

        # Strategy 3: Hook decoder_conv if available (stub/test models)
        if hasattr(net, "decoder_conv"):
            decoder_conv: nn.Module = net.decoder_conv
            n_channels = get_output_channels(decoder_conv)
            if n_channels > 0:
                return decoder_conv, n_channels

        # Strategy 4: Generic fallback — second-to-last Conv3d
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
        """Run forward pass: base model + auxiliary heads.

        The capture hook stores decoder features during the base model's
        forward pass. Then each aux head runs on those features.
        """
        self._captured_features = None
        result: SegmentationOutput = self._base_model(images, **kwargs)

        if not self.aux_head_configs:
            return result

        # Get features for aux heads
        features = self._captured_features
        if features is None:
            # Fallback: use logits if hook didn't fire
            features = result.logits

        # Ensure spatial dims match output
        if features.shape[2:] != result.logits.shape[2:]:
            features = nn.functional.interpolate(
                features,
                size=result.logits.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

        # Run each aux head
        for config in self.aux_head_configs:
            head = self.aux_heads[config.name]
            aux_output: Tensor = head(features)
            result.metadata[config.name] = aux_output

        return result

    # --- ModelAdapter-compatible interface ---

    def get_config(self) -> AdapterConfigInfo:
        """Return config info including aux heads."""
        get_config_fn = getattr(self._base_model, "get_config", None)
        if get_config_fn is not None:
            base_config: AdapterConfigInfo = get_config_fn()
        else:
            base_config = AdapterConfigInfo(family="unknown", name="unknown")
        base_config.extras["aux_heads"] = [
            dataclasses.asdict(c) for c in self.aux_head_configs
        ]
        return base_config

    def load_checkpoint(self, path: Path) -> None:
        """Load full wrapper state (base model + aux heads)."""
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        self.load_state_dict(state_dict, strict=False)

    def save_checkpoint(self, path: Path) -> None:
        """Save full wrapper state (base model + aux heads)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def trainable_parameters(self) -> int:
        """Return total trainable params (base + all aux heads)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def export_onnx(self, path: Path, example_input: Tensor) -> None:
        """Default: delegate to base model (mask-only export)."""
        export_fn = getattr(self._base_model, "export_onnx", None)
        if export_fn is not None:
            export_fn(path, example_input)

    @property
    def config(self) -> Any:
        """Proxy config from base model."""
        return self._base_model.config

    def __del__(self) -> None:
        if hasattr(self, "_hook_handle"):
            self._hook_handle.remove()
