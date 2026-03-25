"""Sam3TopoLoraAdapter — SAM3 + LoRA on FFN + Spatial Adapter + topology-aware loss.

Adapts the TopoLoRA-SAM architecture (Khazem et al., 2025, arXiv:2601.02273)
to SAM3 ViT-32L. Architecture matches the paper:

    1. Frozen SAM3 ViT-32L encoder
    2. Trainable LoRA on FFN layers ONLY (mlp/lin1/lin2) — NOT attention Q/K/V
    3. Trainable Spatial Adapter: Conv_DW(3x3) + Conv(1x1) + BN + GELU + residual
    4. Trainable mask decoder
    5. Loss: config-driven (factorial design varies loss as experimental factor)

Our adaptation: SAM3 ViT-32L (648M) instead of SAM1 ViT-B (93.7M).
Spatial Adapter operates on post-FPN 256-ch features (paper: raw ViT output).

References:
    Paper: https://arxiv.org/html/2601.02273v1
    Code: https://github.com/salimkhazem/Seglab
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from minivess.adapters.base import AdapterConfigInfo, ModelAdapter, SegmentationOutput
from minivess.adapters.sam3_backbone import SAM3_INPUT_SIZE, Sam3Backbone
from minivess.adapters.sam3_decoder import Sam3MaskDecoder

if TYPE_CHECKING:
    from minivess.config.models import ModelConfig

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """Low-Rank Adaptation layer wrapping an existing nn.Linear layer.

    Adds a low-rank decomposition: output = original(x) + (B @ A)(x) * scaling.
    SAM3 ViT-32L is pure Linear (no Conv2d in the transformer blocks), so only
    nn.Linear is supported. Conv2d LoRA would require a different decomposition.

    Parameters
    ----------
    original:
        The original nn.Linear layer to wrap.
    rank:
        Rank of the low-rank decomposition.
    alpha:
        Scaling factor (alpha / rank).
    dropout:
        Dropout probability for LoRA path.
    """

    def __init__(
        self,
        original: nn.Module,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(original, nn.Linear):
            msg = (
                f"LoRALinear only supports nn.Linear, got {type(original).__name__}. "
                f"SAM3 ViT-32L uses only Linear layers in its transformer blocks."
            )
            raise TypeError(msg)

        self.original = original
        self.rank = rank
        self.scaling = alpha / rank

        # Freeze the original layer
        for param in original.parameters():
            param.requires_grad = False

        in_features = original.in_features
        out_features = original.out_features
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def weight(self) -> Tensor:
        """Expose original layer's weight for HF code that accesses .weight.dtype."""
        return self.original.weight

    @property
    def bias(self) -> Tensor | None:
        """Expose original layer's bias for HF code that accesses .bias."""
        return self.original.bias

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: original(x) + LoRA(x).

        Handles FP16/FP32 dtype mismatch: SAM3 encoder runs in FP16
        but LoRA params are initialized in FP32. Cast input to match
        LoRA dtype for matmul, then cast result back to input dtype.
        """
        original_out: Tensor = self.original(x)
        input_dtype = x.dtype

        x_dropped = self.lora_dropout(x).to(self.lora_A.dtype)
        lora_out = (x_dropped @ self.lora_A.T @ self.lora_B.T) * self.scaling
        lora_out = lora_out.to(input_dtype)

        result: Tensor = original_out + lora_out
        return result


# Default LoRA target keywords — paper targets FFN layers ONLY.
# SegLab config: target_keywords: ["mlp", "fc1", "fc2"]
# Our SAM3 uses "mlp", "lin1", "lin2" naming.
LORA_FFN_KEYWORDS: tuple[str, ...] = ("mlp", "lin1", "lin2", "fc1", "fc2")


def _apply_lora_to_encoder(
    encoder: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    target_keywords: tuple[str, ...] = LORA_FFN_KEYWORDS,
) -> list[str]:
    """Apply LoRA adapters to FFN Linear layers in the encoder.

    Per TopoLoRA-SAM (Khazem et al., 2025): LoRA targets FFN layers ONLY
    (mlp.lin1/lin2). Attention Q/K/V projections are frozen — the paper
    argues they encode "more transferable relational patterns."

    Parameters
    ----------
    encoder:
        The SAM3 ViT encoder module.
    rank:
        LoRA rank (paper optimal: 16).
    alpha:
        LoRA scaling factor.
    dropout:
        LoRA dropout probability.
    target_keywords:
        Layer name substrings to match. Only nn.Linear layers whose
        full dotted name contains ANY of these keywords receive LoRA.
        Default: FFN-related keywords per the paper.

    Returns list of module names that received LoRA.
    """
    lora_targets: list[str] = []

    for name, module in list(encoder.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        # Filter: only target layers matching keywords (paper: FFN only)
        if not any(kw in name for kw in target_keywords):
            continue

        # Skip very small layers (< rank features)
        if module.in_features < rank:
            continue

        lora_layer = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)

        # Replace the module in the parent
        parts = name.split(".")
        parent = encoder
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], lora_layer)
        lora_targets.append(name)

    return lora_targets


class SpatialConvAdapter(nn.Module):
    """Spatial Adapter from TopoLoRA-SAM (Khazem et al., 2025).

    Lightweight depthwise-separable convolutional adapter applied to
    the image embedding tensor between encoder and decoder.

    Architecture (matching SegLab ConvAdapter):
        x → DW_Conv(3x3) → PW_Conv(1x1) → BN → GELU → PW_Conv(1x1) → + x (residual)

    Parameters
    ----------
    channels:
        Number of input/output channels (256 for SAM3 FPN output).
    hidden_dim:
        Hidden dimension for the pointwise bottleneck.
    kernel_size:
        Kernel size for depthwise convolution (default 3).
    """

    def __init__(
        self,
        channels: int = 256,
        hidden_dim: int = 256,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.dw_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            padding=pad,
            groups=channels,
            bias=False,
        )
        self.pw1 = nn.Conv2d(channels, hidden_dim, 1, bias=False)
        self.norm = nn.BatchNorm2d(hidden_dim)
        self.pw2 = nn.Conv2d(hidden_dim, channels, 1, bias=False)

        # Zero-init pw2 so residual dominates at initialization
        nn.init.zeros_(self.pw2.weight)

    def forward(self, x: Tensor) -> Tensor:
        """Forward with residual: x + pw2(gelu(norm(pw1(dw_conv(x)))))."""
        y = self.dw_conv(x)
        y = self.pw1(y)
        y = self.norm(y)
        y = F.gelu(y)
        y = self.pw2(y)
        result: Tensor = x + y
        return result


class Sam3TopoLoraAdapter(ModelAdapter):
    """SAM3 + LoRA on encoder FFN + trainable decoder.

    Parameters
    ----------
    config:
        ModelConfig with ``SAM3_TOPOLORA`` family and LoRA params.
    """

    def __init__(
        self,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self._lora_rank = config.lora_rank
        self._lora_alpha = config.lora_alpha

        # Gradient checkpointing: read from architecture_params (set by factorial config).
        # Reduces SAM3 TopoLoRA peak VRAM from ~22 GiB to ~10 GiB on L4.
        # Issue: #966 (A100 option), #940 (SAM3 OOM).
        _gradient_checkpointing = config.architecture_params.get(
            "gradient_checkpointing", False
        )

        # SAM3 backbone (NOT frozen yet — LoRA applied first)
        self.backbone = Sam3Backbone(
            config=config,
            freeze=False,
            gradient_checkpointing=_gradient_checkpointing,
        )

        # Apply LoRA to encoder, then freeze base weights
        lora_targets = _apply_lora_to_encoder(
            self.backbone.encoder,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
        )
        logger.info("LoRA applied to %d layers: %s", len(lora_targets), lora_targets)

        # Freeze FPN neck (no LoRA there)
        for param in self.backbone.fpn_neck.parameters():
            param.requires_grad = False

        # Spatial Adapter (Khazem 2025): Conv_DW(3x3) + Conv(1x1) + residual
        # Operates on 256-ch FPN output (our adaptation: post-FPN, not raw encoder)
        self.spatial_adapter = SpatialConvAdapter(channels=256)

        # Trainable mask decoder
        self.decoder = Sam3MaskDecoder(config=config)

        trainable = self.trainable_parameters()
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            "Sam3TopoLoraAdapter: %d trainable / %d total params (%.1f%%)",
            trainable,
            total,
            100 * trainable / max(total, 1),
        )

    def forward(self, images: Tensor, **kwargs: Any) -> SegmentationOutput:
        """Slice-by-slice forward with LoRA-adapted encoder.

        Parameters
        ----------
        images:
            Input 3D volume of shape (B, C, H, W, D) — MONAI dimension order.

        Returns
        -------
        SegmentationOutput with 2-class predictions.
        """
        # MONAI dimension order: (B, C, H, W, D) — depth is LAST
        b, c, h, w, d = images.shape
        slice_logits: list[Tensor] = []

        for z_idx in range(d):
            slice_2d = images[:, :, :, :, z_idx]  # (B, C, H, W)

            # FPN features (LoRA adapters are trainable, rest is frozen)
            fpn_features = self.backbone.extract_fpn_features(slice_2d)
            # Cast FP16 encoder output to FP32 for decoder (#680)
            fpn_features = fpn_features.float()

            # Spatial Adapter (Khazem 2025): refine spatial features
            fpn_features = self.spatial_adapter(fpn_features)

            # Decode to binary mask
            binary_logits = self.decoder(fpn_features)

            # Resize back to original spatial dims
            if binary_logits.shape[2] != h or binary_logits.shape[3] != w:
                binary_logits = F.interpolate(
                    binary_logits,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )

            two_class = self.decoder.binary_to_2class(binary_logits)
            slice_logits.append(two_class)

        logits_3d = torch.stack(slice_logits, dim=2)
        return self._build_output(logits_3d, "sam3_topolora")

    def get_config(self) -> AdapterConfigInfo:
        """Return adapter configuration info."""
        return self._build_config(
            variant="topolora",
            backbone="vit_32l",
            input_size=SAM3_INPUT_SIZE,
            lora_rank=self._lora_rank,
            lora_alpha=self._lora_alpha,
        )

    def trainable_parameters(self) -> int:
        """Count trainable parameters (LoRA + decoder)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
