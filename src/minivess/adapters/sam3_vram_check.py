"""GPU VRAM pre-flight check for SAM3 adapters.

SAM3 ViT-32L (848M params, BF16) has different VRAM requirements by task:

- **Inference** (frozen model, torch.no_grad): ~6 GB minimum (community-measured).
  Comfortable at 8 GB; reliable at 12 GB+.
- **Training / LoRA** (gradients + optimizer states): ~16 GB minimum (estimated).
  V2 TopoLoRA: ~12-16 GB. V3 Hybrid: ~18-22 GB.

Sources: GitHub Issues #200, #235, #307 at facebookresearch/sam3;
debuggercafe.com SAM3 memory benchmarks; Roboflow SAM3 guide.

Usage::

    from minivess.adapters.sam3_vram_check import check_sam3_vram

    # For training (default — most conservative):
    check_sam3_vram(variant="sam3_vanilla", mode="training")

    # For inference only (less restrictive):
    check_sam3_vram(variant="sam3_vanilla", mode="inference")
"""

from __future__ import annotations

import logging

from minivess.config.adaptive_profiles import detect_hardware

logger = logging.getLogger(__name__)

# Community-measured minimums (verified via GitHub issues and benchmarks).
# Inference: single image, BF16, torch.no_grad(). Confirmed ~4-6 GB.
# Training:  frozen-encoder + decoder, BF16. LoRA minimum ~12-16 GB.
MIN_VRAM_INFERENCE_MB: int = 6_144  # 6 GB — hard floor for single-image inference
MIN_VRAM_TRAINING_MB: int = 16_384  # 16 GB — minimum for any SAM3 training (LoRA)


def check_sam3_vram(variant: str = "unknown", mode: str = "training") -> None:
    """Raise RuntimeError if GPU VRAM is below the SAM3 minimum for the given mode.

    Modes
    -----
    ``"training"`` (default):
        Enforces ≥16 GB. Required for LoRA training (V2 TopoLoRA, V3 Hybrid).
        V1 Vanilla with fully frozen encoder may work at 8 GB, but the 16 GB
        gate is kept as the safe default for any training invocation.

    ``"inference"``:
        Enforces ≥6 GB. Single-image BF16 inference uses ~4-6 GB.
        8 GB (RTX 2070 Super) is marginal but usable with
        ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``.

    Parameters
    ----------
    variant:
        SAM3 variant name for the error message (e.g. ``"sam3_vanilla"``).
    mode:
        ``"training"`` or ``"inference"``. Defaults to ``"training"``.

    Raises
    ------
    RuntimeError
        When detected GPU VRAM is below the threshold for the requested mode.
    ValueError
        When ``mode`` is not ``"training"`` or ``"inference"``.
    """
    if mode not in ("training", "inference"):
        msg = f"mode must be 'training' or 'inference', got {mode!r}"
        raise ValueError(msg)

    hw = detect_hardware()
    vram_mb = hw.gpu_vram_mb
    gpu_name = hw.gpu_name or "unknown GPU"

    threshold_mb = MIN_VRAM_TRAINING_MB if mode == "training" else MIN_VRAM_INFERENCE_MB
    threshold_gb = threshold_mb / 1024

    if vram_mb < threshold_mb:
        vram_gb = vram_mb / 1024
        if mode == "training":
            guidance = (
                "For training, use a machine with an A100-40GB, H100, or RTX 4090.\n"
                "For inference only, pass mode='inference' (requires ≥6 GB)."
            )
        else:
            guidance = (
                "Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to help.\n"
                "If still OOM, reduce input resolution or process fewer slices at once."
            )
        msg = (
            f"SAM3 {mode} requires ≥{threshold_gb:.0f} GB GPU VRAM. "
            f"Detected: {vram_gb:.1f} GB on {gpu_name!r}.\n"
            f"Variant {variant!r} cannot {mode} on this hardware.\n"
            f"{guidance}"
        )
        logger.error(msg)
        raise RuntimeError(msg)
