"""GPU VRAM pre-flight check for SAM3 adapters.

SAM3 ViT-32L (848M params, BF16) has different VRAM requirements by task:

- **Inference** (frozen model, torch.no_grad): ~6 GB minimum (community-measured).
  Comfortable at 8 GB; reliable at 12 GB+.
- **Training / LoRA** (gradients + optimizer states): ~16 GB minimum (estimated).
  V2 TopoLoRA: ~13 GB at BS=1, ~22 GB at BS=2 (OOM on L4).
  V3 Hybrid: ~7.2 GB at BS=1, ~14.4 GB at BS=2.

Sources: GitHub Issues #200, #235, #307 at facebookresearch/sam3;
debuggercafe.com SAM3 memory benchmarks; Roboflow SAM3 guide.
Batch-size-specific data from 8th debug factorial pass (2026-03-24).

Usage::

    from minivess.adapters.sam3_vram_check import check_sam3_vram

    # For training (default — most conservative):
    check_sam3_vram(variant="sam3_vanilla", mode="training")

    # For training with batch_size awareness:
    check_sam3_vram(variant="sam3_topolora", mode="training", batch_size=2)

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


def check_sam3_vram(
    variant: str = "unknown",
    mode: str = "training",
    batch_size: int = 1,
) -> None:
    """Raise RuntimeError if GPU VRAM is below the SAM3 minimum for the given mode.

    Modes
    -----
    ``"training"`` (default):
        Enforces ≥16 GB at batch_size=1. For batch_size > 1, attempts to read
        per-batch-size VRAM data from the model profile YAML. If the profile
        indicates the requested batch_size needs more VRAM than available,
        raises RuntimeError with a batch-size-specific diagnostic.

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
    batch_size:
        Batch size for training. For batch_size > 1, checks per-batch-size
        VRAM data from model profiles if available. Defaults to 1.

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

    # Base threshold check (flat minimum regardless of batch_size)
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

    # Batch-size-aware check for training mode
    if mode == "training" and batch_size > 1:
        _check_batch_size_vram(variant, batch_size, vram_mb, gpu_name)


def _check_batch_size_vram(
    variant: str,
    batch_size: int,
    available_vram_mb: int,
    gpu_name: str,
) -> None:
    """Check per-batch-size VRAM requirement from model profile data.

    Reads per_batch_size data from the model profile YAML. If the profile
    has measured data for the requested batch_size and it exceeds available
    VRAM, raises RuntimeError with an actionable diagnostic.

    This is best-effort: if no profile or no per_batch_size data exists,
    logs a warning and returns (the base threshold check already passed).
    """
    try:
        from minivess.config.model_profiles import load_model_profile

        profile = load_model_profile(variant)
    except FileNotFoundError:
        logger.debug(
            "No model profile for %r — skipping batch-size VRAM check.", variant
        )
        return

    if profile.vram is None or profile.vram.per_batch_size is None:
        logger.debug(
            "Model profile %r has no per_batch_size VRAM data — "
            "skipping batch-size check.",
            variant,
        )
        return

    # Look up the estimated VRAM for this batch_size
    estimated_gb: float | None = None
    if batch_size in profile.vram.per_batch_size:
        estimated_gb = profile.vram.per_batch_size[batch_size]
    elif len(profile.vram.per_batch_size) >= 2:
        # Linear extrapolation from available measurements
        sorted_bs = sorted(profile.vram.per_batch_size.keys())
        bs_low, bs_high = sorted_bs[0], sorted_bs[-1]
        vram_low = profile.vram.per_batch_size[bs_low]
        vram_high = profile.vram.per_batch_size[bs_high]
        slope = (vram_high - vram_low) / (bs_high - bs_low)
        estimated_gb = vram_low + (batch_size - bs_low) * slope

    if estimated_gb is None:
        return

    available_gb = available_vram_mb / 1024

    if estimated_gb > available_gb:
        msg = (
            f"SAM3 {variant!r} at batch_size={batch_size} needs ~{estimated_gb:.1f} GB, "
            f"but only {available_gb:.1f} GB available on {gpu_name!r}. "
            f"Reduce batch_size to 1 or use a GPU with more VRAM."
        )
        logger.error(msg)
        raise RuntimeError(msg)

    margin_gb = available_gb - estimated_gb
    if margin_gb < 3.0:
        logger.warning(
            "SAM3 %r at batch_size=%d needs ~%.1f GB — margin %.1f GB on %r. "
            "DANGEROUS: may OOM under peak activation spikes. "
            "Consider reducing batch_size to 1.",
            variant,
            batch_size,
            estimated_gb,
            margin_gb,
            gpu_name,
        )
