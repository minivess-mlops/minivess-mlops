"""GPU VRAM pre-flight check for SAM3 adapters.

SAM3 ViT-32L requires ≥16 GB GPU VRAM. This module enforces that requirement
before any SAM3 model weights are loaded, preventing silent low-VRAM failures.

Usage::

    from minivess.adapters.sam3_vram_check import check_sam3_vram

    check_sam3_vram(variant="sam3_vanilla")  # raises RuntimeError if < 16 GB
"""

from __future__ import annotations

import logging

from minivess.config.adaptive_profiles import detect_hardware

logger = logging.getLogger(__name__)

# SAM3 ViT-32L requires ≥16 GB GPU VRAM.
# V1 Vanilla: ~16 GB (encoder frozen, but full ViT-32L must be loaded)
# V2 TopoLoRA: ~18 GB (LoRA unfrozen, gradient checkpointing recommended)
# V3 Hybrid: ~22 GB (ViT-32L + DynUNet-3D branches)
MIN_VRAM_MB: int = 16_384  # 16 GB


def check_sam3_vram(variant: str = "unknown") -> None:
    """Raise RuntimeError if GPU VRAM is below the SAM3 minimum (16 GB).

    This is a hard pre-flight check — it fires before any SAM3 weights are
    loaded or any SAM3 class is instantiated. Call this at the top of
    ``build_adapter()`` for any SAM3 family.

    Parameters
    ----------
    variant:
        SAM3 variant name for the error message (e.g. ``"sam3_vanilla"``).
        Defaults to ``"unknown"`` if not provided.

    Raises
    ------
    RuntimeError
        When detected GPU VRAM is below ``MIN_VRAM_MB`` (16 GB).
    """
    hw = detect_hardware()
    vram_mb = hw.gpu_vram_mb
    gpu_name = hw.gpu_name or "unknown GPU"

    if vram_mb < MIN_VRAM_MB:
        vram_gb = vram_mb / 1024
        min_gb = MIN_VRAM_MB / 1024
        msg = (
            f"SAM3 requires ≥{min_gb:.0f} GB GPU VRAM. "
            f"Detected: {vram_gb:.1f} GB on {gpu_name!r}.\n"
            f"Variant {variant!r} cannot run on this hardware.\n"
            "Use a machine with an A100-40GB, H100, or RTX 4090."
        )
        logger.error(msg)
        raise RuntimeError(msg)
