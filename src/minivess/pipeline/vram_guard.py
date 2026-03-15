"""VRAM-aware validation guard for GPU training.

Replaces fragile val_interval sentinel with explicit VRAM budget check.
Models that need more VRAM for validation than available are skipped with
a clear log message instead of silently setting val_interval > max_epochs.

See: #710 (sam3_hybrid val_loss=NaN on RTX 4090)
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# Minimum VRAM (MB) required for validation, per model family.
# Includes model weights + inference overhead + sliding window buffer.
# Models not listed here are assumed to fit on any GPU.
_VALIDATION_VRAM_REQUIREMENTS_MB: dict[str, int] = {
    "sam3_hybrid": 12000,  # 6.65 GB model + 5 GB inference = ~11.65 GB
    "sam3_topolora": 16000,  # LoRA gradients flow through full encoder
}


def get_available_vram_mb() -> int:
    """Return total GPU VRAM in MB, or 0 if no GPU available."""
    if not torch.cuda.is_available():
        return 0
    props = torch.cuda.get_device_properties(0)
    return int(props.total_memory / (1024 * 1024))


def should_skip_validation(
    model_family: str,
    available_vram_mb: int,
) -> bool:
    """Determine whether validation should be skipped due to VRAM constraints.

    Parameters
    ----------
    model_family:
        Model family name (e.g., "sam3_hybrid", "dynunet").
    available_vram_mb:
        Available GPU VRAM in MB (0 for CPU).

    Returns
    -------
    True if validation should be skipped (insufficient VRAM).
    """
    required = _VALIDATION_VRAM_REQUIREMENTS_MB.get(model_family)
    if required is None:
        # Model has no VRAM requirement listed — assume it fits
        return False

    if available_vram_mb < required:
        logger.warning(
            "Validation skipped: %s requires %d MB VRAM for validation, "
            "but only %d MB available. Train-only mode.",
            model_family,
            required,
            available_vram_mb,
        )
        return True

    return False
