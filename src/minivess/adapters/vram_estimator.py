"""Pre-training VRAM estimation and budget checking.

Reads model profile YAML data (measured per-batch-size VRAM) to estimate
whether a given model + batch_size combination will fit in the available
GPU VRAM. Raises RuntimeError when it will not — failing loudly before
wasting spot instance time on an inevitable OOM.

Usage::

    from minivess.adapters.vram_estimator import check_vram_budget

    # Raises RuntimeError if sam3_topolora at BS=2 won't fit on a 24 GB GPU:
    check_vram_budget("sam3_topolora", batch_size=2, available_vram_gb=24.0)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from minivess.config.model_profiles import load_model_profile

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Safety margin: warn at 90% utilization, reject at 100%.
_WARN_THRESHOLD: float = 0.90


def estimate_training_vram(
    model_family: str,
    batch_size: int,
    search_dirs: list[Path] | None = None,
) -> float:
    """Estimate training VRAM in GB for a model at a given batch size.

    Resolution order:
    1. If the model profile has ``vram.per_batch_size[batch_size]``, return that
       directly (measured/estimated data from real runs).
    2. If ``per_batch_size`` exists but not for the requested ``batch_size``,
       linearly extrapolate from the closest available measurement.
    3. If no ``per_batch_size`` data, fall back to ``vram.training_gb``
       (assumed to be for the ``measured_config.batch_size``).

    Parameters
    ----------
    model_family:
        Model profile name (e.g., ``"sam3_topolora"``, ``"dynunet"``).
    batch_size:
        Batch size for the training run.
    search_dirs:
        Optional custom directories for model profile YAML lookup.

    Returns
    -------
    float
        Estimated training VRAM in GB.

    Raises
    ------
    FileNotFoundError
        If no model profile is found for ``model_family``.
    ValueError
        If the profile has no VRAM data (``vram`` is None or ``training_gb`` is None).
    """
    profile = load_model_profile(model_family, search_dirs=search_dirs)

    if profile.vram is None:
        msg = (
            f"Model profile '{model_family}' has no VRAM data. "
            f"Cannot estimate training VRAM."
        )
        raise ValueError(msg)

    # Case 1: exact per_batch_size match
    if profile.vram.per_batch_size is not None and batch_size in profile.vram.per_batch_size:
        return profile.vram.per_batch_size[batch_size]

    # Case 2: extrapolate from per_batch_size data
    if profile.vram.per_batch_size is not None and len(profile.vram.per_batch_size) >= 2:
        # Use the two closest measurements to linearly extrapolate
        sorted_bs = sorted(profile.vram.per_batch_size.keys())
        bs_low, bs_high = sorted_bs[0], sorted_bs[-1]
        vram_low = profile.vram.per_batch_size[bs_low]
        vram_high = profile.vram.per_batch_size[bs_high]
        # Linear extrapolation: vram = vram_low + (batch_size - bs_low) * slope
        slope = (vram_high - vram_low) / (bs_high - bs_low)
        return vram_low + (batch_size - bs_low) * slope

    # Case 3: single per_batch_size entry — scale linearly from measured reference
    if profile.vram.per_batch_size is not None and len(profile.vram.per_batch_size) == 1:
        ref_bs = next(iter(profile.vram.per_batch_size))
        ref_vram = profile.vram.per_batch_size[ref_bs]
        if ref_bs == 0:
            msg = f"Model profile '{model_family}' has per_batch_size with batch_size=0."
            raise ValueError(msg)
        per_sample = ref_vram / ref_bs
        return per_sample * batch_size

    # Case 4: fall back to training_gb with measured_config.batch_size scaling
    if profile.vram.training_gb is None:
        msg = (
            f"Model profile '{model_family}' has no training_gb or per_batch_size data. "
            f"Cannot estimate training VRAM."
        )
        raise ValueError(msg)

    measured_bs = 1
    if (
        profile.vram.measured_config is not None
        and "batch_size" in profile.vram.measured_config
        and profile.vram.measured_config["batch_size"] is not None
    ):
        measured_bs = int(profile.vram.measured_config["batch_size"])

    if measured_bs == 0:
        msg = f"Model profile '{model_family}' has measured_config.batch_size=0."
        raise ValueError(msg)

    # Linear scaling from measured reference point
    per_sample = profile.vram.training_gb / measured_bs
    return per_sample * batch_size


def check_vram_budget(
    model_family: str,
    batch_size: int,
    available_vram_gb: float,
    search_dirs: list[Path] | None = None,
) -> None:
    """Raise RuntimeError if estimated VRAM exceeds available GPU VRAM.

    Also logs a WARNING if estimated VRAM exceeds 90% of available VRAM
    (dangerous margin that may OOM under peak activation spikes).

    Parameters
    ----------
    model_family:
        Model profile name.
    batch_size:
        Batch size for the training run.
    available_vram_gb:
        Available GPU VRAM in GB (e.g., 24.0 for L4).
    search_dirs:
        Optional custom directories for model profile YAML lookup.

    Raises
    ------
    RuntimeError
        When estimated VRAM exceeds ``available_vram_gb``.
    """
    estimated = estimate_training_vram(
        model_family, batch_size, search_dirs=search_dirs
    )

    if estimated > available_vram_gb:
        msg = (
            f"VRAM budget exceeded: {model_family} at batch_size={batch_size} "
            f"needs ~{estimated:.1f} GB, but only {available_vram_gb:.1f} GB available. "
            f"Reduce batch_size or use a GPU with more VRAM."
        )
        logger.error(msg)
        raise RuntimeError(msg)

    utilization = estimated / available_vram_gb if available_vram_gb > 0 else 1.0
    if utilization >= _WARN_THRESHOLD:
        logger.warning(
            "VRAM utilization WARNING: %s at batch_size=%d needs ~%.1f GB "
            "(%.0f%% of %.1f GB available). Risk of OOM under peak activation spikes. "
            "Consider reducing batch_size.",
            model_family,
            batch_size,
            estimated,
            utilization * 100,
            available_vram_gb,
        )
    else:
        logger.info(
            "VRAM budget OK: %s at batch_size=%d needs ~%.1f GB "
            "(%.0f%% of %.1f GB available).",
            model_family,
            batch_size,
            estimated,
            utilization * 100,
            available_vram_gb,
        )
