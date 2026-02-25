from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minivess.config.models import DataConfig, TrainingConfig


@dataclass(frozen=True)
class ComputeProfile:
    """Hardware-specific compute profile for training.

    Parameters
    ----------
    name:
        Profile identifier.
    batch_size:
        Training batch size.
    patch_size:
        3D patch dimensions (D, H, W).
    num_workers:
        DataLoader worker count.
    mixed_precision:
        Whether to use AMP (automatic mixed precision).
    gradient_accumulation_steps:
        Number of gradient accumulation steps before optimizer step.
    """

    name: str
    batch_size: int
    patch_size: tuple[int, int, int]
    num_workers: int
    mixed_precision: bool
    gradient_accumulation_steps: int


_PROFILES: dict[str, ComputeProfile] = {
    "cpu": ComputeProfile(
        name="cpu",
        batch_size=1,
        patch_size=(64, 64, 16),
        num_workers=2,
        mixed_precision=False,
        gradient_accumulation_steps=4,
    ),
    "gpu_low": ComputeProfile(
        name="gpu_low",
        batch_size=2,
        patch_size=(96, 96, 24),
        num_workers=4,
        mixed_precision=True,
        gradient_accumulation_steps=2,
    ),
    "gpu_high": ComputeProfile(
        name="gpu_high",
        batch_size=4,
        patch_size=(128, 128, 32),
        num_workers=8,
        mixed_precision=True,
        gradient_accumulation_steps=1,
    ),
    "dgx_spark": ComputeProfile(
        name="dgx_spark",
        batch_size=8,
        patch_size=(128, 128, 48),
        num_workers=12,
        mixed_precision=True,
        gradient_accumulation_steps=1,
    ),
    "cloud_single": ComputeProfile(
        name="cloud_single",
        batch_size=8,
        patch_size=(128, 128, 64),
        num_workers=16,
        mixed_precision=True,
        gradient_accumulation_steps=1,
    ),
    "cloud_multi": ComputeProfile(
        name="cloud_multi",
        batch_size=32,
        patch_size=(128, 128, 64),
        num_workers=16,
        mixed_precision=True,
        gradient_accumulation_steps=1,
    ),
}


def get_compute_profile(name: str) -> ComputeProfile:
    """Look up a compute profile by name.

    Parameters
    ----------
    name:
        Profile identifier (e.g. ``"cpu"``, ``"gpu_low"``).

    Raises
    ------
    ValueError
        If the profile name is not recognized.
    """
    if name not in _PROFILES:
        available = ", ".join(sorted(_PROFILES))
        msg = f"Unknown compute profile: {name!r}. Available: {available}"
        raise ValueError(msg)
    return _PROFILES[name]


def list_profiles() -> list[str]:
    """Return all available profile names."""
    return sorted(_PROFILES)


def apply_profile(
    profile: ComputeProfile,
    data_config: DataConfig,
    training_config: TrainingConfig,
) -> None:
    """Apply a compute profile to data and training configs (mutates in-place).

    Parameters
    ----------
    profile:
        Compute profile to apply.
    data_config:
        Data configuration to modify.
    training_config:
        Training configuration to modify.
    """
    data_config.patch_size = profile.patch_size
    data_config.num_workers = profile.num_workers
    training_config.batch_size = profile.batch_size
    training_config.mixed_precision = profile.mixed_precision
