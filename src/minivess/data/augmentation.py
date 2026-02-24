from __future__ import annotations

from torchio.transforms import (
    Compose,
    RandomBiasField,
    RandomGamma,
    RandomNoise,
)


def build_intensity_augmentation() -> Compose:
    """Build TorchIO intensity augmentation pipeline."""
    return Compose(
        [
            RandomNoise(std=0.01, p=0.2),
            RandomGamma(log_gamma=(-0.3, 0.3), p=0.2),
            RandomBiasField(coefficients=0.5, p=0.2),
        ]
    )
