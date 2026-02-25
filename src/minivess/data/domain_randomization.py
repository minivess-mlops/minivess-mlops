"""SynthICL domain randomization for in-context learning (Terms et al., 2025).

Provides synthetic domain randomization augmentation for medical image
segmentation. Generates randomized intensity, contrast, noise, and blur
variations to improve model robustness to distribution shift.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

if TYPE_CHECKING:
    from numpy.typing import NDArray


class RandomizationParam(StrEnum):
    """Domain randomization parameter types."""

    INTENSITY = "intensity"
    CONTRAST = "contrast"
    NOISE = "noise"
    BLUR = "blur"
    SPACING = "spacing"


@dataclass
class DomainRandomizationConfig:
    """Configuration for domain randomization pipeline.

    Parameters
    ----------
    intensity_range:
        Min/max intensity scaling factor.
    noise_std_range:
        Min/max standard deviation for Gaussian noise.
    contrast_range:
        Min/max gamma for contrast adjustment.
    blur_sigma_range:
        Min/max sigma for Gaussian blur.
    seed:
        Random seed for reproducibility. When ``None``, falls back to
        ``torch.initial_seed()`` for deterministic behavior.
    """

    intensity_range: tuple[float, float] = (0.7, 1.3)
    noise_std_range: tuple[float, float] = (0.0, 0.05)
    contrast_range: tuple[float, float] = (0.5, 2.0)
    blur_sigma_range: tuple[float, float] = (0.0, 1.0)
    seed: int | None = None


def _resolve_seed(seed: int | None) -> int:
    """Resolve seed, falling back to torch.initial_seed() when None.

    Parameters
    ----------
    seed:
        Explicit seed or None.

    Returns
    -------
    Resolved integer seed.
    """
    if seed is not None:
        return seed
    return int(torch.initial_seed() % (2**31))


class SyntheticVesselGenerator:
    """Generate synthetic vessel-like structures and randomized volumes.

    Parameters
    ----------
    seed:
        Random seed for reproducibility.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def random_tubular_mask(self, shape: tuple[int, ...]) -> NDArray:
        """Generate a random tubular binary mask.

        Creates synthetic vessel-like structures using random walks
        through the volume with Gaussian smoothing.

        Parameters
        ----------
        shape:
            Shape of the output volume.
        """
        mask = np.zeros(shape, dtype=np.float32)
        n_tubes = self._rng.integers(3, 8)

        for _ in range(n_tubes):
            # Random starting point (inside volume)
            pos = np.array([
                self._rng.integers(2, s - 2) for s in shape
            ], dtype=np.float32)

            n_steps = self._rng.integers(10, max(shape) * 2)
            for _step in range(n_steps):
                # Random walk direction
                direction = self._rng.normal(0, 1, len(shape))
                direction = direction / (np.linalg.norm(direction) + 1e-8)
                pos = pos + direction

                # Clamp to volume bounds
                idx = tuple(
                    int(np.clip(p, 0, s - 1))
                    for p, s in zip(pos, shape, strict=True)
                )
                mask[idx] = 1.0

        # Dilate with Gaussian to create tube-like structures
        mask = gaussian_filter(mask, sigma=1.0)
        return (mask > 0.1).astype(np.uint8)

    def randomize_intensity(
        self,
        volume: NDArray,
        scale_range: tuple[float, float] = (0.7, 1.3),
    ) -> NDArray:
        """Apply random intensity scaling.

        Parameters
        ----------
        volume:
            Input volume.
        scale_range:
            Min/max scaling factor.
        """
        scale = self._rng.uniform(*scale_range)
        return (volume * scale).astype(volume.dtype)

    def randomize_contrast(
        self,
        volume: NDArray,
        gamma_range: tuple[float, float] = (0.5, 2.0),
    ) -> NDArray:
        """Apply random gamma contrast adjustment.

        Parameters
        ----------
        volume:
            Input volume (assumed [0, 1] range).
        gamma_range:
            Min/max gamma value.
        """
        gamma = self._rng.uniform(*gamma_range)
        # Clamp to [0, 1] to avoid negative values in power
        clamped = np.clip(volume, 0, 1)
        return np.power(clamped, gamma).astype(volume.dtype)

    def add_noise(self, volume: NDArray, std: float = 0.05) -> NDArray:
        """Add Gaussian noise to the volume.

        Parameters
        ----------
        volume:
            Input volume.
        std:
            Standard deviation of Gaussian noise.
        """
        noise = self._rng.normal(0, std, size=volume.shape).astype(volume.dtype)
        return volume + noise


class DomainRandomizationPipeline:
    """Orchestrates multi-parameter domain randomization.

    Applies intensity scaling, contrast adjustment, noise injection,
    and Gaussian blur in sequence to create domain-randomized samples.

    When ``config.seed`` is None, falls back to ``torch.initial_seed()``
    so that determinism is preserved when a global torch seed is set.

    Parameters
    ----------
    config:
        Randomization configuration.
    """

    def __init__(self, config: DomainRandomizationConfig) -> None:
        self.config = config
        self.resolved_seed: int = _resolve_seed(config.seed)
        self._generator = SyntheticVesselGenerator(seed=self.resolved_seed)
        self._rng = np.random.default_rng(self.resolved_seed)

    def apply(
        self, volume: NDArray, mask: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Apply domain randomization to a volume.

        Intensity-only transforms — the mask is returned unchanged.

        Parameters
        ----------
        volume:
            Input volume.
        mask:
            Segmentation mask (returned unchanged).
        """
        result = volume.copy()

        # Intensity scaling
        result = self._generator.randomize_intensity(
            result, self.config.intensity_range,
        )

        # Contrast adjustment
        result = self._generator.randomize_contrast(
            result, self.config.contrast_range,
        )

        # Noise injection
        noise_std = self._rng.uniform(*self.config.noise_std_range)
        result = self._generator.add_noise(result, std=noise_std)

        # Gaussian blur
        sigma = self._rng.uniform(*self.config.blur_sigma_range)
        if sigma > 0.01:
            result = gaussian_filter(result, sigma=sigma).astype(result.dtype)

        return result, mask.copy()

    def generate_batch(
        self,
        volume: NDArray,
        mask: NDArray,
        n: int = 10,
    ) -> list[tuple[NDArray, NDArray]]:
        """Generate N domain-randomized samples from a single volume.

        Parameters
        ----------
        volume:
            Template volume.
        mask:
            Template mask.
        n:
            Number of randomized samples.
        """
        batch: list[tuple[NDArray, NDArray]] = []
        for i in range(n):
            # Use different sub-seed per sample
            self._generator = SyntheticVesselGenerator(
                seed=self.resolved_seed + i + 1,
            )
            self._rng = np.random.default_rng(
                self.resolved_seed + i + 1,
            )
            aug_vol, aug_mask = self.apply(volume, mask)
            batch.append((aug_vol, aug_mask))
        return batch

    def to_markdown(self) -> str:
        """Generate a report of the randomization configuration."""
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        sections = [
            "# SynthICL Domain Randomization Configuration",
            "",
            f"**Generated:** {now}",
            "",
            "## Parameters",
            "",
            "| Parameter | Range |",
            "|-----------|-------|",
            f"| Intensity scaling | {self.config.intensity_range} |",
            f"| Contrast (gamma) | {self.config.contrast_range} |",
            f"| Noise std | {self.config.noise_std_range} |",
            f"| Blur sigma | {self.config.blur_sigma_range} |",
            f"| Seed | {self.config.seed} (resolved: {self.resolved_seed}) |",
            "",
            "## Pipeline Steps",
            "",
            "1. **Intensity scaling** — Multiply volume by random factor",
            "2. **Contrast adjustment** — Gamma correction with random gamma",
            "3. **Noise injection** — Additive Gaussian noise",
            "4. **Gaussian blur** — Random sigma smoothing",
            "",
        ]
        return "\n".join(sections)
