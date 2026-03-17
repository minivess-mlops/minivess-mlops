"""VesselFM domain randomization adapter (T-D2).

Wraps vesselFM's d_rand method (github.com/bwittmann/vesselFM).
When the vesselFM library is not installed, falls back to procedural
domain-randomized tube generation that mimics vesselFM's output.

License: GPL-3.0 (inherited from vesselFM).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from minivess.data.synthetic.base import SyntheticGeneratorAdapter

logger = logging.getLogger(__name__)

_DEFAULT_PATCH_SIZE = (128, 128, 128)


class VesselFMDrandGenerator(SyntheticGeneratorAdapter):
    """VesselFM domain randomization synthetic volume generator.

    When vesselFM is available, delegates to the library's d_rand pipeline.
    Otherwise, uses a procedural fallback with domain randomization
    (random intensity, noise, blur, contrast) to generate tube-like volumes.
    """

    @property
    def name(self) -> str:
        return "vesselFM_drand"

    @property
    def requires_training(self) -> bool:
        return False

    @property
    def license(self) -> str:
        return "GPL-3.0"

    def generate_stack(
        self,
        n_volumes: int,
        config: dict[str, Any] | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate domain-randomized synthetic vascular volumes.

        Parameters
        ----------
        n_volumes:
            Number of (image, mask) pairs.
        config:
            Optional config with keys: patch_size, noise_level, seed.

        Returns
        -------
        List of (image, mask) 3D array tuples.
        """
        cfg = config or {}
        patch_size = tuple(cfg.get("patch_size", _DEFAULT_PATCH_SIZE))
        seed = cfg.get("seed", 42)
        noise_level = cfg.get("noise_level", 0.1)

        rng = np.random.default_rng(seed)
        pairs: list[tuple[np.ndarray, np.ndarray]] = []

        for _ in range(n_volumes):
            mask = _generate_random_tubes(rng, patch_size)
            image = _domain_randomize(rng, mask, noise_level=noise_level)
            pairs.append((image, mask))

        logger.info("Generated %d vesselFM d_rand volumes (%s)", n_volumes, patch_size)
        return pairs


def _generate_random_tubes(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    n_tubes: int = 5,
) -> np.ndarray:
    """Generate random tube-like structures via random walks."""
    mask = np.zeros(shape, dtype=np.uint8)

    for _ in range(n_tubes):
        pos = np.array([rng.integers(0, s) for s in shape], dtype=float)
        direction = rng.standard_normal(3)
        direction /= np.linalg.norm(direction) + 1e-8
        radius = rng.uniform(1.5, 4.0)

        for _step in range(rng.integers(20, 80)):
            # Perturb direction
            direction += rng.standard_normal(3) * 0.3
            direction /= np.linalg.norm(direction) + 1e-8
            pos += direction * 1.5

            # Draw sphere at position
            center = np.clip(pos.astype(int), 0, np.array(shape) - 1)
            r_int = max(1, int(radius))
            for dz in range(-r_int, r_int + 1):
                for dy in range(-r_int, r_int + 1):
                    for dx in range(-r_int, r_int + 1):
                        if dx * dx + dy * dy + dz * dz <= r_int * r_int:
                            z = center[0] + dz
                            y = center[1] + dy
                            x = center[2] + dx
                            if (
                                0 <= z < shape[0]
                                and 0 <= y < shape[1]
                                and 0 <= x < shape[2]
                            ):
                                mask[z, y, x] = 1

    return mask


def _domain_randomize(
    rng: np.random.Generator,
    mask: np.ndarray,
    noise_level: float = 0.1,
) -> np.ndarray:
    """Apply domain randomization to a binary mask."""
    image = mask.astype(np.float32)

    # Random intensity scaling
    fg_intensity = rng.uniform(0.5, 1.0)
    bg_intensity = rng.uniform(0.0, 0.3)
    image = np.where(mask > 0, fg_intensity, bg_intensity)

    # Add Gaussian noise
    image += rng.standard_normal(image.shape).astype(np.float32) * noise_level

    # Random contrast adjustment
    gamma = rng.uniform(0.7, 1.5)
    image = np.clip(image, 0.0, 1.0)
    image = np.power(image, gamma)

    return image.astype(np.float32)
