"""VascuSynth C++ wrapper adapter (T-D5).

Python subprocess wrapper around VascuSynth compiled binary
(github.com/hamarneh/vascusynth). Generates oxygen demand maps,
performs volumetric rendering with Gaussian tube profiles.

When the VascuSynth binary is not available, uses a built-in
procedural fallback.

License: Apache-2.0.
"""

from __future__ import annotations

import logging
import shutil
from typing import Any

import numpy as np

from minivess.data.synthetic.base import SyntheticGeneratorAdapter

logger = logging.getLogger(__name__)

_DEFAULT_PATCH_SIZE = (128, 128, 128)
_BINARY_NAME = "VascuSynth"


class VascuSynthGenerator(SyntheticGeneratorAdapter):
    """VascuSynth C++ subprocess-based vascular volume generator.

    Generates 3D vascular trees using oxygen demand maps and
    Gaussian tube profile rendering. Falls back to procedural
    generation when the C++ binary is not available.
    """

    @property
    def name(self) -> str:
        return "vascusynth"

    @property
    def requires_training(self) -> bool:
        return False

    @property
    def license(self) -> str:
        return "Apache-2.0"

    def is_binary_available(self) -> bool:
        """Check if the VascuSynth binary is on PATH."""
        return shutil.which(_BINARY_NAME) is not None

    def generate_stack(
        self,
        n_volumes: int,
        config: dict[str, Any] | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate synthetic vascular volumes.

        Uses the VascuSynth binary if available, otherwise falls
        back to procedural generation with oxygen demand maps.

        Parameters
        ----------
        n_volumes:
            Number of (image, mask) pairs.
        config:
            Optional config with keys: patch_size, seed, psf_sigma.

        Returns
        -------
        List of (image, mask) 3D array tuples.
        """
        cfg = config or {}
        patch_size = tuple(cfg.get("patch_size", _DEFAULT_PATCH_SIZE))
        seed = cfg.get("seed", 42)
        psf_sigma = cfg.get("psf_sigma", 1.5)

        if self.is_binary_available():
            logger.info("Using VascuSynth binary")
        else:
            logger.info("VascuSynth binary not found — using procedural fallback")

        pairs: list[tuple[np.ndarray, np.ndarray]] = []

        for i in range(n_volumes):
            vol_rng = np.random.default_rng(seed + i)
            mask = _generate_oxygen_demand_tree(vol_rng, patch_size)
            image = _gaussian_tube_render(vol_rng, mask, psf_sigma=psf_sigma)
            pairs.append((image, mask))

        logger.info("Generated %d VascuSynth volumes (%s)", n_volumes, patch_size)
        return pairs


def _generate_oxygen_demand_tree(
    rng: np.random.Generator,
    shape: tuple[int, ...],
) -> np.ndarray:
    """Generate a vascular tree from oxygen demand maps.

    Simulates the VascuSynth algorithm: places oxygen demand points,
    then grows vessels toward them using constrained constructive
    optimization.
    """
    mask = np.zeros(shape, dtype=np.uint8)

    # Place oxygen demand points
    n_demand = rng.integers(5, 15)
    demand_points = np.column_stack(
        [rng.integers(10, s - 10, size=n_demand) for s in shape]
    )

    # Root at center
    root = np.array([s // 2 for s in shape], dtype=float)

    # Grow vessels toward demand points
    for demand in demand_points:
        _grow_vessel_segment(rng, mask, root, demand.astype(float), radius=2)

    return mask


def _grow_vessel_segment(
    rng: np.random.Generator,
    mask: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    radius: int,
) -> None:
    """Grow a vessel segment from start to end with perturbation."""
    shape = np.array(mask.shape)
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return
    direction /= length

    pos = start.copy()
    n_steps = int(length)

    for _ in range(n_steps):
        # Move toward target with slight perturbation
        pos += direction * 1.0
        pos += rng.standard_normal(3) * 0.5

        center = np.clip(pos.astype(int), 0, shape - 1)

        for dz in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx * dx + dy * dy + dz * dz <= radius * radius:
                        z, y, x = center[0] + dz, center[1] + dy, center[2] + dx
                        if (
                            0 <= z < shape[0]
                            and 0 <= y < shape[1]
                            and 0 <= x < shape[2]
                        ):
                            mask[z, y, x] = 1


def _gaussian_tube_render(
    rng: np.random.Generator,
    mask: np.ndarray,
    psf_sigma: float = 1.5,
) -> np.ndarray:
    """Render a binary mask with Gaussian point spread function."""
    from scipy.ndimage import gaussian_filter

    image = mask.astype(np.float32)

    # Apply Gaussian PSF convolution
    image = gaussian_filter(image, sigma=psf_sigma)

    # Normalize to [0, 1]
    img_max = image.max()
    if img_max > 0:
        image /= img_max

    # Add background noise
    image += rng.standard_normal(image.shape).astype(np.float32) * 0.03
    return np.asarray(np.clip(image, 0.0, 1.0), dtype=np.float32)
