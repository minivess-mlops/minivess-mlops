"""Debug tube generator — random-walk 3D tubes for fast test fixtures.

Wraps the logic from ``minivess.data.debug_dataset`` as a
``SyntheticGeneratorAdapter`` so it participates in the registry and
can be selected via ``generate_stack(method='debug')``.

This generator is CPU-only, requires no training, and produces volumes
quickly.  The tubes are NOT morphologically realistic vasculature — they
are purely for smoke tests and drift-detection pipeline validation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from minivess.data.synthetic.base import SyntheticGeneratorAdapter


class DebugTubeGenerator(SyntheticGeneratorAdapter):
    """Generate random-walk tube volumes with binary ground-truth masks.

    Parameters via *config*:
        shape: Tuple[int, int, int]  — volume dimensions (default 64³).
        n_tubes: int                 — tubes per volume (default 5).
        tube_radius: int             — radius in voxels (default 2).
        noise_std: float             — Gaussian noise σ (default 0.05).
        seed: int | None             — reproducibility seed.
    """

    @property
    def name(self) -> str:
        return "debug"

    @property
    def requires_training(self) -> bool:
        return False

    def generate_stack(
        self,
        n_volumes: int,
        config: dict[str, Any] | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate *n_volumes* random-walk tube volumes."""
        cfg = config or {}
        shape = tuple(cfg.get("shape", (64, 64, 64)))
        n_tubes: int = cfg.get("n_tubes", 5)
        tube_radius: int = cfg.get("tube_radius", 2)
        noise_std: float = cfg.get("noise_std", 0.05)
        seed: int | None = cfg.get("seed", None)

        rng = np.random.default_rng(seed)
        results: list[tuple[np.ndarray, np.ndarray]] = []

        for _ in range(n_volumes):
            image, mask = _generate_tube_volume(
                shape=shape,
                n_tubes=n_tubes,
                tube_radius=tube_radius,
                noise_std=noise_std,
                rng=rng,
            )
            results.append((image, mask))

        return results


def _generate_tube_volume(
    *,
    shape: tuple[int, ...],
    n_tubes: int,
    tube_radius: int,
    noise_std: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a single 3-D volume with random-walk tubes.

    Returns (image, mask) where image is float32 and mask is uint8.
    """
    d, h, w = shape
    mask = np.zeros(shape, dtype=np.uint8)

    for _ in range(n_tubes):
        # Random start point
        pos = np.array(
            [rng.integers(0, d), rng.integers(0, h), rng.integers(0, w)],
            dtype=np.float64,
        )
        # Random direction with drift
        direction = rng.standard_normal(3)
        direction = direction / (np.linalg.norm(direction) + 1e-8)

        n_steps = rng.integers(max(d, h, w) // 2, max(d, h, w))
        for _step in range(n_steps):
            # Perturb direction slightly for curvature
            perturbation = rng.standard_normal(3) * 0.3
            direction = direction + perturbation
            direction = direction / (np.linalg.norm(direction) + 1e-8)

            pos = pos + direction
            # Clamp to volume bounds
            iz, iy, ix = int(round(pos[0])), int(round(pos[1])), int(round(pos[2]))

            # Draw sphere at current position
            for dz in range(-tube_radius, tube_radius + 1):
                for dy in range(-tube_radius, tube_radius + 1):
                    for dx in range(-tube_radius, tube_radius + 1):
                        if dz * dz + dy * dy + dx * dx <= tube_radius * tube_radius:
                            nz, ny, nx = iz + dz, iy + dy, ix + dx
                            if 0 <= nz < d and 0 <= ny < h and 0 <= nx < w:
                                mask[nz, ny, nx] = 1

    # Create image: bright foreground + noise
    image = mask.astype(np.float32) * 0.8
    image += rng.normal(0, noise_std, shape).astype(np.float32)
    image = np.clip(image, 0.0, 1.0)

    return image, mask
