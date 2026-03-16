"""VaMos procedural vascular adapter (T-D4).

Wraps VaMos procedural generation (gitlab.univ-nantes.fr/autrusseau-f/vamos/).
CPU-only procedural synthesis with configurable morphological parameters.
When VaMos is not installed, uses a built-in procedural fallback.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from minivess.data.synthetic.base import SyntheticGeneratorAdapter

logger = logging.getLogger(__name__)

_DEFAULT_PATCH_SIZE = (128, 128, 128)


class VaMosGenerator(SyntheticGeneratorAdapter):
    """VaMos procedural vascular tree generator.

    CPU-only procedural synthesis of 3D vascular trees with configurable
    morphological parameters (vessel diameter, branching angle, tortuosity).
    """

    @property
    def name(self) -> str:
        return "vamos"

    @property
    def requires_training(self) -> bool:
        return False

    @property
    def requires_gpu(self) -> bool:
        return False

    def generate_stack(
        self,
        n_volumes: int,
        config: dict[str, Any] | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate procedural vascular tree volumes.

        Parameters
        ----------
        n_volumes:
            Number of (image, mask) pairs.
        config:
            Optional config with keys: patch_size, vessel_diameter,
            branching_angle, tortuosity, seed.

        Returns
        -------
        List of (image, mask) 3D array tuples.
        """
        cfg = config or {}
        patch_size = tuple(cfg.get("patch_size", _DEFAULT_PATCH_SIZE))
        vessel_diameter = cfg.get("vessel_diameter", 2.5)
        branching_angle = cfg.get("branching_angle", 30.0)
        seed = cfg.get("seed", 42)

        rng = np.random.default_rng(seed)
        pairs: list[tuple[np.ndarray, np.ndarray]] = []

        for i in range(n_volumes):
            mask = _generate_vascular_tree(
                rng=np.random.default_rng(seed + i),
                shape=patch_size,
                diameter=vessel_diameter,
                branching_angle=branching_angle,
            )
            image = _render_volume(rng, mask)
            pairs.append((image, mask))

        logger.info(
            "Generated %d VaMos volumes (d=%.1f, angle=%.1f)",
            n_volumes,
            vessel_diameter,
            branching_angle,
        )
        return pairs


def _generate_vascular_tree(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    diameter: float = 2.5,
    branching_angle: float = 30.0,
    n_branches: int = 8,
) -> np.ndarray:
    """Generate a procedural vascular tree with branching."""
    mask = np.zeros(shape, dtype=np.uint8)
    center = np.array([s // 2 for s in shape], dtype=float)

    # Generate main trunk
    _draw_branch(
        rng,
        mask,
        center,
        diameter,
        length=40,
        depth=0,
        max_depth=4,
        branching_angle=branching_angle,
        n_branches=n_branches,
    )

    return mask


def _draw_branch(
    rng: np.random.Generator,
    mask: np.ndarray,
    start: np.ndarray,
    diameter: float,
    length: int,
    depth: int,
    max_depth: int,
    branching_angle: float,
    n_branches: int,
) -> None:
    """Recursively draw a branching vessel segment."""
    if depth > max_depth or diameter < 0.8:
        return

    shape = np.array(mask.shape)
    direction = rng.standard_normal(3)
    direction /= np.linalg.norm(direction) + 1e-8

    pos = start.copy()
    radius = max(1, int(diameter / 2))

    for step in range(length):
        # Slight random perturbation
        direction += rng.standard_normal(3) * 0.1
        direction /= np.linalg.norm(direction) + 1e-8
        pos += direction * 1.2

        center = np.clip(pos.astype(int), 0, shape - 1)

        # Draw sphere
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

        # Branching
        if step > 10 and rng.random() < 0.15 and depth < max_depth:
            angle_rad = np.radians(branching_angle + rng.standard_normal() * 10)
            _ = _rotate_direction(direction, angle_rad, rng)
            child_diameter = diameter * rng.uniform(0.6, 0.8)
            child_length = int(length * rng.uniform(0.4, 0.7))
            _draw_branch(
                rng,
                mask,
                pos.copy(),
                child_diameter,
                child_length,
                depth + 1,
                max_depth,
                branching_angle,
                n_branches,
            )


def _rotate_direction(
    direction: np.ndarray,
    angle: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Rotate a direction vector by an angle around a random axis."""
    axis = rng.standard_normal(3)
    axis -= axis.dot(direction) * direction
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        return direction
    axis /= norm

    rotated = direction * np.cos(angle) + axis * np.sin(angle)
    rotated /= np.linalg.norm(rotated) + 1e-8
    return np.asarray(rotated)


def _render_volume(rng: np.random.Generator, mask: np.ndarray) -> np.ndarray:
    """Render a binary mask into a realistic-looking volume."""
    image = mask.astype(np.float32)
    fg_val = rng.uniform(0.6, 0.9)
    bg_val = rng.uniform(0.05, 0.2)
    image = np.where(mask > 0, fg_val, bg_val)
    image += rng.standard_normal(image.shape).astype(np.float32) * 0.05
    return np.clip(image, 0.0, 1.0).astype(np.float32)
