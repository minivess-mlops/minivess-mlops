"""Synthetic vascular volume generation — multi-method adapter registry.

Provides a config-driven interface for generating synthetic 3D vascular
volumes with ground truth segmentation masks. Multiple generator backends
are supported, selected via YAML config or the `generate_stack()` API.

Usage:
    from minivess.data.synthetic import generate_stack

    volumes = generate_stack(method="debug", n_volumes=5)
    for image, mask in volumes:
        print(image.shape, mask.shape)  # (D, H, W) 3D arrays
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from minivess.data.synthetic.base import SyntheticGeneratorAdapter

if TYPE_CHECKING:
    import numpy as np
from minivess.data.synthetic.debug_generator import DebugTubeGenerator

# ---------------------------------------------------------------------------
# Registry — maps method names to SyntheticGeneratorAdapter *classes*
# Adapters register themselves here or via register_generator().
# ---------------------------------------------------------------------------
SYNTHETIC_GENERATORS: dict[str, type[SyntheticGeneratorAdapter]] = {
    "debug": DebugTubeGenerator,
}


def register_generator(
    method_name: str, generator_cls: type[SyntheticGeneratorAdapter]
) -> None:
    """Register a new synthetic generator adapter.

    Args:
        method_name: Config-friendly name (e.g. 'vesselFM_drand').
        generator_cls: A *class* (not instance) that subclasses
            ``SyntheticGeneratorAdapter``.
    """
    if not issubclass(generator_cls, SyntheticGeneratorAdapter):
        msg = f"{generator_cls.__name__} must subclass SyntheticGeneratorAdapter"
        raise TypeError(msg)
    SYNTHETIC_GENERATORS[method_name] = generator_cls


def get_generator(method: str) -> SyntheticGeneratorAdapter:
    """Instantiate and return a generator by its registry name.

    Raises:
        KeyError: If *method* is not registered.
    """
    if method not in SYNTHETIC_GENERATORS:
        available = ", ".join(sorted(SYNTHETIC_GENERATORS))
        msg = f"Unknown synthetic generator method '{method}'. Available: {available}"
        raise KeyError(msg)
    return SYNTHETIC_GENERATORS[method]()


def list_generators() -> list[str]:
    """Return sorted list of registered generator method names."""
    return sorted(SYNTHETIC_GENERATORS.keys())


def generate_stack(
    method: str,
    n_volumes: int,
    config: dict[str, Any] | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Top-level API: generate synthetic (image, mask) pairs.

    Args:
        method: Registry name of the generator (e.g. ``'debug'``).
        n_volumes: Number of (image, mask) pairs to generate.
        config: Optional per-method configuration dict.

    Returns:
        List of ``(image, mask)`` tuples where both are 3-D ``ndarray``.
    """
    generator = get_generator(method)
    return generator.generate_stack(n_volumes=n_volumes, config=config)


__all__ = [
    "SYNTHETIC_GENERATORS",
    "SyntheticGeneratorAdapter",
    "generate_stack",
    "get_generator",
    "list_generators",
    "register_generator",
]
