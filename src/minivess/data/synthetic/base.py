"""Abstract base class for synthetic vascular volume generators.

Every generator method (vesselFM d_drand, MONAI VQ-VAE, VaMos, VascuSynth,
debug tubes) implements this ABC.  The registry in ``__init__.py`` maps
config-friendly names to concrete subclasses.

Usage from YAML config::

    synthetic:
      method: vesselFM_drand
      n_volumes: 10
      config:
        patch_size: [128, 128, 128]
        noise_level: 0.1
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


class SyntheticGeneratorAdapter(ABC):
    """ABC for all synthetic vascular volume generators.

    Implementors produce 3-D volumetric image + segmentation mask pairs.
    The interface is intentionally minimal so that wrapping an external
    library (vesselFM, VascuSynth, MONAI VQ-VAE, …) requires only a
    thin adapter layer.
    """

    @abstractmethod
    def generate_stack(
        self,
        n_volumes: int,
        config: dict[str, Any] | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate *n_volumes* synthetic (image, mask) pairs.

        Args:
            n_volumes: Number of 3-D volumes to generate.
            config: Optional method-specific parameters.

        Returns:
            List of ``(image, mask)`` tuples.  Both arrays are 3-D
            ``np.ndarray`` with shape ``(D, H, W)``.  Images are
            ``float32``; masks are ``uint8`` (binary or multi-label).
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the generator (e.g. ``'vesselFM_drand'``)."""

    @property
    @abstractmethod
    def requires_training(self) -> bool:
        """Whether this generator needs a training step before generation.

        Procedural generators (VascuSynth, VaMos, debug tubes) return
        ``False``.  Learned generators (VQ-VAE, diffusion) return ``True``.
        """
