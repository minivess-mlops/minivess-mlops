"""Generic multi-task target loading framework for MONAI pipelines.

Supports ANY auxiliary ground truth that can be loaded as a NIfTI file
or computed on-the-fly from the primary label. Task-agnostic — researchers
define auxiliary targets via config, not code changes.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable  # noqa: TC003 — used at runtime in dataclass
from typing import TYPE_CHECKING, Any

import nibabel as nib
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class AuxTargetConfig:
    """Configuration for a single auxiliary target.

    Args:
        name: Key name in the MONAI data dict (e.g., "sdf", "centerline_dist").
        suffix: File suffix for precomputed NIfTI (e.g., "sdf" → {vol_id}_sdf.nii.gz).
        compute_fn: Function(mask: ndarray) -> ndarray for on-the-fly computation.
    """

    name: str
    suffix: str
    compute_fn: Callable[[np.ndarray], np.ndarray]


class LoadAuxiliaryTargetsd:
    """MONAI-compatible MapTransform for loading auxiliary GT targets.

    For each configured auxiliary target:
    1. Try to load precomputed NIfTI from precomputed_dir/{volume_id}_{suffix}.nii.gz
    2. Fall back to on-the-fly computation via compute_fn(label)

    This is completely task-agnostic — it does not know about SDF, centerline,
    or any specific task. It just loads/computes whatever configs specify.

    Args:
        label_key: Key for primary label in data dict.
        aux_configs: List of AuxTargetConfig defining auxiliary targets.
        precomputed_dir: Directory containing precomputed NIfTI files.
            If None, always falls back to on-the-fly computation.
    """

    def __init__(
        self,
        label_key: str = "label",
        aux_configs: list[AuxTargetConfig] | None = None,
        precomputed_dir: Path | None = None,
    ) -> None:
        self.label_key = label_key
        self.aux_configs = aux_configs or []
        self.precomputed_dir = precomputed_dir

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """Add auxiliary target keys to the data dict."""
        label = data[self.label_key]
        volume_id = data.get("volume_id")
        result = dict(data)

        for config in self.aux_configs:
            target = self._load_or_compute(label, volume_id, config)
            result[config.name] = target

        return result

    def _load_or_compute(
        self,
        label: np.ndarray,
        volume_id: str | None,
        config: AuxTargetConfig,
    ) -> np.ndarray:
        """Load precomputed target or compute on-the-fly."""
        # Try precomputed file first
        if self.precomputed_dir is not None and volume_id is not None:
            nifti_path = self.precomputed_dir / f"{volume_id}_{config.suffix}.nii.gz"
            if nifti_path.exists():
                logger.debug("Loading precomputed %s from %s", config.name, nifti_path)
                img = nib.load(str(nifti_path))  # type: ignore[attr-defined]
                return np.asarray(img.dataobj, dtype=np.float32)  # type: ignore[attr-defined]

        # Fall back to on-the-fly computation
        logger.debug(
            "Computing %s on-the-fly via %s", config.name, config.compute_fn.__name__
        )
        target = config.compute_fn(label)
        return target.astype(np.float32)
