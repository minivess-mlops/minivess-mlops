"""Data -- Data loading, preprocessing, and DVC integration."""

from __future__ import annotations

from minivess.data.annotation_session import AnnotationSession
from minivess.data.domain_randomization import (
    DomainRandomizationConfig,
    DomainRandomizationPipeline,
    RandomizationParam,
    SyntheticVesselGenerator,
)
from minivess.data.loader import (
    create_train_loader,
    create_val_loader,
    discover_nifti_pairs,
)
from minivess.data.transforms import build_train_transforms, build_val_transforms

__all__ = [
    "AnnotationSession",
    "DomainRandomizationConfig",
    "DomainRandomizationPipeline",
    "RandomizationParam",
    "SyntheticVesselGenerator",
    "build_intensity_augmentation",
    "build_train_transforms",
    "build_val_transforms",
    "create_train_loader",
    "create_val_loader",
    "discover_nifti_pairs",
]


def build_intensity_augmentation():  # type: ignore[no-untyped-def]
    """Lazy import to avoid TorchIO segfault during pytest collection."""
    from minivess.data.augmentation import (
        build_intensity_augmentation as _build,
    )

    return _build()
