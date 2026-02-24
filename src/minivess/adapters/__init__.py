"""Model adapters -- ModelAdapter protocol and concrete implementations."""

from __future__ import annotations

from minivess.adapters.adaptation_comparison import (
    AdaptationMethod,
    AdaptationResult,
    FeasibilityReport,
    compare_adaptation_methods,
)
from minivess.adapters.atlas import (
    AtlasConfig,
    AtlasRegistrationMethod,
    AtlasRegistrationResult,
    register_atlas,
)
from minivess.adapters.base import ModelAdapter, SegmentationOutput
from minivess.adapters.comma import CommaAdapter
from minivess.adapters.dynunet import DynUNetAdapter
from minivess.adapters.lora import LoraModelAdapter
from minivess.adapters.segresnet import SegResNetAdapter
from minivess.adapters.swinunetr import SwinUNETRAdapter
from minivess.adapters.vesselfm import VesselFMAdapter
from minivess.adapters.vista3d import Vista3dAdapter

__all__ = [
    "AdaptationMethod",
    "AdaptationResult",
    "AtlasConfig",
    "AtlasRegistrationMethod",
    "AtlasRegistrationResult",
    "CommaAdapter",
    "DynUNetAdapter",
    "FeasibilityReport",
    "LoraModelAdapter",
    "ModelAdapter",
    "SegmentationOutput",
    "SegResNetAdapter",
    "SwinUNETRAdapter",
    "VesselFMAdapter",
    "Vista3dAdapter",
    "compare_adaptation_methods",
    "register_atlas",
]
