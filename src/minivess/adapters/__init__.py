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
from minivess.adapters.base import AdapterConfigInfo, ModelAdapter, SegmentationOutput
from minivess.adapters.comma import CommaAdapter
from minivess.adapters.dynunet import DynUNetAdapter
from minivess.adapters.lora import LoraModelAdapter
from minivess.adapters.medsam3 import (
    AnnotationPrompt,
    MedicalConcept,
    MedSAM3Config,
    MedSAM3Predictor,
    PromptType,
)
from minivess.adapters.model_builder import build_adapter
from minivess.adapters.sam2_backbone import Sam2Backbone
from minivess.adapters.sam2_decoder import Sam2MaskDecoder
from minivess.adapters.sam3_hybrid import Sam3HybridAdapter
from minivess.adapters.sam3_topolora import Sam3TopoLoraAdapter
from minivess.adapters.sam3_vanilla import Sam3VanillaAdapter
from minivess.adapters.segresnet import SegResNetAdapter
from minivess.adapters.swinunetr import SwinUNETRAdapter
from minivess.adapters.vesselfm import VesselFMAdapter
from minivess.adapters.vista3d import Vista3dAdapter

__all__ = [
    "AdaptationMethod",
    "AdaptationResult",
    "AdapterConfigInfo",
    "AnnotationPrompt",
    "AtlasConfig",
    "AtlasRegistrationMethod",
    "AtlasRegistrationResult",
    "CommaAdapter",
    "DynUNetAdapter",
    "FeasibilityReport",
    "LoraModelAdapter",
    "MedSAM3Config",
    "MedSAM3Predictor",
    "MedicalConcept",
    "ModelAdapter",
    "PromptType",
    "Sam2Backbone",
    "Sam2MaskDecoder",
    "Sam3HybridAdapter",
    "Sam3TopoLoraAdapter",
    "Sam3VanillaAdapter",
    "SegResNetAdapter",
    "SegmentationOutput",
    "SwinUNETRAdapter",
    "VesselFMAdapter",
    "Vista3dAdapter",
    "build_adapter",
    "compare_adaptation_methods",
    "register_atlas",
]
