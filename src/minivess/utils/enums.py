"""Centralized StrEnum discovery module.

Re-exports all project StrEnums from their original locations.
Developers can import from either the original module or from here.

Example::

    # Original (preferred for domain clarity):
    from minivess.config.models import ModelFamily

    # Discovery (find all available enums):
    from minivess.utils.enums import ModelFamily
"""

from __future__ import annotations

# Adapters
from minivess.adapters.adaptation_comparison import AdaptationMethod
from minivess.adapters.atlas import AtlasRegistrationMethod
from minivess.adapters.medsam3 import MedicalConcept, PromptType

# Compliance
from minivess.compliance.complops import RegulatoryTemplate
from minivess.compliance.eu_ai_act import EUAIActRiskLevel
from minivess.compliance.iec62304 import LifecycleStage, SoftwareSafetyClass
from minivess.compliance.regulatory_docs import SaMDRiskClass

# Config
from minivess.config.models import EnsembleStrategy, ModelFamily

# Data
from minivess.data.domain_randomization import RandomizationParam
from minivess.data.drift_synthetic import DriftType

# Ensemble
from minivess.ensemble.calibration_shift import ShiftType
from minivess.ensemble.generative_uq import GenerativeUQMethod

# Observability
from minivess.observability.model_registry import ModelStage

# Pipeline
from minivess.pipeline.federated import FLStrategy
from minivess.pipeline.segmentation_qc import QCFlag

# Serving
from minivess.serving.clinical_deploy import DeploymentTarget

# Validation
from minivess.validation.data_care import QualityDimension

__all__ = [
    # Adapters
    "AdaptationMethod",
    "AtlasRegistrationMethod",
    "MedicalConcept",
    "PromptType",
    # Compliance
    "EUAIActRiskLevel",
    "LifecycleStage",
    "RegulatoryTemplate",
    "SaMDRiskClass",
    "SoftwareSafetyClass",
    # Config
    "EnsembleStrategy",
    "ModelFamily",
    # Data
    "DriftType",
    "RandomizationParam",
    # Ensemble
    "GenerativeUQMethod",
    "ShiftType",
    # Observability
    "ModelStage",
    # Pipeline
    "FLStrategy",
    "QCFlag",
    # Serving
    "DeploymentTarget",
    # Validation
    "QualityDimension",
]
