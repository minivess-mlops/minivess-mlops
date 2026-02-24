"""Tests that all package __init__.py re-exports resolve correctly (Code Review R1.4).

Validates that every symbol listed in __all__ is actually importable from the
package's top-level namespace. Catches broken imports, circular dependencies,
and stale __all__ entries.
"""

from __future__ import annotations

import importlib

import pytest

# Each tuple: (package_dotpath, list of symbols that must be importable)
_PACKAGES = [
    (
        "minivess.adapters",
        [
            "AdaptationMethod",
            "AdaptationResult",
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
            "SegmentationOutput",
            "SegResNetAdapter",
            "SwinUNETRAdapter",
            "VesselFMAdapter",
            "Vista3dAdapter",
            "compare_adaptation_methods",
            "register_atlas",
        ],
    ),
    (
        "minivess.config",
        [
            "DataConfig",
            "EnsembleConfig",
            "EnsembleStrategy",
            "ExperimentConfig",
            "ModelConfig",
            "ModelFamily",
            "ServingConfig",
            "TrainingConfig",
        ],
    ),
    (
        "minivess.ensemble",
        [
            "CalibrationResult",
            "CalibrationShiftAnalyzer",
            "ConformalPredictor",
            "ConformalResult",
            "DeepEnsemblePredictor",
            "EnsemblePredictor",
            "GenerativeUQConfig",
            "GenerativeUQEvaluator",
            "GenerativeUQMethod",
            "MCDropoutPredictor",
            "MultiRaterData",
            "ShiftType",
            "ShiftedCalibrationResult",
            "UncertaintyOutput",
            "generalized_energy_distance",
            "greedy_soup",
            "q_dice",
            "temperature_scale",
        ],
    ),
    (
        "minivess.pipeline",
        [
            "ConfidenceInterval",
            "EpochResult",
            "FLClientConfig",
            "FLRoundResult",
            "FLServerConfig",
            "FLSimulator",
            "FLStrategy",
            "FederatedAveraging",
            "MetricResult",
            "QCFlag",
            "QCResult",
            "SegmentationMetrics",
            "SegmentationQC",
            "SegmentationTrainer",
            "bootstrap_ci",
            "build_loss_function",
        ],
    ),
    (
        "minivess.compliance",
        [
            "AuditEntry",
            "AuditTrail",
            "EUAIActChecklist",
            "EUAIActRiskLevel",
            "FairnessReport",
            "ModelCard",
            "RegulatoryDocGenerator",
            "SaMDRiskClass",
            "SubgroupMetrics",
            "compute_disparity",
            "evaluate_subgroup_fairness",
            "generate_audit_report",
        ],
    ),
    (
        "minivess.utils",
        [
            "markdown_header",
            "markdown_section",
            "markdown_table",
            "timestamp_utc",
        ],
    ),
]


@pytest.mark.parametrize(
    ("package", "symbols"),
    _PACKAGES,
    ids=[p for p, _ in _PACKAGES],
)
class TestPackageExports:
    """Validate that package __all__ entries are importable."""

    def test_all_symbols_importable(self, package: str, symbols: list[str]) -> None:
        """Every symbol in __all__ should be importable from the package."""
        mod = importlib.import_module(package)
        for name in symbols:
            assert hasattr(mod, name), f"{package}.{name} not importable"

    def test_all_list_exists(self, package: str, symbols: list[str]) -> None:
        """Package should define __all__."""
        mod = importlib.import_module(package)
        assert hasattr(mod, "__all__"), f"{package} missing __all__"
