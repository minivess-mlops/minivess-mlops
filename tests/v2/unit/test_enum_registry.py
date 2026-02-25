"""Tests for centralized enum discovery module (Code Review R4.2).

Validates that all StrEnums are discoverable from utils/enums.py and
that DriftType uses StrEnum consistently.
"""

from __future__ import annotations

from enum import StrEnum

# ---------------------------------------------------------------------------
# T1: All enums re-exported from utils/enums
# ---------------------------------------------------------------------------


class TestEnumRegistry:
    """Test that utils/enums.py re-exports all project StrEnums."""

    def test_all_enums_importable_from_registry(self) -> None:
        """Every project StrEnum should be importable from utils.enums."""
        from minivess.utils.enums import (
            AdaptationMethod,
            AtlasRegistrationMethod,
            DeploymentTarget,
            DriftType,
            EnsembleStrategy,
            EUAIActRiskLevel,
            FLStrategy,
            GenerativeUQMethod,
            LifecycleStage,
            MedicalConcept,
            ModelFamily,
            ModelStage,
            PromptType,
            QCFlag,
            QualityDimension,
            RandomizationParam,
            RegulatoryTemplate,
            SaMDRiskClass,
            ShiftType,
            SoftwareSafetyClass,
        )

        # All should be StrEnum subclasses
        all_enums = [
            AdaptationMethod, AtlasRegistrationMethod, DeploymentTarget,
            DriftType, EnsembleStrategy, EUAIActRiskLevel, FLStrategy,
            GenerativeUQMethod, LifecycleStage, MedicalConcept,
            ModelFamily, ModelStage, PromptType, QCFlag,
            QualityDimension, RandomizationParam, RegulatoryTemplate,
            SaMDRiskClass, ShiftType, SoftwareSafetyClass,
        ]
        for enum_cls in all_enums:
            assert issubclass(enum_cls, StrEnum), f"{enum_cls.__name__} is not StrEnum"

    def test_enum_count(self) -> None:
        """Registry should contain all 20 project enums."""
        from minivess.utils import enums

        enum_classes = [
            v for v in vars(enums).values()
            if isinstance(v, type) and issubclass(v, StrEnum) and v is not StrEnum
        ]
        assert len(enum_classes) >= 20


# ---------------------------------------------------------------------------
# T2: DriftType uses StrEnum (not str, Enum)
# ---------------------------------------------------------------------------


class TestDriftTypeConsistency:
    """Test that DriftType uses StrEnum like all other enums."""

    def test_drift_type_is_strenum(self) -> None:
        """DriftType should be a StrEnum subclass."""
        from minivess.data.drift_synthetic import DriftType

        assert issubclass(DriftType, StrEnum)

    def test_drift_type_values_preserved(self) -> None:
        """DriftType values should be unchanged after migration."""
        from minivess.data.drift_synthetic import DriftType

        assert DriftType.INTENSITY_SHIFT == "intensity_shift"
        assert DriftType.NOISE_INJECTION == "noise_injection"
        assert DriftType.RESOLUTION_DEGRADATION == "resolution_degradation"
        assert DriftType.TOPOLOGY_CORRUPTION == "topology_corruption"
