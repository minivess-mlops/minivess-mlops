"""Tests for AtlasSegFM one-shot foundation model customization (Issue #15)."""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# T1: AtlasRegistrationMethod enum
# ---------------------------------------------------------------------------


class TestAtlasRegistrationMethod:
    """Test atlas registration method enum."""

    def test_enum_values(self) -> None:
        """AtlasRegistrationMethod should have three methods."""
        from minivess.adapters.atlas import AtlasRegistrationMethod

        assert AtlasRegistrationMethod.AFFINE == "affine"
        assert AtlasRegistrationMethod.DEFORMABLE == "deformable"
        assert AtlasRegistrationMethod.LANDMARK == "landmark"


# ---------------------------------------------------------------------------
# T2: AtlasConfig
# ---------------------------------------------------------------------------


class TestAtlasConfig:
    """Test atlas configuration."""

    def test_construction(self) -> None:
        """AtlasConfig should capture atlas settings."""
        from minivess.adapters.atlas import AtlasConfig

        config = AtlasConfig(
            atlas_name="vessel_atlas_v1",
            registration_method="affine",
            spatial_dims=3,
        )
        assert config.atlas_name == "vessel_atlas_v1"
        assert config.registration_method == "affine"
        assert config.spatial_dims == 3

    def test_defaults(self) -> None:
        """AtlasConfig should have sensible defaults."""
        from minivess.adapters.atlas import AtlasConfig

        config = AtlasConfig(atlas_name="test_atlas")
        assert config.registration_method == "affine"
        assert config.spatial_dims == 3
        assert config.num_atlas_channels == 1


# ---------------------------------------------------------------------------
# T3: AtlasRegistrationResult
# ---------------------------------------------------------------------------


class TestAtlasRegistrationResult:
    """Test atlas registration result."""

    def test_construction(self) -> None:
        """AtlasRegistrationResult should capture registration outputs."""
        from minivess.adapters.atlas import AtlasRegistrationResult

        warped = np.zeros((16, 16, 16), dtype=np.float32)
        result = AtlasRegistrationResult(
            warped_atlas=warped,
            similarity_score=0.85,
            method="affine",
        )
        assert result.similarity_score == 0.85
        assert result.method == "affine"
        assert result.warped_atlas.shape == (16, 16, 16)

    def test_deformation_field_optional(self) -> None:
        """Deformation field should be optional (None for affine)."""
        from minivess.adapters.atlas import AtlasRegistrationResult

        warped = np.zeros((16, 16, 16), dtype=np.float32)
        result = AtlasRegistrationResult(
            warped_atlas=warped,
            similarity_score=0.9,
            method="affine",
        )
        assert result.deformation_field is None


# ---------------------------------------------------------------------------
# T4: register_atlas function
# ---------------------------------------------------------------------------


class TestRegisterAtlas:
    """Test atlas registration function."""

    def test_affine_registration(self) -> None:
        """Affine registration should return a warped atlas."""
        from minivess.adapters.atlas import register_atlas

        atlas = np.random.default_rng(42).normal(0, 1, (16, 16, 16)).astype(
            np.float32
        )
        target = np.random.default_rng(99).normal(0, 1, (16, 16, 16)).astype(
            np.float32
        )
        result = register_atlas(atlas, target, method="affine")
        assert result.warped_atlas.shape == target.shape
        assert result.method == "affine"
        assert 0.0 <= result.similarity_score <= 1.0

    def test_identity_registration(self) -> None:
        """Registering identical volumes should yield high similarity."""
        from minivess.adapters.atlas import register_atlas

        vol = np.ones((8, 8, 8), dtype=np.float32) * 0.5
        result = register_atlas(vol, vol, method="affine")
        assert result.similarity_score > 0.9

    def test_deformable_registration(self) -> None:
        """Deformable registration should include a deformation field."""
        from minivess.adapters.atlas import register_atlas

        atlas = np.random.default_rng(42).normal(0, 1, (8, 8, 8)).astype(np.float32)
        target = np.random.default_rng(99).normal(0, 1, (8, 8, 8)).astype(np.float32)
        result = register_atlas(atlas, target, method="deformable")
        assert result.deformation_field is not None
        assert result.method == "deformable"


# ---------------------------------------------------------------------------
# T5: AdaptationMethod and comparison
# ---------------------------------------------------------------------------


class TestAdaptationComparison:
    """Test adaptation method comparison infrastructure."""

    def test_adaptation_method_enum(self) -> None:
        """AdaptationMethod should have four methods."""
        from minivess.adapters.adaptation_comparison import AdaptationMethod

        assert AdaptationMethod.FULL_FINETUNE == "full_finetune"
        assert AdaptationMethod.LORA == "lora"
        assert AdaptationMethod.ATLAS_ONESHOT == "atlas_oneshot"
        assert AdaptationMethod.ZERO_SHOT == "zero_shot"

    def test_adaptation_result(self) -> None:
        """AdaptationResult should capture method performance."""
        from minivess.adapters.adaptation_comparison import AdaptationResult

        result = AdaptationResult(
            method="lora",
            dice_score=0.82,
            trainable_params=8000,
            total_params=4_700_000,
        )
        assert result.method == "lora"
        assert result.dice_score == 0.82
        assert result.parameter_efficiency < 0.01

    def test_compare_methods(self) -> None:
        """compare_adaptation_methods should produce a ranked list."""
        from minivess.adapters.adaptation_comparison import (
            AdaptationResult,
            compare_adaptation_methods,
        )

        results = [
            AdaptationResult(
                method="full_finetune",
                dice_score=0.88,
                trainable_params=4_700_000,
                total_params=4_700_000,
            ),
            AdaptationResult(
                method="lora",
                dice_score=0.85,
                trainable_params=8000,
                total_params=4_700_000,
            ),
            AdaptationResult(
                method="atlas_oneshot",
                dice_score=0.79,
                trainable_params=0,
                total_params=4_700_000,
            ),
        ]
        table = compare_adaptation_methods(results)
        assert len(table) == 3
        # Should be sorted by dice descending
        assert table[0].dice_score >= table[1].dice_score

    def test_feasibility_report(self) -> None:
        """FeasibilityReport should generate a markdown analysis."""
        from minivess.adapters.adaptation_comparison import (
            AdaptationResult,
            FeasibilityReport,
        )

        results = [
            AdaptationResult(
                method="lora",
                dice_score=0.85,
                trainable_params=8000,
                total_params=4_700_000,
            ),
            AdaptationResult(
                method="atlas_oneshot",
                dice_score=0.79,
                trainable_params=0,
                total_params=4_700_000,
            ),
        ]
        report = FeasibilityReport(
            model_name="SegResNet",
            target_anatomy="cerebral_vessels",
            results=results,
        )
        md = report.to_markdown()
        assert "Feasibility" in md
        assert "SegResNet" in md
        assert "lora" in md
        assert "atlas_oneshot" in md
