"""Tests for model registry with promotion stages (Issue #50)."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# T1: ModelStage enum
# ---------------------------------------------------------------------------


class TestModelStage:
    """Test model lifecycle stage enum."""

    def test_enum_values(self) -> None:
        """ModelStage should have four lifecycle stages."""
        from minivess.observability.model_registry import ModelStage

        assert ModelStage.DEVELOPMENT == "development"
        assert ModelStage.STAGING == "staging"
        assert ModelStage.PRODUCTION == "production"
        assert ModelStage.ARCHIVED == "archived"

    def test_all_stages(self) -> None:
        """All four stages should be present."""
        from minivess.observability.model_registry import ModelStage

        assert len(ModelStage) == 4


# ---------------------------------------------------------------------------
# T2: ModelVersion
# ---------------------------------------------------------------------------


class TestModelVersion:
    """Test model version dataclass."""

    def test_construction(self) -> None:
        """ModelVersion should capture model metadata."""
        from minivess.observability.model_registry import ModelStage, ModelVersion

        mv = ModelVersion(
            model_name="dynunet-full",
            version="2.1.0",
            stage=ModelStage.DEVELOPMENT,
            metrics={"dice": 0.85, "hd95": 3.2},
        )
        assert mv.model_name == "dynunet-full"
        assert mv.version == "2.1.0"
        assert mv.stage == ModelStage.DEVELOPMENT

    def test_semantic_version_parts(self) -> None:
        """Semantic version should parse major.minor.patch."""
        from minivess.observability.model_registry import ModelStage, ModelVersion

        mv = ModelVersion(
            model_name="test",
            version="3.2.1",
            stage=ModelStage.DEVELOPMENT,
            metrics={},
        )
        assert mv.major == 3
        assert mv.minor == 2
        assert mv.patch == 1


# ---------------------------------------------------------------------------
# T3: PromotionCriteria
# ---------------------------------------------------------------------------


class TestPromotionCriteria:
    """Test promotion criteria threshold checking."""

    def test_metrics_meet_criteria(self) -> None:
        """check should pass when all metrics meet thresholds."""
        from minivess.observability.model_registry import PromotionCriteria

        criteria = PromotionCriteria(
            min_thresholds={"dice": 0.80},
            max_thresholds={"hd95": 5.0},
        )
        result = criteria.check({"dice": 0.85, "hd95": 3.2})
        assert result.approved is True

    def test_metrics_fail_min(self) -> None:
        """check should reject when min threshold not met."""
        from minivess.observability.model_registry import PromotionCriteria

        criteria = PromotionCriteria(min_thresholds={"dice": 0.80})
        result = criteria.check({"dice": 0.70})
        assert result.approved is False
        assert "dice" in result.reason

    def test_metrics_fail_max(self) -> None:
        """check should reject when max threshold exceeded."""
        from minivess.observability.model_registry import PromotionCriteria

        criteria = PromotionCriteria(max_thresholds={"hd95": 5.0})
        result = criteria.check({"hd95": 7.0})
        assert result.approved is False
        assert "hd95" in result.reason


# ---------------------------------------------------------------------------
# T4: ModelRegistry
# ---------------------------------------------------------------------------


class TestModelRegistry:
    """Test model registry with promotion workflow."""

    def test_register_version(self) -> None:
        """register_version should add a model in DEVELOPMENT stage."""
        from minivess.observability.model_registry import ModelRegistry

        registry = ModelRegistry()
        mv = registry.register_version(
            model_name="dynunet-full",
            version="1.0.0",
            metrics={"dice": 0.85},
        )
        assert mv.model_name == "dynunet-full"
        assert mv.stage.value == "development"

    def test_promote_to_staging(self) -> None:
        """promote should transition from development to staging."""
        from minivess.observability.model_registry import (
            ModelRegistry,
            ModelStage,
            PromotionCriteria,
        )

        registry = ModelRegistry()
        registry.register_version("dynunet-full", "1.0.0", {"dice": 0.85})
        criteria = PromotionCriteria(min_thresholds={"dice": 0.80})

        result = registry.promote(
            "dynunet-full",
            "1.0.0",
            target_stage=ModelStage.STAGING,
            criteria=criteria,
        )
        assert result.approved is True
        mv = registry.get_version("dynunet-full", "1.0.0")
        assert mv.stage == ModelStage.STAGING

    def test_promote_rejected(self) -> None:
        """promote should reject if criteria not met."""
        from minivess.observability.model_registry import (
            ModelRegistry,
            ModelStage,
            PromotionCriteria,
        )

        registry = ModelRegistry()
        registry.register_version("dynunet-full", "1.0.0", {"dice": 0.70})
        criteria = PromotionCriteria(min_thresholds={"dice": 0.80})

        result = registry.promote(
            "dynunet-full",
            "1.0.0",
            target_stage=ModelStage.STAGING,
            criteria=criteria,
        )
        assert result.approved is False
        mv = registry.get_version("dynunet-full", "1.0.0")
        assert mv.stage.value == "development"

    def test_get_production_model(self) -> None:
        """get_production_model should return the current production version."""
        from minivess.observability.model_registry import (
            ModelRegistry,
            ModelStage,
            PromotionCriteria,
        )

        registry = ModelRegistry()
        registry.register_version("dynunet-full", "1.0.0", {"dice": 0.85})
        criteria = PromotionCriteria(min_thresholds={"dice": 0.80})
        registry.promote("dynunet-full", "1.0.0", ModelStage.STAGING, criteria)
        registry.promote("dynunet-full", "1.0.0", ModelStage.PRODUCTION, criteria)

        prod = registry.get_production_model("dynunet-full")
        assert prod is not None
        assert prod.version == "1.0.0"
        assert prod.stage == ModelStage.PRODUCTION

    def test_no_production_model(self) -> None:
        """get_production_model should return None if no production version."""
        from minivess.observability.model_registry import ModelRegistry

        registry = ModelRegistry()
        registry.register_version("dynunet-full", "1.0.0", {"dice": 0.85})
        assert registry.get_production_model("dynunet-full") is None

    def test_to_markdown(self) -> None:
        """to_markdown should produce a registry report."""
        from minivess.observability.model_registry import ModelRegistry

        registry = ModelRegistry()
        registry.register_version("dynunet-full", "1.0.0", {"dice": 0.85})
        registry.register_version("dynunet-half", "1.0.0", {"dice": 0.80})
        md = registry.to_markdown()
        assert "Model Registry" in md
        assert "dynunet-full" in md
        assert "dynunet-half" in md
