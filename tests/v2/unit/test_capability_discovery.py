"""Tests for dynamic method capability discovery (Phase 0, #332).

Verifies that capability_discovery.py can dynamically discover all
implemented models, losses, metrics, plugins, ensemble strategies,
and deployment methods from the codebase.
"""

from __future__ import annotations

import pytest

from minivess.testing.capability_discovery import (
    CapabilitySchema,
    discover_all_losses,
    discover_deployment_methods,
    discover_ensemble_strategies,
    discover_implemented_models,
    discover_metrics,
    discover_post_training_plugins,
    get_valid_losses_for_model,
    load_capability_schema,
)


class TestDiscoverImplementedModels:
    """discover_implemented_models returns all build_adapter-dispatchable models."""

    def test_returns_nonempty_list(self) -> None:
        models = discover_implemented_models()
        assert len(models) >= 10

    def test_contains_dynunet(self) -> None:
        models = discover_implemented_models()
        assert "dynunet" in models

    def test_contains_sam3_variants(self) -> None:
        models = discover_implemented_models()
        for variant in ("sam3_vanilla", "sam3_topolora", "sam3_hybrid"):
            assert variant in models

    def test_excludes_not_implemented(self) -> None:
        models = discover_implemented_models()
        for excluded in ("sam3_lora", "multitask_dynunet", "custom"):
            assert excluded not in models


class TestDiscoverAllLosses:
    """discover_all_losses returns all loss names from build_loss_function."""

    def test_returns_nonempty(self) -> None:
        losses = discover_all_losses()
        assert len(losses) >= 18

    def test_contains_default_loss(self) -> None:
        losses = discover_all_losses()
        assert "cbdice_cldice" in losses

    def test_contains_all_tiers(self) -> None:
        losses = discover_all_losses()
        # At least one from each tier
        assert "dice_ce" in losses  # LIBRARY
        assert "dice_ce_cldice" in losses  # LIBRARY-COMPOUND
        assert "skeleton_recall" in losses  # HYBRID
        assert "betti" in losses  # EXPERIMENTAL


class TestDiscoverMetrics:
    """discover_metrics returns all metric names from YAML registry."""

    def test_returns_nonempty(self) -> None:
        metrics = discover_metrics()
        assert len(metrics) >= 15

    def test_contains_core_metrics(self) -> None:
        metrics = discover_metrics()
        for core in ("dsc", "centreline_dsc", "val_loss", "val_dice"):
            assert core in metrics


class TestDiscoverPostTrainingPlugins:
    """discover_post_training_plugins returns all registered plugin names."""

    def test_returns_six_plugins(self) -> None:
        plugins = discover_post_training_plugins()
        assert len(plugins) == 6

    def test_contains_expected_plugins(self) -> None:
        plugins = discover_post_training_plugins()
        expected = {
            "swa",
            "multi_swa",
            "model_merging",
            "calibration",
            "crc_conformal",
            "conseco_fp_control",
        }
        assert expected == set(plugins)


class TestDiscoverEnsembleStrategies:
    """discover_ensemble_strategies returns valid strategies minus excluded."""

    def test_excludes_swag(self) -> None:
        strategies = discover_ensemble_strategies()
        assert "swag" not in strategies

    def test_excludes_learned_stacking(self) -> None:
        strategies = discover_ensemble_strategies()
        assert "learned_stacking" not in strategies

    def test_includes_mean(self) -> None:
        strategies = discover_ensemble_strategies()
        assert "mean" in strategies


class TestDiscoverDeploymentMethods:
    """discover_deployment_methods returns enabled deploy methods."""

    def test_includes_onnx(self) -> None:
        methods = discover_deployment_methods()
        assert "onnx" in methods

    def test_includes_bentoml(self) -> None:
        methods = discover_deployment_methods()
        assert "bentoml" in methods


class TestLoadCapabilitySchema:
    """load_capability_schema parses method_capabilities.yaml."""

    def test_loads_default_schema(self) -> None:
        schema = load_capability_schema()
        assert isinstance(schema, CapabilitySchema)

    def test_schema_has_version(self) -> None:
        schema = load_capability_schema()
        assert schema.version == "1.0"

    def test_schema_has_implemented_models(self) -> None:
        schema = load_capability_schema()
        assert len(schema.implemented_models) >= 10

    def test_schema_not_implemented_separate(self) -> None:
        schema = load_capability_schema()
        # implemented and not_implemented should be disjoint
        assert not set(schema.implemented_models) & set(schema.not_implemented)


class TestGetValidLossesForModel:
    """get_valid_losses_for_model returns all losses minus exclusions."""

    def test_dynunet_gets_all_losses(self) -> None:
        # No exclusions currently, so dynunet gets everything
        losses = get_valid_losses_for_model("dynunet")
        all_losses = discover_all_losses()
        assert set(losses) == set(all_losses)

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(ValueError, match="not in implemented_models"):
            get_valid_losses_for_model("nonexistent_model")


class TestCapabilityConsistency:
    """Cross-check discovery functions against schema for consistency."""

    def test_all_model_families_accounted_for(self) -> None:
        """Every ModelFamily enum is either implemented or not_implemented."""
        from minivess.config.models import ModelFamily

        schema = load_capability_schema()
        all_accounted = set(schema.implemented_models) | set(schema.not_implemented)
        for member in ModelFamily:
            assert member.value in all_accounted, (
                f"ModelFamily.{member.name} ({member.value}) not in "
                f"implemented_models or not_implemented"
            )

    def test_discovery_deterministic(self) -> None:
        """Two calls produce identical results."""
        models1 = discover_implemented_models()
        models2 = discover_implemented_models()
        assert models1 == models2

        losses1 = discover_all_losses()
        losses2 = discover_all_losses()
        assert losses1 == losses2

    def test_default_losses_are_valid(self) -> None:
        """model_default_loss references implemented losses."""
        schema = load_capability_schema()
        all_losses = discover_all_losses()
        for model, loss in schema.model_default_loss.items():
            assert loss in all_losses, f"Default loss {loss} for {model} not found"
