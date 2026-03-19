"""Tests for dynamic method capability discovery (Phase 0+2, #332, #334).

Verifies that capability_discovery.py can dynamically discover all
implemented models, losses, metrics, plugins, ensemble strategies,
and deployment methods from the codebase. Also tests combination generators.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from minivess.testing.capability_discovery import (
    CapabilitySchema,
    build_full_combinations,
    build_practical_combinations,
    discover_all_losses,
    discover_deployment_methods,
    discover_ensemble_strategies,
    discover_implemented_models,
    discover_metrics,
    discover_post_training_plugins,
    generate_combos_yaml,
    get_valid_losses_for_model,
    load_capability_schema,
)
from minivess.testing.capability_discovery import (
    TestCombination as _TestCombination,
)


class TestDiscoverImplementedModels:
    """discover_implemented_models returns all build_adapter-dispatchable models."""

    def test_returns_nonempty_list(self) -> None:
        models = discover_implemented_models()
        # 6 paper models (non-paper models removed)
        assert len(models) >= 6

    def test_contains_dynunet(self) -> None:
        models = discover_implemented_models()
        assert "dynunet" in models

    def test_contains_sam3_variants(self) -> None:
        models = discover_implemented_models()
        for variant in ("sam3_vanilla", "sam3_topolora", "sam3_hybrid"):
            assert variant in models

    def test_excludes_not_implemented(self) -> None:
        models = discover_implemented_models()
        for excluded in ("custom",):
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
        # 6 paper models (non-paper models removed)
        assert len(schema.implemented_models) >= 6

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


# -----------------------------------------------------------------------
# Phase 2: Combination generators (#334)
# -----------------------------------------------------------------------


class TestBuildFullCombinations:
    """build_full_combinations produces all valid (model, loss) pairs."""

    def test_returns_nonempty(self) -> None:
        combos = build_full_combinations()
        assert len(combos) > 0

    def test_all_are_test_combinations(self) -> None:
        combos = build_full_combinations()
        for combo in combos:
            assert isinstance(combo, _TestCombination)

    def test_count_matches_model_times_loss(self) -> None:
        combos = build_full_combinations()
        models = discover_implemented_models()
        losses = discover_all_losses()
        # No exclusions currently, so full cross product
        assert len(combos) == len(models) * len(losses)


class TestBuildPracticalCombinations:
    """build_practical_combinations covers all models with reduced combos."""

    def test_covers_all_models(self) -> None:
        combos = build_practical_combinations()
        combo_models = {c.model for c in combos}
        models = set(discover_implemented_models())
        assert combo_models == models

    def test_fewer_than_full(self) -> None:
        practical = build_practical_combinations()
        full = build_full_combinations()
        assert len(practical) < len(full)

    def test_covers_all_loss_tiers(self) -> None:
        """Practical variant tests at least one loss from each tier."""
        from minivess.pipeline.loss_functions import (
            _HYBRID_LOSSES,
            _LIBRARY_COMPOUND_LOSSES,
            _LIBRARY_LOSSES,
        )

        combos = build_practical_combinations()
        combo_losses = {c.loss for c in combos}
        assert combo_losses & _LIBRARY_LOSSES, "Missing LIBRARY tier"
        assert combo_losses & _LIBRARY_COMPOUND_LOSSES, "Missing LIBRARY-COMPOUND tier"
        assert combo_losses & set(_HYBRID_LOSSES.keys()), "Missing HYBRID tier"
        # EXPERIMENTAL may or may not be in practical — not required

    def test_reproducible(self) -> None:
        """Same call twice produces same result."""
        combos1 = build_practical_combinations()
        combos2 = build_practical_combinations()
        assert combos1 == combos2


class TestGenerateCombosYaml:
    """generate_combos_yaml writes valid YAML from combinations."""

    def test_writes_yaml_file(self, tmp_path: Path) -> None:
        combos = build_practical_combinations()
        output = tmp_path / "combos.yaml"
        result = generate_combos_yaml(combos, output)
        assert result.exists()
        assert result.suffix == ".yaml"

    def test_yaml_is_loadable(self, tmp_path: Path) -> None:
        import yaml

        combos = build_practical_combinations()
        output = tmp_path / "combos.yaml"
        generate_combos_yaml(combos, output)
        with output.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        assert "combinations" in data
        assert len(data["combinations"]) == len(combos)

    def test_yaml_is_reproducible(self, tmp_path: Path) -> None:
        combos = build_practical_combinations()
        out1 = tmp_path / "combos1.yaml"
        out2 = tmp_path / "combos2.yaml"
        generate_combos_yaml(combos, out1)
        generate_combos_yaml(combos, out2)
        assert out1.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")
