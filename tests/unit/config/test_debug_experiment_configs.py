"""Tests for the 5 debug experiment YAML configs.

Each debug config must compose successfully via compose_experiment_config()
and satisfy structural constraints:
- debug=true
- max_epochs <= 5
- max_train_volumes == 2 (data subsetting for speed)

Plan: docs/planning/overnight-child-debug-configs.xml Phase 2
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def _compose():
    from minivess.config.compose import compose_experiment_config

    return compose_experiment_config


class TestDebugSingleModel:
    def test_composes_without_error(self, _compose) -> None:
        cfg = _compose("debug_single_model")
        assert cfg is not None

    def test_has_debug_flag(self, _compose) -> None:
        cfg = _compose("debug_single_model")
        assert cfg["debug"] is True

    def test_max_epochs_is_1(self, _compose) -> None:
        cfg = _compose("debug_single_model")
        assert cfg["max_epochs"] == 1

    def test_has_volume_subset(self, _compose) -> None:
        cfg = _compose("debug_single_model")
        assert cfg["max_train_volumes"] == 2

    def test_num_folds_is_1(self, _compose) -> None:
        cfg = _compose("debug_single_model")
        assert cfg["num_folds"] == 1


class TestDebugAllModels:
    def test_composes_without_error(self, _compose) -> None:
        cfg = _compose("debug_all_models")
        assert cfg is not None

    def test_has_debug_flag(self, _compose) -> None:
        cfg = _compose("debug_all_models")
        assert cfg["debug"] is True

    def test_models_to_test_has_multiple_entries(self, _compose) -> None:
        cfg = _compose("debug_all_models")
        assert len(cfg["models_to_test"]) >= 3

    def test_has_volume_subset(self, _compose) -> None:
        cfg = _compose("debug_all_models")
        assert cfg["max_train_volumes"] == 2


class TestDebugFullPipeline:
    def test_composes_without_error(self, _compose) -> None:
        cfg = _compose("debug_full_pipeline")
        assert cfg is not None

    def test_has_debug_flag(self, _compose) -> None:
        cfg = _compose("debug_full_pipeline")
        assert cfg["debug"] is True

    def test_has_flows_list(self, _compose) -> None:
        cfg = _compose("debug_full_pipeline")
        assert "flows" in cfg
        assert isinstance(cfg["flows"], list)
        assert len(cfg["flows"]) >= 2

    def test_has_volume_subset(self, _compose) -> None:
        cfg = _compose("debug_full_pipeline")
        assert cfg["max_train_volumes"] == 2


class TestDebugMultiLoss:
    def test_composes_without_error(self, _compose) -> None:
        cfg = _compose("debug_multi_loss")
        assert cfg is not None

    def test_has_debug_flag(self, _compose) -> None:
        cfg = _compose("debug_multi_loss")
        assert cfg["debug"] is True

    def test_losses_has_multiple_entries(self, _compose) -> None:
        cfg = _compose("debug_multi_loss")
        assert len(cfg["losses"]) >= 3

    def test_has_volume_subset(self, _compose) -> None:
        cfg = _compose("debug_multi_loss")
        assert cfg["max_train_volumes"] == 2


class TestDebugDataValidation:
    def test_composes_without_error(self, _compose) -> None:
        cfg = _compose("debug_data_validation")
        assert cfg is not None

    def test_has_debug_flag(self, _compose) -> None:
        cfg = _compose("debug_data_validation")
        assert cfg["debug"] is True

    def test_has_flows_list(self, _compose) -> None:
        cfg = _compose("debug_data_validation")
        assert "flows" in cfg


class TestAllDebugConfigs:
    """Cross-cutting constraints on all 5 debug configs."""

    DEBUG_CONFIGS = [
        "debug_single_model",
        "debug_all_models",
        "debug_full_pipeline",
        "debug_multi_loss",
        "debug_data_validation",
    ]

    @pytest.mark.parametrize("name", DEBUG_CONFIGS)
    def test_all_have_debug_flag(self, _compose, name: str) -> None:
        cfg = _compose(name)
        assert cfg["debug"] is True, f"{name} must have debug=true"

    @pytest.mark.parametrize(
        "name",
        [
            "debug_single_model",
            "debug_all_models",
            "debug_full_pipeline",
            "debug_multi_loss",
        ],
    )
    def test_training_configs_have_volume_subset(self, _compose, name: str) -> None:
        cfg = _compose(name)
        assert "max_train_volumes" in cfg, f"{name} must have max_train_volumes"
        assert cfg["max_train_volumes"] <= 4, (
            f"{name} max_train_volumes must be small for debug"
        )


class TestModelConfigs:
    """Verify that every model listed in debug_all_models.yaml has YAML configs.

    T-1.5 RED: these tests FAIL when model profile YAMLs are absent for
    models listed in debug_all_models.yaml.
    """

    def _models_to_test(self) -> list[str]:
        """Load models_to_test list from debug_all_models.yaml (yaml.safe_load, not regex)."""
        from pathlib import Path

        import yaml

        cfg_path = Path("configs/experiment/debug_all_models.yaml")
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        return cfg.get("models_to_test", [])

    def test_all_debug_models_have_model_yaml(self) -> None:
        """Every model in debug_all_models.yaml must have a configs/model/<model>.yaml."""
        from pathlib import Path

        missing = []
        for model in self._models_to_test():
            yaml_path = Path(f"configs/model/{model}.yaml")
            if not yaml_path.exists():
                missing.append(str(yaml_path))

        assert not missing, "Missing model YAML files:\n" + "\n".join(
            f"  {p}" for p in missing
        )

    def test_all_debug_models_have_profile_yaml(self) -> None:
        """Every model in debug_all_models.yaml must have a configs/model_profiles/<model>.yaml."""
        from pathlib import Path

        # dynunet maps to dynunet.yaml (already exists)
        missing = []
        for model in self._models_to_test():
            profile_path = Path(f"configs/model_profiles/{model}.yaml")
            if not profile_path.exists():
                missing.append(str(profile_path))

        assert not missing, "Missing model_profiles YAML files:\n" + "\n".join(
            f"  {p}" for p in missing
        )
