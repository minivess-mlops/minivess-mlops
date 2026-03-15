"""Tests for Hydra config group composition system.

Covers:
- Phase 1: Config group directory structure + YAML files (#287)
- Phase 2: Compose bridge (#288)
- Phase 7: Experiment configs via Hydra composition (#293)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


# ---------------------------------------------------------------------------
# Phase 1: Config group directory structure
# ---------------------------------------------------------------------------


class TestConfigGroupStructure:
    """Verify that all Hydra config group files exist and are valid YAML."""

    def test_base_yaml_exists(self) -> None:
        assert (CONFIGS_DIR / "base.yaml").exists(), "configs/base.yaml missing"

    def test_base_yaml_has_defaults_list(self) -> None:
        cfg = _load_yaml(CONFIGS_DIR / "base.yaml")
        assert "defaults" in cfg, "base.yaml must have 'defaults' key"
        assert isinstance(cfg["defaults"], list), "'defaults' must be a list"

    def test_base_yaml_defaults_reference_groups(self) -> None:
        """Defaults list should reference data, training, checkpoint groups."""
        cfg = _load_yaml(CONFIGS_DIR / "base.yaml")
        defaults = cfg["defaults"]
        # Flatten: each element is either a string or a dict
        keys: set[str] = set()
        for item in defaults:
            if isinstance(item, dict):
                keys.update(item.keys())
            elif isinstance(item, str):
                keys.add(item)
        assert "data" in keys, "defaults should reference 'data' group"
        assert "training" in keys, "defaults should reference 'training' group"
        assert "checkpoint" in keys, "defaults should reference 'checkpoint' group"

    def test_data_minivess_yaml_exists(self) -> None:
        assert (CONFIGS_DIR / "data" / "minivess.yaml").exists()

    def test_training_default_yaml_exists(self) -> None:
        assert (CONFIGS_DIR / "training" / "default.yaml").exists()

    def test_checkpoint_standard_yaml_exists(self) -> None:
        assert (CONFIGS_DIR / "checkpoint" / "standard.yaml").exists()

    def test_checkpoint_lightweight_yaml_exists(self) -> None:
        assert (CONFIGS_DIR / "checkpoint" / "lightweight.yaml").exists()

    @pytest.mark.parametrize(
        "model_name",
        [
            "dynunet",
            "sam3_vanilla",
            "sam3_topolora",
            "sam3_hybrid",
            "vesselfm",
        ],
    )
    def test_model_group_files_exist(self, model_name: str) -> None:
        assert (CONFIGS_DIR / "model" / f"{model_name}.yaml").exists()

    def test_experiment_dir_exists(self) -> None:
        """configs/experiment/ dir exists for Hydra composition."""
        assert (CONFIGS_DIR / "experiment").is_dir()

    def test_old_experiments_dir_deleted(self) -> None:
        """Old configs/experiments/ directory should NOT exist."""
        assert not (CONFIGS_DIR / "experiments").exists(), (
            "configs/experiments/ should be deleted — fully migrated to configs/experiment/"
        )

    def test_hpo_dir_exists(self) -> None:
        """configs/hpo/ dir exists for HPO configs (separate schema)."""
        assert (CONFIGS_DIR / "hpo").is_dir()


class TestConfigGroupContents:
    """Verify config group YAML contents are correct."""

    def test_data_minivess_has_data_dir(self) -> None:
        cfg = _load_yaml(CONFIGS_DIR / "data" / "minivess.yaml")
        assert "data_dir" in cfg

    def test_training_default_has_seed(self) -> None:
        cfg = _load_yaml(CONFIGS_DIR / "training" / "default.yaml")
        assert "seed" in cfg
        assert cfg["seed"] == 42

    def test_training_default_has_compute(self) -> None:
        cfg = _load_yaml(CONFIGS_DIR / "training" / "default.yaml")
        assert "compute" in cfg

    def test_training_default_has_max_epochs(self) -> None:
        cfg = _load_yaml(CONFIGS_DIR / "training" / "default.yaml")
        assert "max_epochs" in cfg
        assert cfg["max_epochs"] == 100

    def test_checkpoint_standard_has_tracked_metrics(self) -> None:
        cfg = _load_yaml(CONFIGS_DIR / "checkpoint" / "standard.yaml")
        ckpt = cfg.get("checkpoint", cfg)
        assert "tracked_metrics" in ckpt
        assert len(ckpt["tracked_metrics"]) >= 5

    def test_checkpoint_standard_has_primary_metric(self) -> None:
        cfg = _load_yaml(CONFIGS_DIR / "checkpoint" / "standard.yaml")
        ckpt = cfg.get("checkpoint", cfg)
        assert "primary_metric" in ckpt

    def test_checkpoint_lightweight_fewer_metrics(self) -> None:
        std = _load_yaml(CONFIGS_DIR / "checkpoint" / "standard.yaml")
        lite = _load_yaml(CONFIGS_DIR / "checkpoint" / "lightweight.yaml")
        std_ckpt = std.get("checkpoint", std)
        lite_ckpt = lite.get("checkpoint", lite)
        assert len(lite_ckpt["tracked_metrics"]) < len(std_ckpt["tracked_metrics"]), (
            "Lightweight should have fewer tracked metrics"
        )

    def test_checkpoint_lightweight_lower_patience(self) -> None:
        lite = _load_yaml(CONFIGS_DIR / "checkpoint" / "lightweight.yaml")
        lite_ckpt = lite.get("checkpoint", lite)
        for m in lite_ckpt["tracked_metrics"]:
            assert m["patience"] <= 15, f"Lightweight patience too high for {m['name']}"

    def test_model_dynunet_yaml_model_key(self) -> None:
        cfg = _load_yaml(CONFIGS_DIR / "model" / "dynunet.yaml")
        assert cfg.get("model") == "dynunet"

    def test_model_vesselfm_yaml_model_key(self) -> None:
        cfg = _load_yaml(CONFIGS_DIR / "model" / "vesselfm.yaml")
        assert cfg.get("model") == "vesselfm"

    def test_model_vesselfm_has_architecture_params(self) -> None:
        cfg = _load_yaml(CONFIGS_DIR / "model" / "vesselfm.yaml")
        assert "architecture_params" in cfg
        params = cfg["architecture_params"]
        assert params.get("pretrained") is True
        assert params.get("normalization") == "percentile"


# ---------------------------------------------------------------------------
# Phase 2: Compose bridge
# ---------------------------------------------------------------------------


class TestComposeBridge:
    """Test the Hydra Compose API bridge."""

    def test_compose_module_importable(self) -> None:
        from minivess.config.compose import compose_experiment_config  # noqa: F401

    def test_compose_returns_dict(self) -> None:
        from minivess.config.compose import compose_experiment_config

        result = compose_experiment_config()
        assert isinstance(result, dict)

    def test_compose_has_base_keys(self) -> None:
        """Composed config should contain keys from base + data + training."""
        from minivess.config.compose import compose_experiment_config

        result = compose_experiment_config()
        # From training/default.yaml
        assert "seed" in result
        # From data/minivess.yaml
        assert "data_dir" in result

    def test_compose_with_model_override(self) -> None:
        from minivess.config.compose import compose_experiment_config

        result = compose_experiment_config(overrides=["model=vesselfm"])
        assert result.get("model") == "vesselfm"

    def test_compose_with_experiment_override(self) -> None:
        from minivess.config.compose import compose_experiment_config

        result = compose_experiment_config(experiment_name="vesselfm_zeroshot")
        assert result.get("experiment_name") == "vesselfm_zeroshot_eval"

    def test_compose_with_scalar_override(self) -> None:
        from minivess.config.compose import compose_experiment_config

        result = compose_experiment_config(overrides=["training.max_epochs=10"])
        # After merge, check the nested or flat key is set
        training = result.get("training", result)
        max_epochs = training.get("max_epochs", result.get("max_epochs"))
        assert max_epochs == 10

    def test_compose_fallback_without_hydra(self) -> None:
        """When hydra is not available, falls back to None or raises cleanly."""
        from minivess.config.compose import _hydra_available

        # Just verify the flag exists — actual fallback tested via mocking
        assert isinstance(_hydra_available(), bool)

    def test_plus_prefix_override_applies_new_key(self) -> None:
        """'+key=value' Hydra-style override: new key is applied correctly.

        This is the GCP smoke test pattern: '+mixed_precision=false' adds a key
        that doesn't exist in the base struct. The fallback manual merge must
        strip the '+' prefix so the key is stored as 'mixed_precision', not
        '+mixed_precision'. Regression test for the val_loss=NaN bug (2026-03-15).
        """
        from minivess.config.compose import compose_experiment_config

        result = compose_experiment_config(
            experiment_name="smoke_sam3_hybrid",
            overrides=["+mixed_precision=false", "++val_interval=1"],
        )
        assert result.get("mixed_precision") is False, (
            "+mixed_precision=false must set mixed_precision=False, not '+mixed_precision'"
        )
        assert result.get("val_interval") == 1, (
            "++val_interval=1 must override val_interval (was 3 in smoke_sam3_hybrid.yaml)"
        )

    def test_double_plus_prefix_override_existing_key(self) -> None:
        """'++key=value' Hydra force-override: works for both new and existing keys.

        '+val_interval=1' fails if val_interval already exists in the config.
        '++val_interval=1' works unconditionally. The manual merge fallback must
        strip '++' prefix the same way it strips '+'.
        """
        from minivess.config.compose import compose_experiment_config

        # val_interval=3 in smoke_sam3_hybrid.yaml — force override to 1
        result = compose_experiment_config(
            experiment_name="smoke_sam3_hybrid",
            overrides=["++val_interval=1"],
        )
        assert result.get("val_interval") == 1, (
            "++val_interval=1 must force-override val_interval=3 in smoke_sam3_hybrid"
        )


# ---------------------------------------------------------------------------
# Phase 7: Converted experiment configs
# ---------------------------------------------------------------------------


class TestConvertedExperiments:
    """Verify delta-only Hydra experiments are correct."""

    def test_dynunet_losses_hydra_experiment_exists(self) -> None:
        path = CONFIGS_DIR / "experiment" / "dynunet_losses.yaml"
        assert path.exists(), "Hydra experiment/dynunet_losses.yaml not found"

    def test_sam3_vanilla_hydra_experiment_exists(self) -> None:
        path = CONFIGS_DIR / "experiment" / "sam3_vanilla_baseline.yaml"
        assert path.exists()

    def test_converted_dynunet_losses_has_core_fields(self) -> None:
        """Hydra composed dynunet_losses has correct core fields."""
        from minivess.config.compose import compose_experiment_config

        composed = compose_experiment_config(experiment_name="dynunet_losses")
        assert composed.get("experiment_name") == "dynunet_loss_variation_v2"
        assert composed.get("model") == "dynunet"
        assert isinstance(composed.get("losses"), list)
        assert composed.get("seed") == 42

    @pytest.mark.parametrize(
        "experiment_name",
        [
            "dynunet_losses",
            "dynunet_half_width",
            "dynunet_topology",
            "dynunet_graph_topology",
            "dynunet_tffm_ablation",
            "dynunet_d2c_ablation",
            "dynunet_multitask_ablation",
            "dynunet_topology_all_approaches",
            "sam3_vanilla_baseline",
            "sam3_hybrid_fusion",
            "sam3_topolora_topology",
            "vesselfm_zeroshot",
            "vesselfm_finetune",
            "dynunet_e2e_debug",
            "dynunet_all_losses_debug",
            "dynunet_graph_topology_debug",
            "dynunet_topology_all_approaches_debug",
            "sam3_vanilla_debug",
            "sam3_topolora_debug",
            "sam3_hybrid_debug",
        ],
    )
    def test_all_experiments_compose_with_experiment_name(
        self, experiment_name: str
    ) -> None:
        """Every experiment config composes and has experiment_name."""
        from minivess.config.compose import compose_experiment_config

        result = compose_experiment_config(experiment_name=experiment_name)
        assert "experiment_name" in result, (
            f"{experiment_name} composed config missing experiment_name"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    assert isinstance(data, dict), f"Expected dict, got {type(data)} from {path}"
    return data
