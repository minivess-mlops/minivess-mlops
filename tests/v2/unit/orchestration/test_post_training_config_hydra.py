"""Tests for Hydra post-training config composition.

Verifies that configs/post_training/{none,swag}.yaml load correctly
and match the SWAGPluginConfig Pydantic model.

Plan: docs/planning/training-and-post-training-into-two-subflows-under-one-flow.md
"""

from __future__ import annotations

from pathlib import Path

import yaml


CONFIGS_DIR = Path("configs/post_training")


class TestPostTrainingYamlFiles:
    """Verify YAML config files exist and parse correctly."""

    def test_none_yaml_loads(self) -> None:
        """configs/post_training/none.yaml must load and have method='none'."""
        cfg_path = CONFIGS_DIR / "none.yaml"
        assert cfg_path.exists(), f"Missing {cfg_path}"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        assert cfg["method"] == "none"

    def test_swag_yaml_loads(self) -> None:
        """configs/post_training/swag.yaml must load and have method='swag'."""
        cfg_path = CONFIGS_DIR / "swag.yaml"
        assert cfg_path.exists(), f"Missing {cfg_path}"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        assert cfg["method"] == "swag"

    def test_swag_yaml_has_required_keys(self) -> None:
        """SWAG config must have swa_lr, swa_epochs, max_rank, n_samples, update_bn."""
        cfg_path = CONFIGS_DIR / "swag.yaml"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        swag = cfg.get("swag", {})
        for key in ("swa_lr", "swa_epochs", "max_rank", "n_samples", "update_bn"):
            assert key in swag, f"SWAG config missing key: {key}"

    def test_config_matches_swag_plugin_config(self) -> None:
        """SWAG YAML values must be valid per SWAGPluginConfig Pydantic model."""
        from minivess.config.post_training_config import SWAGPluginConfig

        cfg_path = CONFIGS_DIR / "swag.yaml"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        swag_dict = cfg.get("swag", {})
        # Must not raise ValidationError
        plugin_cfg = SWAGPluginConfig(**swag_dict)
        assert plugin_cfg.swa_lr > 0
        assert plugin_cfg.swa_epochs >= 1
        assert plugin_cfg.max_rank >= 1


class TestBaseYamlDefaultsIncludePostTraining:
    """Verify base.yaml defaults list includes post_training group."""

    def test_base_yaml_has_post_training_default(self) -> None:
        """configs/base.yaml defaults must include post_training: none."""
        base_path = Path("configs/base.yaml")
        assert base_path.exists(), f"Missing {base_path}"
        cfg = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        defaults = cfg.get("defaults", [])
        # Defaults is a list of dicts or strings
        post_training_found = False
        for item in defaults:
            if isinstance(item, dict) and "post_training" in item:
                assert item["post_training"] == "none", (
                    f"post_training default should be 'none', got {item['post_training']}"
                )
                post_training_found = True
        assert post_training_found, (
            "base.yaml defaults list missing 'post_training' group"
        )
