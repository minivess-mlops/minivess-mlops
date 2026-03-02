"""Tests for conditions-based experiment runner (T3 — topology real-data plan)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from pathlib import Path


def _make_conditions_config() -> dict[str, Any]:
    """Minimal conditions-based experiment config."""
    return {
        "experiment_name": "test_topology",
        "model_family": "dynunet",
        "loss": "cbdice_cldice",
        "folds": 3,
        "epochs": 6,
        "conditions": [
            {
                "name": "baseline",
                "wrappers": [],
                "d2c_enabled": False,
            },
            {
                "name": "d2c_only",
                "wrappers": [],
                "d2c_enabled": True,
                "d2c_probability": 0.3,
            },
        ],
    }


def _make_losses_config() -> dict[str, Any]:
    """Legacy losses-based experiment config."""
    return {
        "experiment_name": "test_losses",
        "model": "dynunet",
        "losses": ["dice_ce", "cbdice_cldice"],
    }


class TestLoadExperimentConfig:
    """Test config loading with relaxed REQUIRED_FIELDS."""

    def test_conditions_config_loads(self, tmp_path: Path) -> None:
        """Config with 'conditions' key (no 'losses') loads without error."""
        import sys

        sys.path.insert(0, "scripts")
        from run_experiment import load_experiment_config

        config_path = tmp_path / "test.yaml"
        config_path.write_text(yaml.dump(_make_conditions_config()), encoding="utf-8")
        config = load_experiment_config(config_path)
        assert config["experiment_name"] == "test_topology"
        assert "conditions" in config

    def test_losses_config_still_loads(self, tmp_path: Path) -> None:
        """Legacy config with 'losses' key still loads."""
        import sys

        sys.path.insert(0, "scripts")
        from run_experiment import load_experiment_config

        config_path = tmp_path / "test.yaml"
        config_path.write_text(yaml.dump(_make_losses_config()), encoding="utf-8")
        config = load_experiment_config(config_path)
        assert "losses" in config


class TestDetectExperimentMode:
    """Test experiment mode detection."""

    def test_detects_conditions_mode(self) -> None:
        """Config with 'conditions' key -> 'conditions' mode."""
        import sys

        sys.path.insert(0, "scripts")
        from run_experiment import detect_experiment_mode

        config = _make_conditions_config()
        assert detect_experiment_mode(config) == "conditions"

    def test_detects_losses_mode(self) -> None:
        """Config with 'losses' key -> 'losses' mode."""
        import sys

        sys.path.insert(0, "scripts")
        from run_experiment import detect_experiment_mode

        config = _make_losses_config()
        assert detect_experiment_mode(config) == "losses"


class TestConditionsDryRun:
    """Test dry-run validation of conditions-based configs."""

    def test_dry_run_validates_conditions(self, tmp_path: Path) -> None:
        """Dry run reports condition names and count."""
        import sys

        sys.path.insert(0, "scripts")
        from run_experiment import run_dry_run

        config = _make_conditions_config()
        result = run_dry_run(config)
        assert result is not None
        # Should not crash; validation dict returned
        assert isinstance(result, dict)


class TestExtractExtraTargetKeys:
    """Test extraction of extra target keys from condition wrappers."""

    def test_no_wrappers_no_keys(self) -> None:
        """Baseline condition has no extra target keys."""
        import sys

        sys.path.insert(0, "scripts")
        from run_experiment import extract_extra_target_keys

        condition: dict[str, Any] = {"name": "baseline", "wrappers": []}
        assert extract_extra_target_keys(condition) == []

    def test_multitask_extracts_gt_keys(self) -> None:
        """Multitask condition extracts gt_key from each auxiliary head."""
        import sys

        sys.path.insert(0, "scripts")
        from run_experiment import extract_extra_target_keys

        condition: dict[str, Any] = {
            "name": "multitask",
            "wrappers": [
                {
                    "type": "multitask",
                    "auxiliary_heads": [
                        {
                            "name": "sdf",
                            "type": "regression",
                            "out_channels": 1,
                            "gt_key": "sdf",
                        },
                        {
                            "name": "centerline_dist",
                            "type": "regression",
                            "out_channels": 1,
                            "gt_key": "centerline_dist",
                        },
                    ],
                }
            ],
        }
        keys = extract_extra_target_keys(condition)
        assert "sdf" in keys
        assert "centerline_dist" in keys


class TestDebugConfigFile:
    """Test the actual debug YAML config file loads correctly."""

    def test_debug_config_loads(self) -> None:
        """dynunet_topology_all_approaches_debug.yaml loads as conditions mode."""
        import sys
        from pathlib import Path

        sys.path.insert(0, "scripts")
        from run_experiment import detect_experiment_mode, load_experiment_config

        config_path = Path(
            "configs/experiments/dynunet_topology_all_approaches_debug.yaml"
        )
        if not config_path.exists():
            import pytest

            pytest.skip("Debug config not found")

        config = load_experiment_config(config_path)
        assert detect_experiment_mode(config) == "conditions"
        assert len(config["conditions"]) == 6
        assert config["max_epochs"] == 6
