"""Tests for debug experiment config YAML and naming enforcement.

Covers:
- dynunet_e2e_debug.yaml structure and values
- is_debug_experiment() suffix detection
- validate_debug_experiment_name() epoch-limit enforcement
- Edge cases: empty names, no suffix, boundary epochs

Closes #187.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestDebugExperimentConfig:
    """Validate dynunet_e2e_debug composed config structure and values."""

    @pytest.fixture()
    def config(self) -> dict[str, Any]:
        from minivess.config.compose import compose_experiment_config

        return compose_experiment_config(experiment_name="dynunet_e2e_debug")

    def test_experiment_name_has_debug_suffix(self, config: dict[str, Any]) -> None:
        assert config["experiment_name"].endswith("_debug")

    def test_uses_default_loss(self, config: dict[str, Any]) -> None:
        assert config["losses"] == ["cbdice_cldice"]

    def test_max_epochs_debug_length(self, config: dict[str, Any]) -> None:
        assert config["max_epochs"] <= 20

    def test_num_folds(self, config: dict[str, Any]) -> None:
        assert config["num_folds"] == 3

    def test_checkpoint_section_present(self, config: dict[str, Any]) -> None:
        assert "checkpoint" in config
        ckpt = config["checkpoint"]
        assert "tracked_metrics" in ckpt
        assert "primary_metric" in ckpt

    def test_seed_is_42(self, config: dict[str, Any]) -> None:
        assert config["seed"] == 42


class TestExperimentNaming:
    """Validate experiment naming enforcement module."""

    def test_is_debug_with_suffix(self) -> None:
        from minivess.config.experiment_naming import is_debug_experiment

        assert is_debug_experiment("dynunet_e2e_debug") is True

    def test_is_debug_without_suffix(self) -> None:
        from minivess.config.experiment_naming import is_debug_experiment

        assert is_debug_experiment("dynunet_loss_variation_v2") is False

    def test_validate_debug_name_accepts_low_epochs_with_suffix(self) -> None:
        from minivess.config.experiment_naming import validate_debug_experiment_name

        result = validate_debug_experiment_name("my_exp_debug", max_epochs=10)
        assert result == "my_exp_debug"

    def test_validate_debug_name_rejects_low_epochs_without_suffix(self) -> None:
        from minivess.config.experiment_naming import validate_debug_experiment_name

        with pytest.raises(ValueError, match="_debug"):
            validate_debug_experiment_name("my_experiment", max_epochs=10)

    def test_validate_debug_name_allows_high_epochs_without_suffix(self) -> None:
        from minivess.config.experiment_naming import validate_debug_experiment_name

        result = validate_debug_experiment_name("my_experiment", max_epochs=100)
        assert result == "my_experiment"

    def test_validate_debug_name_boundary_20_epochs(self) -> None:
        from minivess.config.experiment_naming import validate_debug_experiment_name

        # 20 epochs is the boundary — should require _debug suffix
        with pytest.raises(ValueError, match="_debug"):
            validate_debug_experiment_name("my_experiment", max_epochs=20)

    def test_validate_debug_name_boundary_21_epochs(self) -> None:
        from minivess.config.experiment_naming import validate_debug_experiment_name

        # 21 epochs — no suffix needed
        result = validate_debug_experiment_name("my_experiment", max_epochs=21)
        assert result == "my_experiment"

    def test_validate_empty_name_raises(self) -> None:
        from minivess.config.experiment_naming import validate_debug_experiment_name

        with pytest.raises(ValueError, match="empty"):
            validate_debug_experiment_name("", max_epochs=10)
