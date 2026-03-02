"""Tests for D2C ablation experiment config (T5 — #230)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _load_config() -> dict[str, Any]:
    config_path = Path("configs/experiments/dynunet_d2c_ablation.yaml")
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestD2CAblationConfig:
    """Tests for D2C ablation YAML config."""

    def test_d2c_ablation_config_loads(self) -> None:
        """YAML config loads without error."""
        config = _load_config()
        assert config["experiment_name"] == "dynunet_d2c_ablation_v1"

    def test_d2c_ablation_config_has_two_conditions(self) -> None:
        """Baseline and D2C conditions present."""
        config = _load_config()
        conditions = config["conditions"]
        assert len(conditions) == 2
        names = [c["name"] for c in conditions]
        assert "baseline" in names
        assert "d2c_augmented" in names

    def test_d2c_ablation_config_valid_loss(self) -> None:
        """Uses cbdice_cldice loss."""
        config = _load_config()
        assert config["loss"] == "cbdice_cldice"
