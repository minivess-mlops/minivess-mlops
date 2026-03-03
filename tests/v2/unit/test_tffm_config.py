"""Tests for TFFM experiment config (T13 — #240)."""

from __future__ import annotations

from typing import Any


def _load_config() -> dict[str, Any]:
    from minivess.config.compose import compose_experiment_config

    return compose_experiment_config(experiment_name="dynunet_tffm_ablation")


class TestTFFMConfig:
    """Tests for TFFM experiment config."""

    def test_tffm_config_loads(self) -> None:
        """YAML config loads without error."""
        config = _load_config()
        assert config["experiment_name"] == "dynunet_tffm_ablation_v1"

    def test_tffm_config_two_conditions(self) -> None:
        """Baseline and TFFM conditions present."""
        config = _load_config()
        conditions = config["conditions"]
        assert len(conditions) == 2
        names = [c["name"] for c in conditions]
        assert "baseline" in names
        assert "tffm" in names

    def test_tffm_config_wrapper_params(self) -> None:
        """grid_size, hidden_dim correct in TFFM condition."""
        config = _load_config()
        conditions = config["conditions"]
        tffm_cond = next(c for c in conditions if c["name"] == "tffm")
        wrapper = tffm_cond["wrappers"][0]
        assert wrapper["type"] == "tffm"
        assert wrapper["grid_size"] == 8
        assert wrapper["hidden_dim"] == 32
