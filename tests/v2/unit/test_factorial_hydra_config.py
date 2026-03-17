"""Tests for Hydra config integration of aux_calib factorial factor.

Validates that with_aux_calib and aux_calib_weight are exposed via Hydra
config composition and that factorial_base.yaml loads correctly.
"""

from __future__ import annotations


class TestHydraConfigAuxCalibDefaultFalse:
    """T5: with_aux_calib defaults to false."""

    def test_hydra_config_aux_calib_default_false(self) -> None:
        from minivess.config.compose import compose_experiment_config

        cfg = compose_experiment_config()
        assert cfg.get("with_aux_calib") is False or cfg.get("with_aux_calib") is None


class TestHydraConfigAuxCalibTrue:
    """T5: Hydra override with_aux_calib=true works."""

    def test_hydra_config_aux_calib_true(self) -> None:
        from minivess.config.compose import compose_experiment_config

        cfg = compose_experiment_config(overrides=["with_aux_calib=true"])
        assert cfg["with_aux_calib"] is True


class TestHydraConfigAuxCalibWeight:
    """T5: aux_calib_weight is overridable."""

    def test_hydra_config_aux_calib_weight(self) -> None:
        from minivess.config.compose import compose_experiment_config

        cfg = compose_experiment_config(overrides=["aux_calib_weight=0.5"])
        assert cfg["aux_calib_weight"] == 0.5


class TestFactorialBaseConfigLoads:
    """T5: factorial_base.yaml experiment config loads."""

    def test_factorial_base_config_loads(self) -> None:
        from minivess.config.compose import compose_experiment_config

        cfg = compose_experiment_config(experiment_name="factorial_base")
        assert cfg["experiment_name"] == "factorial_base"
        assert cfg.get("max_epochs") == 50


class TestFactorialBaseConfigOverride:
    """T5: factorial_base.yaml can be overridden."""

    def test_factorial_base_config_override(self) -> None:
        from minivess.config.compose import compose_experiment_config

        cfg = compose_experiment_config(
            experiment_name="factorial_base",
            overrides=["with_aux_calib=true", "model=dynunet"],
        )
        assert cfg["with_aux_calib"] is True
