"""Tests for ProfilingConfig model and Hydra config group integration.

TDD RED phase for T0.1 (#644): ProfilingConfig is a STANDALONE Pydantic
model — NOT a field on TrainingConfig. It is constructed from
config_dict.get("profiling", {}) in train_one_fold_task() and passed
separately to the trainer.

Validates:
- ProfilingConfig defaults (enabled=True, epochs=5, etc.)
- Validation: enabled=True requires epochs>=1
- Standalone: NOT on TrainingConfig
- Hydra integration: base.yaml defaults, debug overrides, compose resolution
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from minivess.config.models import ProfilingConfig, TrainingConfig

# ── Project paths ──────────────────────────────────────────────────────
CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "configs"
BASE_YAML = CONFIGS_DIR / "base.yaml"
PROFILING_DIR = CONFIGS_DIR / "profiling"


class TestProfilingConfigDefaults:
    """Verify ProfilingConfig default field values."""

    def test_defaults(self) -> None:
        cfg = ProfilingConfig()
        assert cfg.enabled is True
        assert cfg.epochs == 5
        assert cfg.activities == ["cpu", "cuda"]
        assert cfg.profile_memory is True
        assert cfg.with_flops is True
        assert cfg.export_chrome_trace is True

    def test_compress_traces_default_true(self) -> None:
        cfg = ProfilingConfig()
        assert cfg.compress_traces is True

    def test_trace_size_limit_default_50(self) -> None:
        cfg = ProfilingConfig()
        assert cfg.trace_size_limit_mb == 50

    def test_record_shapes_default_false(self) -> None:
        cfg = ProfilingConfig()
        assert cfg.record_shapes is False

    def test_with_stack_default_false(self) -> None:
        cfg = ProfilingConfig()
        assert cfg.with_stack is False


class TestProfilingConfigValidation:
    """Verify ProfilingConfig validation rules."""

    def test_enabled_epochs_ge_1(self) -> None:
        """enabled=True + epochs=0 must raise ValidationError."""
        with pytest.raises(Exception):  # noqa: B017 — Pydantic ValidationError
            ProfilingConfig(enabled=True, epochs=0)

    def test_disabled_epochs_zero_ok(self) -> None:
        """enabled=False + epochs=0 is valid (no validation error)."""
        cfg = ProfilingConfig(enabled=False, epochs=0)
        assert cfg.enabled is False
        assert cfg.epochs == 0

    def test_disabled_epochs_any_value_ok(self) -> None:
        """When disabled, epochs value is irrelevant."""
        cfg = ProfilingConfig(enabled=False, epochs=999)
        assert cfg.epochs == 999


class TestProfilingConfigStandalone:
    """Verify ProfilingConfig is NOT on TrainingConfig."""

    def test_not_on_training_config(self) -> None:
        """TrainingConfig must have NO 'profiling' attribute."""
        assert not hasattr(TrainingConfig, "profiling"), (
            "ProfilingConfig must be standalone — NOT a field on TrainingConfig. "
            "See RC4 in the plan."
        )

    def test_not_in_training_config_fields(self) -> None:
        """TrainingConfig.model_fields must not contain 'profiling'."""
        assert "profiling" not in TrainingConfig.model_fields


class TestProfilingHydraYAMLFiles:
    """Verify Hydra config group YAML files exist with correct values."""

    def test_default_yaml_exists(self) -> None:
        assert (PROFILING_DIR / "default.yaml").exists()

    def test_debug_yaml_exists(self) -> None:
        assert (PROFILING_DIR / "debug.yaml").exists()

    def test_disabled_yaml_exists(self) -> None:
        assert (PROFILING_DIR / "disabled.yaml").exists()

    def test_default_yaml_enabled_true(self) -> None:
        with open(PROFILING_DIR / "default.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg["enabled"] is True
        assert cfg["epochs"] == 5

    def test_debug_yaml_epochs_2(self) -> None:
        with open(PROFILING_DIR / "debug.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg["epochs"] == 2

    def test_disabled_yaml_enabled_false(self) -> None:
        with open(PROFILING_DIR / "disabled.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg["enabled"] is False


class TestBaseYAMLProfilingDefault:
    """Verify base.yaml includes profiling in defaults list."""

    def test_base_yaml_includes_profiling_default(self) -> None:
        with open(BASE_YAML, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        defaults = cfg.get("defaults", [])
        # Look for {"profiling": "default"} in defaults list
        profiling_entries = [
            d for d in defaults if isinstance(d, dict) and "profiling" in d
        ]
        assert len(profiling_entries) == 1, (
            f"base.yaml must have exactly one profiling default, "
            f"found: {profiling_entries}"
        )
        assert profiling_entries[0]["profiling"] == "default"


class TestDebugConfigsOverrideProfiling:
    """Verify debug experiment configs include profiling override."""

    @pytest.mark.parametrize(
        "config_name",
        [
            "debug_single_model",
            "debug_all_models",
            "debug_full_pipeline",
            "debug_multi_loss",
        ],
    )
    def test_debug_configs_override_profiling(self, config_name: str) -> None:
        exp_path = CONFIGS_DIR / "experiment" / f"{config_name}.yaml"
        assert exp_path.exists(), f"Missing experiment config: {exp_path}"
        with open(exp_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        defaults = cfg.get("defaults", [])
        # Look for "override /profiling: debug" in defaults list
        profiling_overrides = [
            d for d in defaults if isinstance(d, dict) and "override /profiling" in d
        ]
        assert len(profiling_overrides) == 1, (
            f"{config_name}.yaml must have 'override /profiling: debug' "
            f"in defaults, found: {defaults}"
        )
        assert profiling_overrides[0]["override /profiling"] == "debug"


class TestComposeExperimentResolvesProfiling:
    """Verify compose_experiment_config() produces a profiling key."""

    def test_compose_experiment_resolves_profiling(self) -> None:
        from minivess.config.compose import compose_experiment_config

        cfg = compose_experiment_config()
        assert "profiling" in cfg, (
            "compose_experiment_config() must resolve 'profiling' key. "
            "Check base.yaml defaults list."
        )
        profiling = cfg["profiling"]
        assert isinstance(profiling, dict)
        assert profiling.get("enabled") is True

    def test_compose_debug_experiment_uses_debug_profiling(self) -> None:
        from minivess.config.compose import compose_experiment_config

        cfg = compose_experiment_config(experiment_name="debug_single_model")
        profiling = cfg.get("profiling", {})
        assert isinstance(profiling, dict)
        assert profiling.get("epochs") == 2, (
            "Debug experiment should resolve to profiling/debug.yaml with epochs=2"
        )


class TestProfilingConfigExport:
    """Verify ProfilingConfig is exported from config package."""

    def test_importable_from_config_package(self) -> None:
        from minivess.config import ProfilingConfig as PC

        assert PC is ProfilingConfig
