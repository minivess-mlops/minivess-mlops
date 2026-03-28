"""Tests for model_overrides in factorial config files.

Verifies that SAM3 models get batch_size=1 + gradient_accumulation_steps
to avoid L4 OOM, while DynUNet/MambaVesselNet use global defaults.

Issue: #940 (SAM3 batch_size=1 fix)
Plan: docs/planning/sam3-batch-size-1-and-robustification-plan.xml Task 1.1
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

CONFIGS_DIR = pathlib.Path("configs/factorial")

# Configs that include SAM3 models in factors.training.model_family
SAM3_CONFIGS = ["debug.yaml", "paper_full.yaml"]

# All factorial configs
ALL_CONFIGS = ["debug.yaml", "paper_full.yaml", "smoke_test.yaml", "smoke_local.yaml"]

SAM3_TRAINABLE_MODELS = ["sam3_topolora", "sam3_hybrid"]
SAM3_ZEROSHOT_MODELS = ["sam3_vanilla"]


def _load_config(name: str) -> dict:
    path = CONFIGS_DIR / name
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class TestModelOverridesPresent:
    """model_overrides must exist in configs that sweep SAM3 models."""

    @pytest.mark.parametrize("config_name", SAM3_CONFIGS)
    def test_model_overrides_section_exists(self, config_name: str) -> None:
        cfg = _load_config(config_name)
        assert "model_overrides" in cfg, (
            f"{config_name} must have model_overrides section for SAM3 VRAM safety"
        )

    @pytest.mark.parametrize("config_name", SAM3_CONFIGS)
    @pytest.mark.parametrize("model", SAM3_TRAINABLE_MODELS)
    def test_sam3_trainable_batch_size_1(self, config_name: str, model: str) -> None:
        cfg = _load_config(config_name)
        overrides = cfg.get("model_overrides", {})
        assert model in overrides, f"{config_name}: missing override for {model}"
        assert overrides[model]["batch_size"] == 1, (
            f"{config_name}: {model} must have batch_size=1 (OOM at BS=2 on L4)"
        )

    @pytest.mark.parametrize("config_name", SAM3_CONFIGS)
    @pytest.mark.parametrize("model", SAM3_TRAINABLE_MODELS)
    def test_sam3_trainable_grad_accum_4(self, config_name: str, model: str) -> None:
        cfg = _load_config(config_name)
        overrides = cfg.get("model_overrides", {})
        assert model in overrides, f"{config_name}: missing override for {model}"
        assert overrides[model]["gradient_accumulation_steps"] == 4, (
            f"{config_name}: {model} must have gradient_accumulation_steps=4"
        )

    @pytest.mark.parametrize("config_name", SAM3_CONFIGS)
    def test_sam3_vanilla_override(self, config_name: str) -> None:
        """SAM3 Vanilla is zero-shot (frozen encoder), BS=1 with no grad accum."""
        cfg = _load_config(config_name)
        overrides = cfg.get("model_overrides", {})
        assert "sam3_vanilla" in overrides, (
            f"{config_name}: missing override for sam3_vanilla"
        )
        assert overrides["sam3_vanilla"]["batch_size"] == 1
        assert overrides["sam3_vanilla"]["gradient_accumulation_steps"] == 1


class TestModelOverridesConsistency:
    """model_overrides must be identical across configs that share SAM3 factors."""

    def test_sam3_overrides_identical_across_configs(self) -> None:
        """Rule 27: debug = production — overrides must match."""
        overrides_by_config = {}
        for name in SAM3_CONFIGS:
            cfg = _load_config(name)
            overrides = cfg.get("model_overrides", {})
            # Extract only SAM3 entries for comparison
            sam3_overrides = {
                k: v for k, v in overrides.items() if k.startswith("sam3")
            }
            overrides_by_config[name] = sam3_overrides

        configs = list(overrides_by_config.keys())
        for i in range(1, len(configs)):
            assert overrides_by_config[configs[0]] == overrides_by_config[configs[i]], (
                f"model_overrides mismatch: {configs[0]} vs {configs[i]}"
            )


class TestNonSAM3ConfigsClean:
    """Configs without SAM3 don't need model_overrides, but if present must be valid."""

    def test_smoke_local_no_sam3_no_overrides_required(self) -> None:
        """smoke_local only uses dynunet — no model_overrides needed."""
        cfg = _load_config("smoke_local.yaml")
        models = cfg["factors"]["training"]["model_family"]
        has_sam3 = any(m.startswith("sam3") for m in models)
        if not has_sam3:
            # model_overrides is optional when no SAM3
            return
        # If present, must be valid
        overrides = cfg.get("model_overrides", {})
        for _model, settings in overrides.items():
            assert "batch_size" in settings
            assert "gradient_accumulation_steps" in settings


class TestGradientCheckpointing:
    """SAM3 trainable models must have gradient_checkpointing: true in model_overrides.

    Gradient checkpointing reduces SAM3 TopoLoRA peak VRAM from ~22 GiB to ~10 GiB.
    Issue: #966, Plan: sam3-gradient-checkpointing-plan.xml Task 3.
    """

    @pytest.mark.parametrize("config_name", SAM3_CONFIGS)
    @pytest.mark.parametrize("model", SAM3_TRAINABLE_MODELS)
    def test_sam3_trainable_gradient_checkpointing_enabled(
        self, config_name: str, model: str
    ) -> None:
        """SAM3 trainable models must have gradient_checkpointing: true."""
        cfg = _load_config(config_name)
        overrides = cfg.get("model_overrides", {})
        assert model in overrides, f"{config_name}: missing override for {model}"
        assert overrides[model].get("gradient_checkpointing") is True, (
            f"{config_name}: {model} must have gradient_checkpointing: true "
            f"to reduce VRAM ~22 GiB → ~10 GiB on L4"
        )

    @pytest.mark.parametrize("config_name", SAM3_CONFIGS)
    def test_sam3_vanilla_no_gradient_checkpointing(self, config_name: str) -> None:
        """SAM3 Vanilla (frozen encoder) should NOT have gradient checkpointing."""
        cfg = _load_config(config_name)
        overrides = cfg.get("model_overrides", {})
        if "sam3_vanilla" in overrides:
            # Vanilla uses frozen encoder — gradient checkpointing is irrelevant
            assert not overrides["sam3_vanilla"].get("gradient_checkpointing", False), (
                f"{config_name}: sam3_vanilla should not have gradient_checkpointing "
                f"(frozen encoder uses no_grad, no activations stored)"
            )

    @pytest.mark.parametrize("config_name", SAM3_CONFIGS)
    def test_gradient_checkpointing_consistent_across_configs(
        self, config_name: str
    ) -> None:
        """Rule 27: debug = production — gradient_checkpointing must be identical."""
        # Already implicitly covered by test_sam3_overrides_identical_across_configs,
        # but this makes the gradient_checkpointing requirement explicit.
        cfg = _load_config(config_name)
        overrides = cfg.get("model_overrides", {})
        for model in SAM3_TRAINABLE_MODELS:
            assert overrides.get(model, {}).get("gradient_checkpointing") is True


class TestEffectiveBatchSize:
    """Verify effective batch size is consistent across models."""

    @pytest.mark.parametrize("config_name", SAM3_CONFIGS)
    def test_global_batch_size_unchanged(self, config_name: str) -> None:
        """Global fixed.batch_size must remain 2 (DynUNet/MambaVesselNet default)."""
        cfg = _load_config(config_name)
        assert cfg["fixed"]["batch_size"] == 2

    @pytest.mark.parametrize("config_name", SAM3_CONFIGS)
    def test_sam3_effective_batch_size_reasonable(self, config_name: str) -> None:
        """effective_bs = batch_size * grad_accum_steps should be >= 2."""
        cfg = _load_config(config_name)
        overrides = cfg.get("model_overrides", {})
        for model in SAM3_TRAINABLE_MODELS:
            settings = overrides.get(model, {})
            bs = settings.get("batch_size", cfg["fixed"]["batch_size"])
            accum = settings.get("gradient_accumulation_steps", 1)
            effective = bs * accum
            assert effective >= 2, (
                f"{config_name}: {model} effective_bs={effective} too small"
            )
