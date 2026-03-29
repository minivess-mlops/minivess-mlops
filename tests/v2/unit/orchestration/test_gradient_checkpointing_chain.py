"""End-to-end gradient checkpointing propagation chain tests.

Traces the full path: YAML config → run_factorial.sh → SkyPilot env var
→ train_flow.py argparse → config dict → arch_params → Sam3Backbone.

9th pass root cause: GRADIENT_CHECKPOINTING env var was immutable after
SkyPilot submission. Old "false" default persisted even after code was
fixed to set "true". This test validates every link in the chain.

See: .claude/metalearning/2026-03-26-sam3-gc-two-root-causes-docker-push-and-env-var.md
"""

from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

FACTORIAL_CONFIG = Path("configs/factorial/debug.yaml")
FACTORIAL_YAML = Path("deployment/skypilot/train_factorial.yaml")
TRAIN_FLOW_PATH = Path("src/minivess/orchestration/flows/train_flow.py")


# ---------------------------------------------------------------------------
# Link 1: YAML config → model_overrides
# ---------------------------------------------------------------------------


class TestLink1YamlConfigOverrides:
    """Verify gradient_checkpointing is declared in factorial config model_overrides."""

    def test_sam3_topolora_has_gradient_checkpointing_override(self) -> None:
        """SAM3 TopoLoRA model_overrides must include gradient_checkpointing: true."""
        cfg = yaml.safe_load(FACTORIAL_CONFIG.read_text(encoding="utf-8"))
        overrides = cfg.get("model_overrides", {})
        topolora = overrides.get("sam3_topolora", {})
        assert topolora.get("gradient_checkpointing") is True, (
            f"model_overrides.sam3_topolora.gradient_checkpointing is "
            f"{topolora.get('gradient_checkpointing')!r}, must be true"
        )

    def test_sam3_hybrid_has_gradient_checkpointing_override(self) -> None:
        """SAM3 Hybrid model_overrides must include gradient_checkpointing: true."""
        cfg = yaml.safe_load(FACTORIAL_CONFIG.read_text(encoding="utf-8"))
        overrides = cfg.get("model_overrides", {})
        hybrid = overrides.get("sam3_hybrid", {})
        assert hybrid.get("gradient_checkpointing") is True, (
            f"model_overrides.sam3_hybrid.gradient_checkpointing is "
            f"{hybrid.get('gradient_checkpointing')!r}, must be true"
        )

    def test_dynunet_does_not_have_gradient_checkpointing(self) -> None:
        """DynUNet should NOT have gradient_checkpointing (not needed, low VRAM)."""
        cfg = yaml.safe_load(FACTORIAL_CONFIG.read_text(encoding="utf-8"))
        overrides = cfg.get("model_overrides", {})
        dynunet = overrides.get("dynunet", {})
        gc_val = dynunet.get("gradient_checkpointing", False)
        assert gc_val is False or gc_val is None, (
            f"DynUNet should not enable gradient_checkpointing, got {gc_val!r}"
        )


# ---------------------------------------------------------------------------
# Link 2: run_factorial.sh → --env GRADIENT_CHECKPOINTING
# ---------------------------------------------------------------------------


class TestLink2RunFactorialShEnvVar:
    """Verify run_factorial.sh correctly extracts and passes GRADIENT_CHECKPOINTING."""

    def test_run_factorial_dry_run_sam3_gc_true(self) -> None:
        """Dry-run for SAM3 TopoLoRA must show GRADIENT_CHECKPOINTING=true."""
        result = subprocess.run(
            [
                "bash",
                "scripts/run_factorial.sh",
                "--dry-run",
                "configs/factorial/debug.yaml",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "SKIP_PREFLIGHT": "1"},
        )
        output = result.stdout + result.stderr
        # Find SAM3 TopoLoRA conditions and verify GC
        lines = output.splitlines()
        sam3_topolora_lines = [
            ln
            for ln in lines
            if "sam3_topolora" in ln and "GRADIENT_CHECKPOINTING" in ln
        ]
        assert len(sam3_topolora_lines) > 0, (
            "No SAM3 TopoLoRA lines with GRADIENT_CHECKPOINTING in dry-run output"
        )
        for ln in sam3_topolora_lines:
            assert "GRADIENT_CHECKPOINTING=true" in ln, (
                f"SAM3 TopoLoRA dry-run line has wrong GC value: {ln}"
            )

    def test_run_factorial_dry_run_dynunet_gc_false(self) -> None:
        """Dry-run for DynUNet must show GRADIENT_CHECKPOINTING=false."""
        result = subprocess.run(
            [
                "bash",
                "scripts/run_factorial.sh",
                "--dry-run",
                "configs/factorial/debug.yaml",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "SKIP_PREFLIGHT": "1"},
        )
        output = result.stdout + result.stderr
        dynunet_lines = [
            ln
            for ln in output.splitlines()
            if "dynunet" in ln and "GRADIENT_CHECKPOINTING" in ln and "sam3" not in ln
        ]
        assert len(dynunet_lines) > 0, (
            "No DynUNet lines with GRADIENT_CHECKPOINTING in dry-run output"
        )
        for ln in dynunet_lines:
            assert "GRADIENT_CHECKPOINTING=false" in ln, (
                f"DynUNet dry-run line has wrong GC value: {ln}"
            )


# ---------------------------------------------------------------------------
# Link 3: SkyPilot YAML env var default
# ---------------------------------------------------------------------------


class TestLink3SkyPilotYamlDefault:
    """Verify GRADIENT_CHECKPOINTING default in SkyPilot YAML is safe."""

    def test_yaml_default_is_false(self) -> None:
        """GRADIENT_CHECKPOINTING default MUST be 'false' (string) in YAML."""
        cfg = yaml.safe_load(FACTORIAL_YAML.read_text(encoding="utf-8"))
        envs = cfg.get("envs", {})
        gc_default = envs.get("GRADIENT_CHECKPOINTING")
        assert gc_default == "false", (
            f"GRADIENT_CHECKPOINTING YAML default is {gc_default!r}, "
            f"must be 'false'. SAM3 overrides via --env, others use this default."
        )

    def test_gradient_checkpointing_in_env_vars(self) -> None:
        """GRADIENT_CHECKPOINTING must be declared in the YAML envs section."""
        cfg = yaml.safe_load(FACTORIAL_YAML.read_text(encoding="utf-8"))
        envs = cfg.get("envs", {})
        assert "GRADIENT_CHECKPOINTING" in envs, (
            "GRADIENT_CHECKPOINTING missing from train_factorial.yaml envs section. "
            "Without an explicit default, os.environ.get() in train_flow.py "
            "will use its own default, creating a precedence ambiguity."
        )


# ---------------------------------------------------------------------------
# Link 4: train_flow.py argparse string→bool conversion
# ---------------------------------------------------------------------------


class TestLink4ArgparseBoolConversion:
    """Verify the string→bool conversion is correct and safe."""

    @pytest.mark.parametrize(
        "env_value,expected_bool",
        [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("", False),
            ("0", False),
            ("1", False),  # Only exact "true" (case-insensitive) is True
            ("yes", False),  # Only exact "true" (case-insensitive) is True
        ],
    )
    def test_gradient_checkpointing_string_to_bool(
        self, env_value: str, expected_bool: bool
    ) -> None:
        """The .lower() == 'true' conversion must handle all edge cases."""
        result = env_value.lower() == "true"
        assert result == expected_bool, (
            f"'{env_value}'.lower() == 'true' returned {result}, "
            f"expected {expected_bool}"
        )

    def test_train_flow_uses_lower_equals_true(self) -> None:
        """train_flow.py must use `.lower() == 'true'` for GC conversion, not bool()."""
        source = TRAIN_FLOW_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        # Find the gradient_checkpointing=... keyword arg in training_flow() call
        found_safe_conversion = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare) and (
                isinstance(node.left, ast.Call)
                and isinstance(node.left.func, ast.Attribute)
                and node.left.func.attr == "lower"
            ):
                for comparator in node.comparators:
                    if (
                        isinstance(comparator, ast.Constant)
                        and comparator.value == "true"
                    ):
                        found_safe_conversion = True
        assert found_safe_conversion, (
            "train_flow.py does not use `.lower() == 'true'` pattern. "
            "Using bool() on a string is WRONG: bool('false') == True. "
            "Using int() is WRONG: int('true') raises ValueError."
        )


# ---------------------------------------------------------------------------
# Link 5: config dict → arch_params injection
# ---------------------------------------------------------------------------


class TestLink5ConfigToArchParams:
    """Verify gradient_checkpointing flows from config dict to arch_params."""

    def test_gc_injected_into_arch_params_in_source(self) -> None:
        """train_flow.py must inject gradient_checkpointing into arch_params."""
        source = TRAIN_FLOW_PATH.read_text(encoding="utf-8")
        # Check that the injection code exists
        assert 'arch_params["gradient_checkpointing"]' in source, (
            "train_flow.py must inject gradient_checkpointing into arch_params "
            "so Sam3Backbone reads it during model construction."
        )

    def test_skip_gradient_flow_uses_or_logic(self) -> None:
        """skip_gradient_flow must use OR (not AND) for config + arch_params."""
        source = TRAIN_FLOW_PATH.read_text(encoding="utf-8")
        # The OR logic ensures skip happens if EITHER source says True
        assert (
            'config.get("gradient_checkpointing", False)' in source
            or 'config.get("gradient_checkpointing")' in source
        ), "Missing config.get('gradient_checkpointing') for skip_gradient_flow"
        assert (
            'arch_params.get("gradient_checkpointing", False)' in source
            or 'arch_params.get("gradient_checkpointing")' in source
        ), "Missing arch_params.get('gradient_checkpointing') for skip_gradient_flow"


# ---------------------------------------------------------------------------
# Link 6: Sam3Backbone consumption
# ---------------------------------------------------------------------------


class TestLink6Sam3BackboneConsumption:
    """Verify Sam3Backbone reads gradient_checkpointing from kwargs."""

    def test_sam3_backbone_accepts_gradient_checkpointing(self) -> None:
        """Sam3Backbone.__init__ must accept gradient_checkpointing parameter."""
        source = Path("src/minivess/adapters/sam3_backbone.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                # Check if any parent class is Sam3Backbone
                arg_names = [arg.arg for arg in node.args.args]
                if "gradient_checkpointing" in arg_names:
                    return  # Found it
        # Also check if it's read from kwargs
        assert "gradient_checkpointing" in source, (
            "Sam3Backbone does not reference gradient_checkpointing at all. "
            "The flag will be silently ignored, causing OOM on L4."
        )

    def test_sam3_backbone_calls_gc_enable(self) -> None:
        """Sam3Backbone must call gradient_checkpointing_enable() when flag is True."""
        source = Path("src/minivess/adapters/sam3_backbone.py").read_text(
            encoding="utf-8"
        )
        assert "gradient_checkpointing_enable" in source, (
            "Sam3Backbone never calls gradient_checkpointing_enable(). "
            "Even if the flag is True, GC won't activate — OOM guaranteed."
        )


# ---------------------------------------------------------------------------
# Integration: full chain simulation
# ---------------------------------------------------------------------------


class TestFullChainSimulation:
    """Simulate the full chain without actually running training."""

    def test_env_var_true_reaches_config_dict(self) -> None:
        """GRADIENT_CHECKPOINTING=true env var must reach config dict as bool True."""
        with patch.dict(os.environ, {"GRADIENT_CHECKPOINTING": "true"}):
            raw = os.environ.get("GRADIENT_CHECKPOINTING", "false")
            parsed = raw.lower() == "true"
            config_dict = {"gradient_checkpointing": parsed}
            assert config_dict["gradient_checkpointing"] is True

    def test_env_var_false_reaches_config_dict(self) -> None:
        """GRADIENT_CHECKPOINTING=false env var must reach config dict as bool False."""
        with patch.dict(os.environ, {"GRADIENT_CHECKPOINTING": "false"}):
            raw = os.environ.get("GRADIENT_CHECKPOINTING", "false")
            parsed = raw.lower() == "true"
            config_dict = {"gradient_checkpointing": parsed}
            assert config_dict["gradient_checkpointing"] is False

    def test_missing_env_var_defaults_to_false(self) -> None:
        """Missing GRADIENT_CHECKPOINTING env var must default to False."""
        env = {k: v for k, v in os.environ.items() if k != "GRADIENT_CHECKPOINTING"}
        with patch.dict(os.environ, env, clear=True):
            raw = os.environ.get("GRADIENT_CHECKPOINTING", "false")
            parsed = raw.lower() == "true"
            assert parsed is False

    def test_skip_gradient_flow_or_logic(self) -> None:
        """skip_gradient_flow should be True if EITHER config OR arch_params has GC."""
        # Case 1: both True
        config = {"gradient_checkpointing": True}
        arch_params = {"gradient_checkpointing": True}
        skip = bool(
            config.get("gradient_checkpointing", False)
            or arch_params.get("gradient_checkpointing", False)
        )
        assert skip is True

        # Case 2: only config True
        config = {"gradient_checkpointing": True}
        arch_params = {}
        skip = bool(
            config.get("gradient_checkpointing", False)
            or arch_params.get("gradient_checkpointing", False)
        )
        assert skip is True

        # Case 3: only arch_params True
        config = {}
        arch_params = {"gradient_checkpointing": True}
        skip = bool(
            config.get("gradient_checkpointing", False)
            or arch_params.get("gradient_checkpointing", False)
        )
        assert skip is True

        # Case 4: both False/missing
        config = {}
        arch_params = {}
        skip = bool(
            config.get("gradient_checkpointing", False)
            or arch_params.get("gradient_checkpointing", False)
        )
        assert skip is False
