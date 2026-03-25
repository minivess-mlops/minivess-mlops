"""Cloud readiness tests for train_flow.py — catch issues BEFORE burning GPU credits.

These tests verify code paths that execute during GCP factorial runs but had
ZERO test coverage before this file. Every test here represents a gap found
by reviewer agents during the pre-5th-pass QA audit.

Issue: integration-test-double-check.md, pre-5th-debug-run-qa.xml
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import yaml

TRAIN_FLOW_PATH = Path("src/minivess/orchestration/flows/train_flow.py")
FACTORIAL_YAML = Path("deployment/skypilot/train_factorial.yaml")


# ---------------------------------------------------------------------------
# Argparse fallback path — the EXACT code path SkyPilot uses
# ---------------------------------------------------------------------------


class TestArgparseFallbackPath:
    """The __main__ argparse path is what SkyPilot actually invokes.

    SkyPilot runs: python -m minivess.orchestration.flows.train_flow --model-family ...
    This uses the argparse path (EXPERIMENT env var NOT set).
    Every CLI arg must be accepted without error.
    """

    def test_argparse_accepts_all_skypilot_args(self) -> None:
        """Simulate the exact CLI args SkyPilot passes — must not raise."""
        source = TRAIN_FLOW_PATH.read_text(encoding="utf-8")

        # Extract all argparse argument names
        tree = ast.parse(source)
        argparse_args: list[str] = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and node.args
            ):
                first_arg = node.args[0]
                if isinstance(first_arg, ast.Constant) and isinstance(
                    first_arg.value, str
                ):
                    argparse_args.append(first_arg.value)

        # These are the args SkyPilot YAML passes (from train_factorial.yaml run section)
        required_skypilot_args = [
            "--model-family",
            "--loss-name",
            "--fold",
            "--with-aux-calib",
            "--max-epochs",
            "--max-train-volumes",
            "--max-val-volumes",
            "--experiment-name",
            "--zero-shot",
            "--eval-dataset",
            "--post-training-method",
        ]
        for arg in required_skypilot_args:
            assert arg in argparse_args, (
                f"SkyPilot passes {arg} but train_flow.py argparse does not accept it. "
                f"Available args: {argparse_args}"
            )

    def test_with_aux_calib_parsed_as_string_converted_to_bool(self) -> None:
        """--with-aux-calib 'true'/'false' must be string → bool conversion."""
        source = TRAIN_FLOW_PATH.read_text(encoding="utf-8")
        # Must have .lower() == "true" conversion
        assert (
            'with_aux_calib.lower() == "true"' in source
            or "with_aux_calib.lower() == 'true'" in source
        ), (
            "--with-aux-calib is parsed as string but must be converted to bool. "
            "SkyPilot passes 'true'/'false' as strings."
        )

    def test_zero_shot_parsed_as_string_converted_to_bool(self) -> None:
        """--zero-shot 'true'/'false' must be string → bool conversion."""
        source = TRAIN_FLOW_PATH.read_text(encoding="utf-8")
        assert (
            'zero_shot.lower() == "true"' in source
            or "zero_shot.lower() == 'true'" in source
        ), "--zero-shot is parsed as string but must be converted to bool."

    def test_fold_negative_one_means_all_folds(self) -> None:
        """--fold -1 must be interpreted as 'run all folds'."""
        source = TRAIN_FLOW_PATH.read_text(encoding="utf-8")
        assert "args.fold >= 0" in source or "fold >= 0" in source, (
            "--fold -1 must be treated as 'run all folds'. "
            "The argparse default is -1 and must be checked."
        )


# ---------------------------------------------------------------------------
# Bracket-aware HYDRA_OVERRIDES parsing
# ---------------------------------------------------------------------------


class TestBracketAwareHydraOverridesParsing:
    """HYDRA_OVERRIDES must preserve commas inside brackets [...]."""

    def _parse_overrides(self, overrides_str: str) -> list[str]:
        """Replicate the bracket-aware split logic from train_flow.py __main__."""
        result: list[str] = []
        current: list[str] = []
        depth = 0
        for ch in overrides_str:
            if ch == "[":
                depth += 1
                current.append(ch)
            elif ch == "]":
                depth -= 1
                current.append(ch)
            elif ch == "," and depth == 0:
                part = "".join(current).strip()
                if part:
                    result.append(part)
                current = []
            else:
                current.append(ch)
        part = "".join(current).strip()
        if part:
            result.append(part)
        return result

    def test_simple_comma_split(self) -> None:
        """Simple key=value pairs split on comma."""
        result = self._parse_overrides("loss_name=dice_ce,max_epochs=5")
        assert result == ["loss_name=dice_ce", "max_epochs=5"]

    def test_bracket_preserved(self) -> None:
        """Commas inside brackets are NOT split points."""
        result = self._parse_overrides("patch_size=[32,32,3],loss_name=dice_ce")
        assert result == ["patch_size=[32,32,3]", "loss_name=dice_ce"]

    def test_empty_string(self) -> None:
        """Empty string returns empty list."""
        result = self._parse_overrides("")
        assert result == []

    def test_single_override(self) -> None:
        """Single override without comma."""
        result = self._parse_overrides("max_epochs=10")
        assert result == ["max_epochs=10"]

    def test_nested_brackets(self) -> None:
        """Nested brackets are handled."""
        result = self._parse_overrides("x=[[1,2],[3,4]],y=5")
        assert result == ["x=[[1,2],[3,4]]", "y=5"]

    def test_trailing_comma(self) -> None:
        """Trailing comma produces correct output."""
        result = self._parse_overrides("a=1,")
        assert result == ["a=1"]


# ---------------------------------------------------------------------------
# Debug splits file validation
# ---------------------------------------------------------------------------


class TestDebugSplitsFile:
    """Debug splits file must exist and have correct format."""

    def test_debug_splits_file_exists(self) -> None:
        """configs/splits/debug_half_1fold.json must exist for debug runs."""
        splits_path = Path("configs/splits/debug_half_1fold.json")
        assert splits_path.exists(), (
            "Debug splits file missing. SkyPilot setup copies this to splits.json. "
            "Without it, debug factorial jobs fail in setup."
        )

    def test_debug_splits_file_valid_json(self) -> None:
        """Debug splits file must be valid JSON."""
        splits_path = Path("configs/splits/debug_half_1fold.json")
        if not splits_path.exists():
            pytest.skip("Debug splits file not found")
        data = json.loads(splits_path.read_text(encoding="utf-8"))
        assert isinstance(data, list), "Splits file must be a JSON array of folds"

    def test_debug_splits_has_one_fold(self) -> None:
        """Debug splits should have exactly 1 fold."""
        splits_path = Path("configs/splits/debug_half_1fold.json")
        if not splits_path.exists():
            pytest.skip("Debug splits file not found")
        data = json.loads(splits_path.read_text(encoding="utf-8"))
        assert len(data) == 1, f"Debug splits should have 1 fold, got {len(data)}"

    def test_debug_splits_has_train_val_keys(self) -> None:
        """Each fold must have 'train' and 'val' keys."""
        splits_path = Path("configs/splits/debug_half_1fold.json")
        if not splits_path.exists():
            pytest.skip("Debug splits file not found")
        data = json.loads(splits_path.read_text(encoding="utf-8"))
        for i, fold in enumerate(data):
            assert "train" in fold, f"Fold {i} missing 'train' key"
            assert "val" in fold, f"Fold {i} missing 'val' key"

    def test_debug_splits_has_half_data(self) -> None:
        """Debug splits should have ~23 train / ~12 val volumes (half of 47/23)."""
        splits_path = Path("configs/splits/debug_half_1fold.json")
        if not splits_path.exists():
            pytest.skip("Debug splits file not found")
        data = json.loads(splits_path.read_text(encoding="utf-8"))
        fold = data[0]
        n_train = len(fold["train"])
        n_val = len(fold["val"])
        assert 15 <= n_train <= 30, f"Expected ~23 train volumes, got {n_train}"
        assert 8 <= n_val <= 18, f"Expected ~12 val volumes, got {n_val}"

    def test_production_splits_file_exists(self) -> None:
        """configs/splits/3fold_seed42.json must exist for production runs."""
        splits_path = Path("configs/splits/3fold_seed42.json")
        assert splits_path.exists(), "Production splits file missing"

    def test_production_splits_has_three_folds(self) -> None:
        """Production splits should have 3 folds."""
        splits_path = Path("configs/splits/3fold_seed42.json")
        if not splits_path.exists():
            pytest.skip("Production splits file not found")
        data = json.loads(splits_path.read_text(encoding="utf-8"))
        assert len(data) == 3, f"Production splits should have 3 folds, got {len(data)}"


# ---------------------------------------------------------------------------
# HF weight download failure handling
# ---------------------------------------------------------------------------


class TestHfWeightDownloadFailure:
    """Setup script must handle HuggingFace download failures gracefully."""

    def test_setup_has_hf_login(self) -> None:
        """Setup must authenticate with HuggingFace before weight download."""
        config = yaml.safe_load(FACTORIAL_YAML.read_text(encoding="utf-8"))
        setup = config.get("setup", "")
        assert "huggingface-cli login" in setup, (
            "Setup must login to HuggingFace for gated model weights"
        )

    def test_setup_hf_token_check(self) -> None:
        """Setup must check if HF_TOKEN is set before login."""
        config = yaml.safe_load(FACTORIAL_YAML.read_text(encoding="utf-8"))
        setup = config.get("setup", "")
        assert "HF_TOKEN" in setup, "Setup must check HF_TOKEN"

    def test_setup_precaches_sam3_weights(self) -> None:
        """Setup must pre-cache SAM3 weights during setup (not during GPU time)."""
        config = yaml.safe_load(FACTORIAL_YAML.read_text(encoding="utf-8"))
        setup = config.get("setup", "")
        assert "sam3" in setup.lower() or "SAM3" in setup, (
            "Setup must pre-cache SAM3 weights to avoid burning GPU time on download"
        )

    def test_setup_precaches_vesselfm_weights(self) -> None:
        """Setup must pre-cache VesselFM weights."""
        config = yaml.safe_load(FACTORIAL_YAML.read_text(encoding="utf-8"))
        setup = config.get("setup", "")
        assert "vesselfm" in setup.lower() or "vesselFM" in setup, (
            "Setup must pre-cache VesselFM weights"
        )


# ---------------------------------------------------------------------------
# Spot resume state verification
# ---------------------------------------------------------------------------


class TestSpotResumeState:
    """Spot preemption recovery must be wired correctly."""

    def test_check_resume_state_function_exists(self) -> None:
        """check_resume_state_task must exist in train_flow.py."""
        source = TRAIN_FLOW_PATH.read_text(encoding="utf-8")
        assert "check_resume_state_task" in source or "check_resume_state" in source, (
            "Spot resume detection function missing from train_flow.py"
        )

    def test_config_fingerprint_function_exists(self) -> None:
        """compute_config_fingerprint must exist for resume matching."""
        source = TRAIN_FLOW_PATH.read_text(encoding="utf-8")
        assert "compute_config_fingerprint" in source, (
            "Config fingerprint function missing — needed for spot resume matching"
        )

    def test_epoch_latest_referenced(self) -> None:
        """train_flow must reference epoch_latest for checkpoint resume."""
        source = TRAIN_FLOW_PATH.read_text(encoding="utf-8")
        assert "epoch_latest" in source, (
            "epoch_latest not referenced — training cannot resume from checkpoint"
        )


# ---------------------------------------------------------------------------
# Training flow function signature completeness
# ---------------------------------------------------------------------------


class TestTrainingFlowSignature:
    """training_flow() must accept ALL factorial parameters."""

    def _get_training_flow_params(self) -> list[str]:
        """Extract all parameter names from training_flow (positional + keyword-only)."""
        source = TRAIN_FLOW_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "training_flow":
                return [arg.arg for arg in node.args.args] + [
                    arg.arg for arg in node.args.kwonlyargs
                ]
        raise AssertionError("training_flow function not found")

    def test_training_flow_accepts_post_training_method(self) -> None:
        """training_flow must have post_training_method parameter."""
        params = self._get_training_flow_params()
        assert "post_training_method" in params, (
            f"training_flow() missing post_training_method parameter. Has: {params}"
        )

    def test_training_flow_accepts_zero_shot(self) -> None:
        """training_flow must have zero_shot parameter."""
        params = self._get_training_flow_params()
        assert "zero_shot" in params, (
            f"training_flow() missing zero_shot parameter. Has: {params}"
        )

    def test_training_flow_accepts_eval_dataset(self) -> None:
        """training_flow must have eval_dataset parameter."""
        params = self._get_training_flow_params()
        assert "eval_dataset" in params, (
            f"training_flow() missing eval_dataset parameter. Has: {params}"
        )

    def test_training_flow_accepts_with_aux_calib(self) -> None:
        """training_flow must have with_aux_calib parameter."""
        params = self._get_training_flow_params()
        assert "with_aux_calib" in params, (
            f"training_flow() missing with_aux_calib parameter. Has: {params}"
        )
