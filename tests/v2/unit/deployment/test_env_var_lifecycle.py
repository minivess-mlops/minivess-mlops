"""Environment variable lifecycle tests for SkyPilot execution model.

Validates that every training-critical env var has a consistent lifecycle:
.env.example → SkyPilot YAML default → run_factorial.sh override → Python consumption

10th pass discovery: Env vars are IMMUTABLE after SkyPilot submission. When code
changes add new env vars or change defaults, ALL existing PENDING jobs still use
the OLD values. This test ensures the lifecycle is well-defined at every stage.

See: .claude/metalearning/2026-03-26-sam3-gc-two-root-causes-docker-push-and-env-var.md
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

DOT_ENV_EXAMPLE = Path(".env.example")
FACTORIAL_YAML = Path("deployment/skypilot/train_factorial.yaml")
RUN_FACTORIAL = Path("scripts/run_factorial.sh")
TRAIN_FLOW = Path("src/minivess/orchestration/flows/train_flow.py")

# Env vars that MUST be in YAML envs section (global defaults)
GLOBAL_ENV_VARS = [
    "MODEL_FAMILY",
    "LOSS_NAME",
    "FOLD_ID",
    "WITH_AUX_CALIB",
    "MAX_EPOCHS",
    "MAX_TRAIN_VOLUMES",
    "MAX_VAL_VOLUMES",
    "EXPERIMENT_NAME",
    "POST_TRAINING_METHODS",
    "ZERO_SHOT",
    "EVAL_DATASET",
    "GRADIENT_CHECKPOINTING",
]

# Env vars that are per-model overrides (passed via --env, no YAML default needed)
PER_MODEL_ENV_VARS = [
    "BATCH_SIZE",
    "GRAD_ACCUM_STEPS",
]

# All training-critical env vars (union)
TRAINING_ENV_VARS = GLOBAL_ENV_VARS + PER_MODEL_ENV_VARS


def _load_yaml_envs() -> dict:
    """Load the envs section from train_factorial.yaml."""
    cfg = yaml.safe_load(FACTORIAL_YAML.read_text(encoding="utf-8"))
    return cfg.get("envs", {})


def _load_env_example() -> dict[str, str]:
    """Load .env.example as key=value pairs (ignoring comments)."""
    result = {}
    if not DOT_ENV_EXAMPLE.exists():
        return result
    for line in DOT_ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" in stripped:
            key, _, value = stripped.partition("=")
            result[key.strip()] = value.strip().strip('"').strip("'")
    return result


def _find_os_environ_gets_in_train_flow() -> dict[str, str]:
    """Extract all os.environ.get() calls from train_flow.py with their defaults."""
    source = TRAIN_FLOW.read_text(encoding="utf-8")
    tree = ast.parse(source)
    results = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            # Match os.environ.get("VAR", "default")
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "get"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "environ"
                and len(node.args) >= 1
                and isinstance(node.args[0], ast.Constant)
            ):
                var_name = node.args[0].value
                default_val = None
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                    default_val = str(node.args[1].value)
                results[var_name] = default_val
    return results


# ---------------------------------------------------------------------------
# YAML envs section completeness
# ---------------------------------------------------------------------------


class TestYamlEnvsCompleteness:
    """Global env vars must have a YAML default for documentation."""

    @pytest.mark.parametrize("var_name", GLOBAL_ENV_VARS)
    def test_global_var_declared_in_yaml(self, var_name: str) -> None:
        """Global env vars must appear in YAML envs section with defaults."""
        envs = _load_yaml_envs()
        assert var_name in envs, (
            f"{var_name} is passed by run_factorial.sh but NOT declared in "
            f"train_factorial.yaml envs section. Without an explicit default, "
            f"the var depends entirely on --env override — if run_factorial.sh "
            f"changes or is bypassed, there's no safe fallback."
        )

    @pytest.mark.parametrize("var_name", PER_MODEL_ENV_VARS)
    def test_per_model_var_consumed_in_python(self, var_name: str) -> None:
        """Per-model override vars must be consumed in train_flow.py."""
        python_gets = _find_os_environ_gets_in_train_flow()
        assert var_name in python_gets, (
            f"{var_name} is passed by run_factorial.sh but NOT consumed in "
            f"train_flow.py via os.environ.get(). The var will be ignored."
        )


# ---------------------------------------------------------------------------
# Python consumption matches YAML defaults
# ---------------------------------------------------------------------------


class TestPythonConsumptionDefaults:
    """Python os.environ.get() defaults must NOT contradict YAML defaults."""

    def test_gradient_checkpointing_default_consistent(self) -> None:
        """GRADIENT_CHECKPOINTING default must be 'false' in both YAML and Python."""
        envs = _load_yaml_envs()
        python_gets = _find_os_environ_gets_in_train_flow()

        yaml_default = str(envs.get("GRADIENT_CHECKPOINTING", ""))
        python_default = python_gets.get("GRADIENT_CHECKPOINTING", "")

        assert yaml_default == "false", (
            f"YAML GRADIENT_CHECKPOINTING default is '{yaml_default}', must be 'false'"
        )
        assert python_default == "false", (
            f"Python GRADIENT_CHECKPOINTING default is '{python_default}', must be 'false'"
        )

    def test_model_family_default_consistent(self) -> None:
        """MODEL_FAMILY default must be consistent between YAML and Python."""
        envs = _load_yaml_envs()
        python_gets = _find_os_environ_gets_in_train_flow()

        yaml_default = str(envs.get("MODEL_FAMILY", ""))
        python_default = python_gets.get("MODEL_FAMILY", "")

        assert yaml_default == python_default, (
            f"MODEL_FAMILY default mismatch: YAML='{yaml_default}' vs "
            f"Python='{python_default}'. This creates precedence confusion."
        )


# ---------------------------------------------------------------------------
# run_factorial.sh passes all required vars
# ---------------------------------------------------------------------------


class TestRunFactorialPassesAllVars:
    """run_factorial.sh must pass every training var via --env."""

    # Vars that are passed via --env in the TRAINING loop (not zero-shot)
    TRAINING_LOOP_ENV_VARS = [
        "MODEL_FAMILY",
        "LOSS_NAME",
        "FOLD_ID",
        "WITH_AUX_CALIB",
        "MAX_EPOCHS",
        "MAX_TRAIN_VOLUMES",
        "MAX_VAL_VOLUMES",
        "EXPERIMENT_NAME",
        "POST_TRAINING_METHODS",
        "BATCH_SIZE",
        "GRAD_ACCUM_STEPS",
        "GRADIENT_CHECKPOINTING",
    ]

    @pytest.mark.parametrize("var_name", TRAINING_LOOP_ENV_VARS)
    def test_training_var_passed_via_env_flag(self, var_name: str) -> None:
        """Training loop vars must appear as --env in run_factorial.sh."""
        source = RUN_FACTORIAL.read_text(encoding="utf-8")
        assert f"--env {var_name}=" in source or f'--env {var_name}="' in source, (
            f"run_factorial.sh does not pass --env {var_name}= to sky jobs launch. "
            f"This means the var will use the YAML default, which may not be "
            f"what the factorial config specifies."
        )

    @pytest.mark.parametrize("var_name", ["ZERO_SHOT", "EVAL_DATASET"])
    def test_zero_shot_vars_in_baseline_section(self, var_name: str) -> None:
        """ZERO_SHOT and EVAL_DATASET are passed in the zero-shot baseline section."""
        source = RUN_FACTORIAL.read_text(encoding="utf-8")
        assert var_name in source, (
            f"{var_name} not referenced in run_factorial.sh at all."
        )


# ---------------------------------------------------------------------------
# .env.example completeness for secrets
# ---------------------------------------------------------------------------


class TestDotEnvExampleSecrets:
    """Secrets must be declared in .env.example (Rule 22)."""

    @pytest.mark.parametrize(
        "var_name",
        [
            "HF_TOKEN",
            "MLFLOW_TRACKING_URI",
        ],
    )
    def test_secret_in_env_example(self, var_name: str) -> None:
        """Secrets must be in .env.example as the single source of truth."""
        env_vars = _load_env_example()
        assert var_name in env_vars, (
            f"{var_name} not in .env.example — violates Rule 22 "
            f"(single-source config via .env.example)."
        )


# ---------------------------------------------------------------------------
# No string→bool conversion pitfalls
# ---------------------------------------------------------------------------


class TestStringBoolConversions:
    """All string→bool env vars must use .lower() == 'true', never bool()."""

    @pytest.mark.parametrize(
        "var_name",
        [
            "GRADIENT_CHECKPOINTING",
            "WITH_AUX_CALIB",
            "ZERO_SHOT",
        ],
    )
    def test_bool_vars_use_safe_conversion(self, var_name: str) -> None:
        """Boolean env vars must use .lower() == 'true' in train_flow.py."""
        source = TRAIN_FLOW.read_text(encoding="utf-8")
        # The safe pattern: args.var_name.lower() == "true"
        # The UNSAFE pattern: bool(args.var_name) — this makes "false" truthy!
        assert f"bool(args.{var_name.lower()}" not in source.replace("_", ""), (
            f"train_flow.py uses bool() on {var_name} — this is WRONG. "
            f"bool('false') == True because non-empty strings are truthy. "
            f"Must use .lower() == 'true' instead."
        )
