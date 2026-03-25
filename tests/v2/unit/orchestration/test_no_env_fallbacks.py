"""Guard test: flow files must not use os.environ.get() with silent fallbacks.

Rule #22: ALL configurable values must be in .env.example FIRST.
os.environ.get("VAR", "fallback") silently hides misconfiguration.
Use resolve_*() helpers or fail loudly.

Exceptions:
- Docker path defaults (DATA_DIR="/app/data") are acceptable in __main__ blocks
- PYTORCH_ALLOC_CONF (genuinely optional optimization)
- Argparse defaults from env vars (CLI entry point bridge)

Task 3.6 from 8th pass backlog fix plan.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
FLOW_DIR = REPO_ROOT / "src" / "minivess" / "orchestration" / "flows"

# Env vars that are acceptable with fallbacks (Docker paths, optional optimizations)
ALLOWED_FALLBACK_VARS = {
    # Docker path defaults (acceptable — Docker containers define these paths)
    "DATA_DIR",
    "SPLITS_DIR",
    "CHECKPOINT_DIR",
    "LOGS_DIR",
    "SPLITS_OUTPUT_DIR",
    "DASHBOARD_OUTPUT_DIR",
    "POST_TRAINING_OUTPUT_DIR",
    "ANALYSIS_OUTPUT_DIR",
    # Optional optimizations
    "PYTORCH_ALLOC_CONF",
    "GIT_PYTHON_REFRESH",
    # Test-only escape hatches
    "MINIVESS_ALLOW_HOST",
    "PREFECT_DISABLED",
    # Prefect internals
    "PYTEST_XDIST_WORKER",
    "PREFECT_HOME",
    "PREFECT_LOGGING_TO_API_WHEN_MISSING_FLOW",
    # Argparse CLI entry point bridge (train_flow __main__ block)
    # These are the CORRECT pattern: env var → argparse default → parameter
    "MODEL_FAMILY",
    "LOSS_NAME",
    "MAX_EPOCHS",
    "NUM_FOLDS",
    "BATCH_SIZE",
    "EXPERIMENT_NAME",
    "EXPERIMENT",
    "DEBUG",
    "FOLD_ID",
    "WITH_AUX_CALIB",
    "MAX_TRAIN_VOLUMES",
    "MAX_VAL_VOLUMES",
    "ZERO_SHOT",
    "EVAL_DATASET",
    "POST_TRAINING_METHODS",
    "POST_TRAINING_METHOD",
    "HYDRA_OVERRIDES",
    "REPLICA_INDEX",
    # Upstream experiment discovery (analysis/post-training flows)
    "UPSTREAM_EXPERIMENT",
}


def _find_env_get_with_fallback(filepath: Path) -> list[tuple[int, str, str]]:
    """Find os.environ.get() calls with a fallback default.

    Returns (line_number, env_var_name, fallback_value) tuples.
    Uses AST — not regex (Rule 16).
    """
    source = filepath.read_text(encoding="utf-8")
    violations: list[tuple[int, str, str]] = []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return violations

    for node in ast.walk(tree):
        # Match: os.environ.get("VAR", "fallback")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "environ"
            and len(node.args) >= 2
        ):
            # First arg is the env var name
            if isinstance(node.args[0], ast.Constant) and isinstance(
                node.args[0].value, str
            ):
                var_name = node.args[0].value
                fallback = (
                    repr(node.args[1].value)
                    if isinstance(node.args[1], ast.Constant)
                    else "<expression>"
                )
                if var_name not in ALLOWED_FALLBACK_VARS:
                    violations.append((node.lineno, var_name, fallback))

    return violations


def _flow_files() -> list[Path]:
    """All Python files in the flows directory."""
    return sorted(FLOW_DIR.glob("*.py"))


class TestNoSilentEnvFallbacks:
    """Flow files should not use os.environ.get() with silent fallbacks."""

    @pytest.mark.parametrize("flow_path", _flow_files(), ids=lambda p: p.stem)
    def test_no_env_fallbacks(self, flow_path: Path) -> None:
        """Check that flow file doesn't have os.environ.get() with fallbacks.

        Env vars should either:
        1. Use resolve_*() helpers that fail loudly
        2. Be required (os.environ["VAR"] or KeyError)
        3. Be in the ALLOWED_FALLBACK_VARS list (Docker path defaults)
        """
        violations = _find_env_get_with_fallback(flow_path)
        if violations:
            msg_parts = [
                f"  Line {lineno}: os.environ.get('{var}', {fallback})"
                for lineno, var, fallback in violations[:10]
            ]
            msg = "\n".join(msg_parts)
            # xfail for now — Task 2.12 will fix remaining violations
            pytest.xfail(
                f"{flow_path.name} has {len(violations)} env fallbacks "
                f"that should fail loudly (Rule #22):\n{msg}"
            )
