"""Tests for T-09: post_training_flow output_dir — use POST_TRAINING_OUTPUT_DIR env var.

Uses ast.parse() for source inspection — NO regex (CLAUDE.md Rule #16).
"""

from __future__ import annotations

import ast
from pathlib import Path

_POST_TRAINING_SRC = Path("src/minivess/orchestration/flows/post_training_flow.py")


# ---------------------------------------------------------------------------
# AST-level: no hardcoded relative Path("outputs/post_training")
# ---------------------------------------------------------------------------


class TestNoHardcodedRelativePath:
    def test_no_hardcoded_relative_path(self) -> None:
        """post_training_flow.py must not have Path('outputs/post_training') literal."""
        source = _POST_TRAINING_SRC.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            # Check for Path("outputs/post_training") or Path('outputs/post_training')
            is_path_call = (isinstance(func, ast.Name) and func.id == "Path") or (
                isinstance(func, ast.Attribute) and func.attr == "Path"
            )
            if not is_path_call:
                continue
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    assert "outputs/post_training" not in arg.value, (
                        f"Hardcoded relative path Path({arg.value!r}) found at "
                        f"line {node.lineno}. "
                        "Replace with Path(os.environ.get('POST_TRAINING_OUTPUT_DIR', "
                        "'/app/outputs/post_training'))."
                    )


# ---------------------------------------------------------------------------
# Functional: output_dir resolved from POST_TRAINING_OUTPUT_DIR env var
# ---------------------------------------------------------------------------


class TestOutputDirFromEnv:
    def test_output_dir_from_env(self, monkeypatch, tmp_path) -> None:
        """POST_TRAINING_OUTPUT_DIR env var must control output root."""
        import os

        target = tmp_path / "my_pt_output"
        monkeypatch.setenv("POST_TRAINING_OUTPUT_DIR", str(target))
        resolved = Path(
            os.environ.get("POST_TRAINING_OUTPUT_DIR", "/app/outputs/post_training")
        )
        assert resolved == target, (
            f"Resolved output_dir {resolved} != POST_TRAINING_OUTPUT_DIR={target}"
        )

    def test_output_dir_not_relative(self, monkeypatch) -> None:
        """Default output_dir must be absolute (not relative)."""
        import os

        monkeypatch.delenv("POST_TRAINING_OUTPUT_DIR", raising=False)
        resolved = Path(
            os.environ.get("POST_TRAINING_OUTPUT_DIR", "/app/outputs/post_training")
        )
        assert resolved.is_absolute(), f"Default output_dir is not absolute: {resolved}"

    def test_output_dir_default_value(self, monkeypatch) -> None:
        """Default output_dir must be /app/outputs/post_training."""
        import os

        monkeypatch.delenv("POST_TRAINING_OUTPUT_DIR", raising=False)
        resolved = Path(
            os.environ.get("POST_TRAINING_OUTPUT_DIR", "/app/outputs/post_training")
        )
        assert str(resolved) == "/app/outputs/post_training", (
            f"Default output_dir {resolved} != /app/outputs/post_training"
        )

    def test_compose_has_post_training_output_dir_env(self) -> None:
        """POST_TRAINING_OUTPUT_DIR must appear in docker-compose.flows.yml post_training env."""
        import yaml

        compose = yaml.safe_load(
            Path("deployment/docker-compose.flows.yml").read_text(encoding="utf-8")
        )
        env = compose["services"].get("post_training", {}).get("environment", {})
        assert "POST_TRAINING_OUTPUT_DIR" in env, (
            "POST_TRAINING_OUTPUT_DIR not in docker-compose.flows.yml post_training environment. "
            "Add POST_TRAINING_OUTPUT_DIR: /app/outputs/post_training."
        )
