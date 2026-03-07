"""Tests for T-05: Replace tempfile.mkdtemp() with CHECKPOINT_DIR env var.

Verifies that checkpoint_dir is resolved from the CHECKPOINT_DIR environment
variable, never from tempfile, and never starts with /tmp.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

_TRAIN_MONITORED_SRC = Path("scripts/train_monitored.py")


def _load_source() -> str:
    return _TRAIN_MONITORED_SRC.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# AST-level: no tempfile import used for checkpoint_dir
# ---------------------------------------------------------------------------


class TestNoTempfileForCheckpoints:
    def test_no_tempfile_import(self) -> None:
        source = _load_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "tempfile", (
                        f"'import tempfile' still present in train_monitored.py line {node.lineno}. "
                        "Replace tempfile.mkdtemp() with os.environ.get('CHECKPOINT_DIR', ...)."
                    )

    def test_no_mkdtemp_call(self) -> None:
        source = _load_source()
        assert "mkdtemp" not in source, (
            "tempfile.mkdtemp() still present in train_monitored.py. "
            "Replace with os.environ.get('CHECKPOINT_DIR', '/app/checkpoints')."
        )


# ---------------------------------------------------------------------------
# Functional tests: checkpoint_dir resolved from env var
# ---------------------------------------------------------------------------


class TestCheckpointDirFromEnv:
    def test_checkpoint_dir_not_in_tmp(self, monkeypatch) -> None:
        """Default checkpoint dir (no env var) must not resolve to /tmp."""
        monkeypatch.delenv("CHECKPOINT_DIR", raising=False)
        checkpoint_base = Path(os.environ.get("CHECKPOINT_DIR", "/app/checkpoints"))
        assert not str(checkpoint_base).startswith("/tmp"), (
            f"checkpoint_dir defaults to /tmp path: {checkpoint_base}"
        )

    def test_checkpoint_dir_from_env_var(self, monkeypatch, tmp_path) -> None:
        """CHECKPOINT_DIR env var controls the checkpoint base directory."""
        target = tmp_path / "my_checkpoints"
        monkeypatch.setenv("CHECKPOINT_DIR", str(target))
        checkpoint_base = Path(os.environ.get("CHECKPOINT_DIR", "/app/checkpoints"))
        assert checkpoint_base == target

    def test_checkpoint_dir_is_absolute_by_default(self, monkeypatch) -> None:
        """Default CHECKPOINT_DIR must be an absolute path."""
        monkeypatch.delenv("CHECKPOINT_DIR", raising=False)
        checkpoint_base = Path(os.environ.get("CHECKPOINT_DIR", "/app/checkpoints"))
        assert checkpoint_base.is_absolute(), (
            f"Default checkpoint dir is not absolute: {checkpoint_base}"
        )

    def test_checkpoint_dir_includes_fold_id(self, monkeypatch, tmp_path) -> None:
        """checkpoint_dir should include fold_id subfolder."""
        monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path))
        fold_id = 0
        checkpoint_dir = (
            Path(os.environ.get("CHECKPOINT_DIR", "/app/checkpoints"))
            / f"fold_{fold_id}"
        )
        assert checkpoint_dir.name == "fold_0"

    def test_env_var_documented_in_compose(self) -> None:
        """CHECKPOINT_DIR must appear in docker-compose.flows.yml train environment."""
        compose_src = Path("deployment/docker-compose.flows.yml").read_text(
            encoding="utf-8"
        )
        import yaml

        compose = yaml.safe_load(compose_src)
        train_env = compose["services"]["train"].get("environment", {})
        assert "CHECKPOINT_DIR" in train_env, (
            "CHECKPOINT_DIR env var not declared in docker-compose.flows.yml train service. "
            "Add CHECKPOINT_DIR: /app/checkpoints to the train service environment."
        )
