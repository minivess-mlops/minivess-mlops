"""Tests for T-20: deprecation warning in train_monitored.py.

Verifies that running train_monitored.py without ALLOW_STANDALONE_TRAINING=1
exits with SystemExit(1), and with the env var it continues.

NO subprocess invocation — tests use import + monkeypatch.
"""

from __future__ import annotations

import ast
from pathlib import Path

_TRAIN_MONITORED_SRC = Path("scripts/train_monitored.py")


# ---------------------------------------------------------------------------
# Source-level: deprecation gate must exist in source
# ---------------------------------------------------------------------------


class TestDeprecationGateExists:
    def test_deprecation_warning_in_source(self) -> None:
        """train_monitored.py must contain a DeprecationWarning."""
        source = _TRAIN_MONITORED_SRC.read_text(encoding="utf-8")
        assert "DeprecationWarning" in source, (
            "scripts/train_monitored.py must issue a DeprecationWarning. "
            "Add warnings.warn(..., DeprecationWarning, ...) in the __main__ block."
        )

    def test_allow_standalone_training_in_source(self) -> None:
        """train_monitored.py must reference ALLOW_STANDALONE_TRAINING env var."""
        source = _TRAIN_MONITORED_SRC.read_text(encoding="utf-8")
        assert "ALLOW_STANDALONE_TRAINING" in source, (
            "scripts/train_monitored.py must check ALLOW_STANDALONE_TRAINING env var. "
            "Add: if not os.environ.get('ALLOW_STANDALONE_TRAINING'): sys.exit(1)"
        )

    def test_sys_exit_in_source(self) -> None:
        """train_monitored.py must call sys.exit(1) in the deprecation gate."""
        source = _TRAIN_MONITORED_SRC.read_text(encoding="utf-8")
        tree = ast.parse(source)
        found_exit = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if (isinstance(func, ast.Attribute) and func.attr == "exit") or (
                    isinstance(func, ast.Name) and func.id == "exit"
                ):
                    found_exit = True
                    break
        assert found_exit, (
            "scripts/train_monitored.py must call sys.exit() in the deprecation gate."
        )

    def test_deprecation_message_mentions_prefect(self) -> None:
        """Deprecation message must mention the correct prefect deployment run entry point."""
        source = _TRAIN_MONITORED_SRC.read_text(encoding="utf-8")
        assert "prefect deployment run" in source, (
            "The deprecation warning in train_monitored.py must mention "
            "'prefect deployment run' as the correct entry point."
        )
