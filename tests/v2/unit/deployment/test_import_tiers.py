"""Tests for import chain tier isolation.

Tier C flows (dashboard, pipeline) must be importable without torch.
This requires that tracking.py does NOT have a top-level import of
minivess.serving.model_logger (which transitively pulls in torch).

Rule #16: No regex. Use ast module for Python source analysis.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent.parent
TRACKING_PY = ROOT / "src" / "minivess" / "observability" / "tracking.py"


class TestTrackingImportChain:
    """tracking.py must not pull torch at module level."""

    def test_no_top_level_model_logger_import(self) -> None:
        """model_logger import must be lazy (inside a function), not at module level."""
        tree = ast.parse(TRACKING_PY.read_text(encoding="utf-8"))

        # Check only top-level statements (not inside functions/classes)
        for node in ast.iter_child_nodes(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and "model_logger" in node.module
            ):
                imported_names = [alias.name for alias in node.names]
                msg = (
                    f"tracking.py has top-level import from {node.module}: "
                    f"{imported_names}. This pulls torch transitively. "
                    f"Move inside the function that uses it (log_model method)."
                )
                raise AssertionError(msg)

    def test_resolve_tracking_uri_exists(self) -> None:
        """resolve_tracking_uri must still be defined in tracking.py."""
        tree = ast.parse(TRACKING_PY.read_text(encoding="utf-8"))
        func_names = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        ]
        assert "resolve_tracking_uri" in func_names, (
            "resolve_tracking_uri not found in tracking.py"
        )
