"""Tests for sentinel string removal (T-10, closes #411).

TDD RED phase: "no_upstream" sentinel strings must be replaced with None.
"""

from __future__ import annotations

import ast
from pathlib import Path

from minivess.orchestration.mlflow_helpers import find_upstream_safely

FLOW_FILES = [
    Path("src/minivess/orchestration/flows/train_flow.py"),
    Path("src/minivess/orchestration/flows/post_training_flow.py"),
    Path("src/minivess/orchestration/flows/analysis_flow.py"),
    Path("src/minivess/orchestration/flows/deploy_flow.py"),
]


class TestNoSentinelStrings:
    def test_no_sentinel_strings_in_flows(self) -> None:
        """No flow file may contain the literal string 'no_upstream'."""
        violations: list[str] = []
        for fpath in FLOW_FILES:
            if not fpath.exists():
                continue
            tree = ast.parse(fpath.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and node.value == "no_upstream":
                    violations.append(f"{fpath.name}: found literal 'no_upstream'")
        assert not violations, "Sentinel string 'no_upstream' found:\n" + "\n".join(
            f"  {v}" for v in violations
        )


class TestFindUpstreamSafelyReturnsNone:
    def test_upstream_returns_none_on_failure(self) -> None:
        """find_upstream_safely() returns None when FlowContract raises (not sentinel)."""
        from unittest.mock import patch

        with patch(
            "minivess.orchestration.mlflow_helpers.FlowContract.find_upstream_run",
            side_effect=Exception("mlflow unavailable"),
        ):
            result = find_upstream_safely(
                tracking_uri="mlruns",
                experiment_name="test",
                upstream_flow="test-flow",
            )
        assert result is None
        assert result != "no_upstream"
