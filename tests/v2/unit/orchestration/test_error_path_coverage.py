"""Error path coverage — Phase 5 of 6th pass post-run fix plan.

Tests exercising exception handling paths identified by reviewer agents.
All tests in ONE file to avoid tiny single-test files.

Plan: run-debug-factorial-experiment-report-6th-pass-post-run-fix.xml v2
"""

from __future__ import annotations

import ast
from pathlib import Path

TRAIN_FLOW = Path("src/minivess/orchestration/flows/train_flow.py")


# ---------------------------------------------------------------------------
# Task 5.1: No bare except: pass in train_flow.py
# ---------------------------------------------------------------------------


class TestNoBarExceptPass:
    """train_flow.py must NOT have bare 'except: pass' or 'except Exception: pass'."""

    def test_no_except_pass_in_train_flow(self) -> None:
        """Every except block must log the exception, not silently discard it."""
        source = TRAIN_FLOW.read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ExceptHandler)
                and len(node.body) == 1
                and isinstance(node.body[0], ast.Pass)
            ):
                raise AssertionError(
                    f"train_flow.py line {node.lineno}: bare 'except: pass' "
                    "silently discards exceptions. Must log at WARNING level "
                    "with exc_info=True. Rule #25: Loud failures, never silent."
                )

    def test_setup_timing_except_logs_warning(self) -> None:
        """The except block around parse_setup_timing must log, not pass."""
        source = TRAIN_FLOW.read_text(encoding="utf-8")
        tree = ast.parse(source)

        # Find the try block that contains "parse_setup_timing" call
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            # Check if this try block contains a call to parse_setup_timing
            has_timing_call = False
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Name)
                    and child.func.id == "parse_setup_timing"
                ):
                    has_timing_call = True
                    break
            if not has_timing_call:
                continue

            # Found the right try block — check except handlers
            for handler in node.handlers:
                body_has_logging = False
                for stmt in handler.body:
                    # Check for logger.warning(...) call
                    if (
                        isinstance(stmt, ast.Expr)
                        and isinstance(stmt.value, ast.Call)
                        and isinstance(stmt.value.func, ast.Attribute)
                        and stmt.value.func.attr == "warning"
                    ):
                        body_has_logging = True
                        break
                assert body_has_logging, (
                    f"train_flow.py line {handler.lineno}: except block around "
                    "parse_setup_timing must call logger.warning(), not silently pass."
                )
                return  # Found and validated

        raise AssertionError(
            "Could not find try/except block around parse_setup_timing"
        )
