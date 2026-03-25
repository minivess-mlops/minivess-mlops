"""Metric key consistency — Phase 3 of 6th pass post-run fix plan.

Ensures all eval/test metric keys flow through MetricKeys constants,
preventing the eval_fold2_dsc vs eval/fold2/dsc class of bugs.

Plan: run-debug-factorial-experiment-report-6th-pass-post-run-fix.xml v2
"""

from __future__ import annotations

import ast
from pathlib import Path

# ---------------------------------------------------------------------------
# Task 3.1: MetricKeys must have eval/test prefix constants
# ---------------------------------------------------------------------------


class TestMetricKeysHasEvalPrefixes:
    """MetricKeys must define constants for eval and test metric key prefixes."""

    def test_metric_keys_has_eval_fold_prefix(self) -> None:
        """MetricKeys must have EVAL_FOLD_PREFIX constant."""
        from minivess.observability.metric_keys import MetricKeys

        assert hasattr(MetricKeys, "EVAL_FOLD_PREFIX"), (
            "MetricKeys missing EVAL_FOLD_PREFIX constant. "
            "Needed to prevent drift between tracking.py and builder.py."
        )
        assert MetricKeys.EVAL_FOLD_PREFIX == "eval/fold", (
            f"EVAL_FOLD_PREFIX should be 'eval/fold', got '{MetricKeys.EVAL_FOLD_PREFIX}'"
        )

    def test_metric_keys_has_eval_test_prefix(self) -> None:
        """MetricKeys must have EVAL_TEST_PREFIX constant."""
        from minivess.observability.metric_keys import MetricKeys

        assert hasattr(MetricKeys, "EVAL_TEST_PREFIX"), (
            "MetricKeys missing EVAL_TEST_PREFIX constant. "
            "Needed for test dataset metric keys (test/deepvess/all/dsc)."
        )
        assert MetricKeys.EVAL_TEST_PREFIX == "test", (
            f"EVAL_TEST_PREFIX should be 'test', got '{MetricKeys.EVAL_TEST_PREFIX}'"
        )

    def test_metric_keys_has_eval_prefix(self) -> None:
        """MetricKeys must have EVAL_PREFIX constant for non-fold eval metrics."""
        from minivess.observability.metric_keys import MetricKeys

        assert hasattr(MetricKeys, "EVAL_PREFIX"), (
            "MetricKeys missing EVAL_PREFIX constant."
        )
        assert MetricKeys.EVAL_PREFIX == "eval", (
            f"EVAL_PREFIX should be 'eval', got '{MetricKeys.EVAL_PREFIX}'"
        )


# ---------------------------------------------------------------------------
# Task 3.2: No raw eval/ strings in key modules
# ---------------------------------------------------------------------------

# Modules that MUST use MetricKeys constants instead of raw strings
_KEY_MODULES = [
    Path("src/minivess/ensemble/builder.py"),
    Path("src/minivess/observability/tracking.py"),
]


class TestNoRawEvalStringsInKeyModules:
    """Key modules must use MetricKeys constants, not raw eval/ string literals."""

    def test_builder_uses_metric_keys_import(self) -> None:
        """builder.py must import from metric_keys module."""
        source = Path("src/minivess/ensemble/builder.py").read_text(encoding="utf-8")
        assert "MetricKeys" in source or "metric_keys" in source, (
            "builder.py must import MetricKeys constants instead of "
            "using raw 'eval/fold2/dsc' string literals."
        )

    def test_tracking_uses_metric_keys_import(self) -> None:
        """tracking.py must import from metric_keys module."""
        source = Path("src/minivess/observability/tracking.py").read_text(
            encoding="utf-8"
        )
        assert "MetricKeys" in source or "metric_keys" in source, (
            "tracking.py must import MetricKeys constants instead of "
            "using raw 'eval/' f-string literals."
        )

    def test_no_hardcoded_eval_fold2_dsc_anywhere(self) -> None:
        """No module should contain the literal 'eval/fold2/dsc' — must use MetricKeys."""
        for module_path in _KEY_MODULES:
            source = module_path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and node.value == "eval/fold2/dsc"
                ):
                    raise AssertionError(
                        f"{module_path}: contains hardcoded 'eval/fold2/dsc' "
                        "string literal. Must use MetricKeys.EVAL_FOLD_PREFIX "
                        "to construct the key dynamically."
                    )
