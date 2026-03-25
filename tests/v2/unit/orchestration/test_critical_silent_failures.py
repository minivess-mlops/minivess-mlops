"""Critical silent failure detection — Phase 1 of 6th pass post-run fix plan.

4 CRITICAL bugs that cause production data loss. Each hides behind
except Exception + continue, making training "succeed" with garbage results.

Plan: run-debug-factorial-experiment-report-6th-pass-post-run-fix.xml v2
"""

from __future__ import annotations

import ast
from pathlib import Path

POST_TRAINING_FLOW = Path("src/minivess/orchestration/flows/post_training_flow.py")
ANALYSIS_FLOW = Path("src/minivess/orchestration/flows/analysis_flow.py")
TRAIN_FLOW = Path("src/minivess/orchestration/flows/train_flow.py")
BUILDER = Path("src/minivess/ensemble/builder.py")


# ---------------------------------------------------------------------------
# Task 1.1: Post-training plugin failure must NOT return empty model_paths
# ---------------------------------------------------------------------------


class TestPostTrainingPluginErrorPropagation:
    """Plugin failures must raise, not silently return status=error."""

    def test_plugin_except_block_does_not_return_empty_model_paths(self) -> None:
        """The except block at line ~334 must NOT produce {model_paths: []}."""
        source = POST_TRAINING_FLOW.read_text(encoding="utf-8")
        # The dangerous pattern: except → dict with model_paths: []
        # After the fix, the except block should re-raise or log at ERROR level
        # and NOT return a dict that callers treat as success.
        danger_count = source.count('"model_paths": []')
        assert danger_count == 0, (
            f"post_training_flow.py has {danger_count} occurrences of "
            f'"model_paths": [] in except blocks. Plugin failures must raise, '
            f"not return empty model_paths (SWAG silently fails otherwise)."
        )

    def test_plugin_failure_logged_at_error_not_warning(self) -> None:
        """Plugin failures must use log.error() or log.exception(), not log.warning()."""
        source = POST_TRAINING_FLOW.read_text(encoding="utf-8")
        # log.exception already logs at ERROR level — that's acceptable
        # But if it's changed to log.warning, that's wrong
        assert "log.exception" in source or "logger.exception" in source, (
            "Plugin failure logging should use .exception() (ERROR level)"
        )


# ---------------------------------------------------------------------------
# Task 1.2: External test DataLoader failure must raise, not fall back
# ---------------------------------------------------------------------------


class TestExternalDataLoaderFailure:
    """DataLoader construction failure must NOT fall back to raw pairs."""

    def test_no_raw_pairs_fallback_in_analysis_flow(self) -> None:
        """analysis_flow must NOT have except → result[ds_name] = {"all": pairs}."""
        source = ANALYSIS_FLOW.read_text(encoding="utf-8")
        # The dangerous pattern: catching DataLoader failure and storing raw pairs
        # This produces metrics on unprocessed data (wrong intensity scale)
        assert '{"all": pairs}' not in source and "{'all': pairs}" not in source, (
            "analysis_flow.py falls back to raw pairs on DataLoader failure. "
            "This silently degrades evaluation quality. Must raise instead."
        )


# ---------------------------------------------------------------------------
# Task 1.3: No hardcoded "mlruns" fallback in tracking URI resolution
# ---------------------------------------------------------------------------


class TestNoHardcodedTrackingUriFallback:
    """train_flow must NOT have config.get("tracking_uri", "mlruns")."""

    def test_no_mlruns_default_in_train_flow(self) -> None:
        """No string literal "mlruns" as a default value in train_flow.py."""
        source = TRAIN_FLOW.read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            # Look for config.get("tracking_uri", "mlruns") pattern
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and len(node.args) >= 2
            ):
                # Check if second arg is "mlruns"
                second_arg = node.args[1]
                if (
                    isinstance(second_arg, ast.Constant)
                    and second_arg.value == "mlruns"
                ):
                    first_arg = node.args[0]
                    if (
                        isinstance(first_arg, ast.Constant)
                        and "tracking" in str(first_arg.value).lower()
                    ):
                        key_name = str(first_arg.value)
                        raise AssertionError(
                            f'Found config.get("{key_name}", "mlruns") — '
                            "hardcoded fallback violates Rule #22. "
                            "Use resolve_tracking_uri() instead."
                        )


# ---------------------------------------------------------------------------
# Task 1.4: builder.py eval_fold2_dsc vs eval/fold2/dsc format mismatch
# ---------------------------------------------------------------------------


class TestBuilderMetricFormatMatchesTracking:
    """builder.py must query metrics in the SAME format tracking.py logs them."""

    def test_builder_uses_slash_format_not_underscore(self) -> None:
        """builder.py must NOT check for "eval_fold2_dsc" (underscore format)."""
        source = BUILDER.read_text(encoding="utf-8")
        # The dangerous pattern: underscore format that never matches slash format
        assert "eval_fold2_dsc" not in source, (
            'builder.py checks for "eval_fold2_dsc" but tracking.py logs '
            '"eval/fold2/dsc" (slash format). The builder NEVER finds completed '
            "runs — silently returns empty ensemble. Use slash format."
        )

    def test_builder_uses_slash_separator_for_eval_metrics(self) -> None:
        """Eval metric references in builder.py must use / separator, not _."""
        source = BUILDER.read_text(encoding="utf-8")
        # Any remaining underscore-format eval metrics
        for line_num, line in enumerate(source.splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            # Check for patterns like eval_fold0_dsc, eval_fold1_cldice
            if (
                "eval_fold" in line
                and ("_dsc" in line or "_cldice" in line)
                and ('"eval_fold' in line or "'eval_fold" in line)
            ):
                raise AssertionError(
                    f"builder.py line {line_num}: uses underscore eval metric "
                    f"format. Must use slash format (eval/fold0/dsc). "
                    f"Line: {line.strip()}"
                )
