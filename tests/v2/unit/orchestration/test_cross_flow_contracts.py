"""Cross-flow contract enforcement — Phase 4 of 6th pass post-run fix plan.

Ensures checkpoint filenames and experiment names are sourced from constants.py,
preventing silent cross-flow contract drift.

Plan: run-debug-factorial-experiment-report-6th-pass-post-run-fix.xml v2
"""

from __future__ import annotations

import ast
from pathlib import Path

CONSTANTS = Path("src/minivess/orchestration/constants.py")
DEPLOY_FLOW = Path("src/minivess/orchestration/flows/deploy_flow.py")
DASHBOARD_FLOW = Path("src/minivess/orchestration/flows/dashboard_flow.py")
TRAIN_FLOW = Path("src/minivess/orchestration/flows/train_flow.py")
POST_TRAINING_FLOW = Path("src/minivess/orchestration/flows/post_training_flow.py")
MAINTENANCE_FLOW = Path("src/minivess/orchestration/flows/maintenance_flow.py")


# ---------------------------------------------------------------------------
# Task 4.1: Canonical checkpoint filename
# ---------------------------------------------------------------------------


class TestCheckpointNaming:
    """Train and post-training flows must use the same checkpoint filename."""

    def test_constants_has_checkpoint_best_filename(self) -> None:
        """constants.py must define CHECKPOINT_BEST_FILENAME."""
        from minivess.orchestration.constants import CHECKPOINT_BEST_FILENAME

        assert CHECKPOINT_BEST_FILENAME == "best_val_loss.pth", (
            f"CHECKPOINT_BEST_FILENAME should be 'best_val_loss.pth', "
            f"got '{CHECKPOINT_BEST_FILENAME}'"
        )

    def test_post_training_no_best_ckpt_fallback(self) -> None:
        """post_training_flow must NOT fall back to 'best.ckpt'."""
        source = POST_TRAINING_FLOW.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == "best.ckpt":
                raise AssertionError(
                    "post_training_flow.py contains 'best.ckpt' fallback. "
                    "Must use CHECKPOINT_BEST_FILENAME from constants.py only."
                )


# ---------------------------------------------------------------------------
# Task 4.2: No hardcoded "minivess_training" in flow files
# ---------------------------------------------------------------------------


# Flow files that should use resolve_experiment_name() instead of hardcoding
_FLOW_FILES = [
    DEPLOY_FLOW,
    DASHBOARD_FLOW,
]


class TestExperimentNaming:
    """Flow files must NOT contain hardcoded 'minivess_training' string literals."""

    def test_no_hardcoded_experiment_name_in_deploy_flow(self) -> None:
        """deploy_flow.py must use constants, not raw 'minivess_training'."""
        source = DEPLOY_FLOW.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == "minivess_training":
                raise AssertionError(
                    "deploy_flow.py contains hardcoded 'minivess_training'. "
                    "Must use resolve_experiment_name(EXPERIMENT_TRAINING) "
                    "from constants.py."
                )

    def test_no_hardcoded_experiment_name_in_dashboard_flow(self) -> None:
        """dashboard_flow.py must use constants, not raw 'minivess_training'."""
        source = DASHBOARD_FLOW.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == "minivess_training":
                raise AssertionError(
                    "dashboard_flow.py contains hardcoded 'minivess_training'. "
                    "Must use resolve_experiment_name(EXPERIMENT_TRAINING) "
                    "from constants.py."
                )

    def test_deploy_flow_imports_constants(self) -> None:
        """deploy_flow.py must import EXPERIMENT_TRAINING or resolve_experiment_name."""
        source = DEPLOY_FLOW.read_text(encoding="utf-8")
        has_import = (
            "EXPERIMENT_TRAINING" in source or "resolve_experiment_name" in source
        )
        assert has_import, (
            "deploy_flow.py must import EXPERIMENT_TRAINING or "
            "resolve_experiment_name from constants.py."
        )

    def test_dashboard_flow_imports_constants(self) -> None:
        """dashboard_flow.py must import EXPERIMENT_TRAINING or resolve_experiment_name."""
        source = DASHBOARD_FLOW.read_text(encoding="utf-8")
        has_import = (
            "EXPERIMENT_TRAINING" in source or "resolve_experiment_name" in source
        )
        assert has_import, (
            "dashboard_flow.py must import EXPERIMENT_TRAINING or "
            "resolve_experiment_name from constants.py."
        )
