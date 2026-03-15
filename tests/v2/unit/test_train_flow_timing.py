"""Tests for infrastructure timing integration in train_flow.py and trainer.py.

Verifies that train_flow.py calls log_infrastructure_timing() and
log_cost_analysis(), and that trainer.py records first/steady epoch timing.

Uses ast.parse + ast.walk (Rule #16: no regex for structured data).

Issue: #683
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TRAIN_FLOW = (
    _REPO_ROOT / "src" / "minivess" / "orchestration" / "flows" / "train_flow.py"
)
_TRAINER = _REPO_ROOT / "src" / "minivess" / "pipeline" / "trainer.py"


def _get_all_names(source_path: Path) -> set[str]:
    """Extract all Name and Attribute nodes from a Python source file."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def _get_all_string_literals(source_path: Path) -> set[str]:
    """Extract all string literal values from a Python source file."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    strings: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            strings.add(node.value)
    return strings


class TestTrainFlowTimingIntegration:
    """Tests for infrastructure timing calls in train_flow.py."""

    def test_train_flow_calls_infrastructure_timing(self) -> None:
        """train_flow.py references log_infrastructure_timing."""
        names = _get_all_names(_TRAIN_FLOW)
        assert "log_infrastructure_timing" in names

    def test_train_flow_calls_cost_analysis(self) -> None:
        """train_flow.py references log_cost_analysis."""
        names = _get_all_names(_TRAIN_FLOW)
        assert "log_cost_analysis" in names


class TestTrainerEpochTiming:
    """Tests for first/steady epoch timing in trainer.py."""

    def test_trainer_records_first_epoch_time(self) -> None:
        """trainer.py has prof_first_epoch_seconds metric."""
        strings = _get_all_string_literals(_TRAINER)
        assert "prof_first_epoch_seconds" in strings

    def test_trainer_records_steady_epoch_time(self) -> None:
        """trainer.py has prof_steady_epoch_seconds metric."""
        strings = _get_all_string_literals(_TRAINER)
        assert "prof_steady_epoch_seconds" in strings
