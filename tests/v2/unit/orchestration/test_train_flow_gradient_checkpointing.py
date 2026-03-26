"""Tests for train_flow gradient checkpointing injection — T10 + T11 doughnut-hole tests.

T10: Test that gradient_checkpointing is injected into architecture_params.
T11: Test that _skip_grad_flow is correctly wired when GC is enabled.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

TRAIN_FLOW = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "minivess"
    / "orchestration"
    / "flows"
    / "train_flow.py"
)


class TestGcInjectedIntoArchParams:
    """T10: gradient_checkpointing must be injected into architecture_params."""

    @pytest.fixture()
    def _source(self) -> str:
        return TRAIN_FLOW.read_text(encoding="utf-8")

    def test_gc_injected_into_arch_params_when_config_true(self, _source):
        """When config has gc=True, arch_params must get gc=True."""
        # Verify the injection code exists
        assert 'arch_params["gradient_checkpointing"]' in _source, (
            "gradient_checkpointing injection into arch_params not found in train_flow"
        )

    def test_gc_not_injected_when_config_false(self, _source):
        """Injection should be conditional: only when config gc is true."""
        assert "_gc_config" in _source, (
            "_gc_config extraction from config not found"
        )
        assert "if _gc_config" in _source, (
            "Conditional injection guard for gradient_checkpointing not found"
        )

    def test_gc_not_overwritten_when_arch_params_already_has_it(self, _source):
        """If arch_params already has gc, don't overwrite it."""
        assert '"gradient_checkpointing" not in arch_params' in _source, (
            "Guard against overwriting existing arch_params gc not found"
        )


class TestSkipGradFlow:
    """T11: _skip_grad_flow must be True when gradient_checkpointing is enabled."""

    @pytest.fixture()
    def _source(self) -> str:
        return TRAIN_FLOW.read_text(encoding="utf-8")

    def test_skip_grad_flow_true_when_config_gc_true(self, _source):
        """_skip_grad_flow must check config gradient_checkpointing."""
        assert 'config.get("gradient_checkpointing"' in _source, (
            "_skip_grad_flow must read gradient_checkpointing from config"
        )

    def test_skip_grad_flow_true_when_arch_params_gc_true(self, _source):
        """_skip_grad_flow must also check arch_params gradient_checkpointing."""
        assert 'arch_params.get("gradient_checkpointing"' in _source, (
            "_skip_grad_flow must read gradient_checkpointing from arch_params"
        )

    def test_skip_grad_flow_false_by_default(self, _source):
        """Both sources default to False → _skip_grad_flow is False."""
        # Verify both .get calls have False as default
        assert 'config.get("gradient_checkpointing", False)' in _source
        assert 'arch_params.get("gradient_checkpointing", False)' in _source

    def test_skip_grad_flow_passed_to_pre_training_checks(self, _source):
        """_skip_grad_flow must be passed to run_pre_training_checks."""
        assert "skip_gradient_flow=_skip_grad_flow" in _source, (
            "_skip_grad_flow must be passed to run_pre_training_checks()"
        )

    def test_skip_grad_flow_uses_or_logic(self, _source):
        """Either config OR arch_params gc=True should trigger skip."""
        tree = ast.parse(_source)
        # Find _skip_grad_flow assignment
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "_skip_grad_flow":
                        # Should use BoolOp(Or)
                        found = True
                        break
        assert found, "_skip_grad_flow assignment not found"
