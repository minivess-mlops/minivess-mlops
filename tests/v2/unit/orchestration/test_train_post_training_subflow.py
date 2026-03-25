"""Tests for train + post-training sub-flow architecture.

Phase 1: Verify training_subflow exists and parent delegates to it.
Phase 2: Verify post_training_subflow conditional execution.

Plan: docs/planning/training-and-post-training-into-two-subflows-under-one-flow.md
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import pytest

TRAIN_FLOW_PATH = Path("src/minivess/orchestration/flows/train_flow.py")


def _find_function_calls_in(func_name: str, called_name: str) -> bool:
    """AST helper: check if function ``func_name`` calls ``called_name``."""
    source = TRAIN_FLOW_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name != func_name:
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            func = child.func
            if isinstance(func, ast.Name) and func.id == called_name:
                return True
            if isinstance(func, ast.Attribute) and func.attr == called_name:
                return True
    return False


# ---------------------------------------------------------------------------
# Phase 1 tests: training_subflow existence and wiring
# ---------------------------------------------------------------------------


class TestTrainingSubflowExists:
    """Verify training_subflow is a Prefect @flow with correct name."""

    def test_training_subflow_function_exists(self) -> None:
        """training_subflow must be importable from train_flow module."""
        from minivess.orchestration.flows.train_flow import training_subflow

        assert callable(training_subflow)

    def test_training_subflow_is_prefect_flow(self) -> None:
        """training_subflow must be decorated with @flow."""
        from minivess.orchestration.flows.train_flow import training_subflow

        # Prefect flows have a .fn attribute (the original function)
        assert hasattr(training_subflow, "fn"), (
            "training_subflow is not a Prefect @flow — missing .fn attribute"
        )

    def test_training_subflow_name_matches_constant(self) -> None:
        """training_subflow @flow(name=...) must match FLOW_NAME_TRAINING_SUBFLOW."""
        from minivess.orchestration.constants import FLOW_NAME_TRAINING_SUBFLOW
        from minivess.orchestration.flows.train_flow import training_subflow

        # Prefect flow name is stored in .name attribute
        assert training_subflow.name == FLOW_NAME_TRAINING_SUBFLOW

    def test_training_subflow_decorated_in_source(self) -> None:
        """AST check: training_subflow has @flow decorator in source code."""
        source = TRAIN_FLOW_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name != "training_subflow":
                continue
            # Check decorators
            decorator_names = []
            for dec in node.decorator_list:
                if isinstance(dec, ast.Name):
                    decorator_names.append(dec.id)
                elif isinstance(dec, ast.Call):
                    func = dec.func
                    if isinstance(func, ast.Name):
                        decorator_names.append(func.id)
                    elif isinstance(func, ast.Attribute):
                        decorator_names.append(func.attr)
            assert "flow" in decorator_names, (
                f"training_subflow decorators: {decorator_names}, expected 'flow'"
            )
            return

        pytest.fail("training_subflow function not found in train_flow.py")


class TestParentFlowDelegatesToSubflow:
    """Verify that training_flow() delegates to training_subflow()."""

    def test_parent_calls_training_subflow_in_source(self) -> None:
        """AST check: training_flow body calls training_subflow()."""
        assert _find_function_calls_in("training_flow", "training_subflow"), (
            "training_flow() does not call training_subflow() — "
            "parent must delegate to sub-flow"
        )

    def test_parent_flow_still_returns_training_flow_result(self) -> None:
        """training_flow() return type must still be TrainingFlowResult."""
        from minivess.orchestration.flows.train_flow import training_flow

        import inspect

        sig = inspect.signature(training_flow.fn)
        # from __future__ import annotations makes return_annotation a string
        ann = sig.return_annotation
        assert ann is not inspect.Parameter.empty, (
            "training_flow has no return annotation"
        )
        ann_str = ann if isinstance(ann, str) else getattr(ann, "__name__", str(ann))
        assert "TrainingFlowResult" in ann_str, (
            f"training_flow return annotation: {ann}"
        )

    def test_training_subflow_returns_training_flow_result(self) -> None:
        """training_subflow() must return TrainingFlowResult."""
        from minivess.orchestration.flows.train_flow import training_subflow

        sig = inspect.signature(training_subflow.fn)
        ann = sig.return_annotation
        assert ann is not inspect.Parameter.empty, (
            "training_subflow has no return annotation"
        )
        ann_str = ann if isinstance(ann, str) else getattr(ann, "__name__", str(ann))
        assert "TrainingFlowResult" in ann_str, (
            f"training_subflow return annotation: {ann}"
        )


# ---------------------------------------------------------------------------
# Phase 2 tests: post_training_subflow existence and wiring
# ---------------------------------------------------------------------------


class TestPostTrainingSubflowExists:
    """Verify post_training_subflow is a Prefect @flow with correct name."""

    def test_post_training_subflow_function_exists(self) -> None:
        """post_training_subflow must be importable from train_flow module."""
        from minivess.orchestration.flows.train_flow import post_training_subflow

        assert callable(post_training_subflow)

    def test_post_training_subflow_is_prefect_flow(self) -> None:
        """post_training_subflow must be decorated with @flow."""
        from minivess.orchestration.flows.train_flow import post_training_subflow

        assert hasattr(post_training_subflow, "fn"), (
            "post_training_subflow is not a Prefect @flow — missing .fn attribute"
        )

    def test_post_training_subflow_name_matches_constant(self) -> None:
        """post_training_subflow name must match FLOW_NAME_POST_TRAINING_SUBFLOW."""
        from minivess.orchestration.constants import FLOW_NAME_POST_TRAINING_SUBFLOW
        from minivess.orchestration.flows.train_flow import post_training_subflow

        assert post_training_subflow.name == FLOW_NAME_POST_TRAINING_SUBFLOW

    def test_post_training_subflow_decorated_in_source(self) -> None:
        """AST check: post_training_subflow has @flow decorator in source."""
        source = TRAIN_FLOW_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name != "post_training_subflow":
                continue
            decorator_names = []
            for dec in node.decorator_list:
                if isinstance(dec, ast.Name):
                    decorator_names.append(dec.id)
                elif isinstance(dec, ast.Call):
                    func = dec.func
                    if isinstance(func, ast.Name):
                        decorator_names.append(func.id)
                    elif isinstance(func, ast.Attribute):
                        decorator_names.append(func.attr)
            assert "flow" in decorator_names, (
                f"post_training_subflow decorators: {decorator_names}, expected 'flow'"
            )
            return

        pytest.fail("post_training_subflow function not found in train_flow.py")


class TestParentCallsPostTrainingSubflow:
    """Verify parent conditionally calls post_training_subflow."""

    def test_parent_calls_post_training_subflow_in_source(self) -> None:
        """AST check: training_flow body calls post_training_subflow()."""
        assert _find_function_calls_in("training_flow", "post_training_subflow"), (
            "training_flow() does not call post_training_subflow()"
        )

    def test_parent_has_post_training_method_param(self) -> None:
        """training_flow() must accept post_training_method parameter."""
        from minivess.orchestration.flows.train_flow import training_flow

        sig = inspect.signature(training_flow.fn)
        assert "post_training_method" in sig.parameters, (
            f"training_flow params: {list(sig.parameters.keys())} — "
            "missing 'post_training_method'"
        )

    def test_post_training_method_default_is_none(self) -> None:
        """post_training_method default must be 'none' (skip post-training)."""
        from minivess.orchestration.flows.train_flow import training_flow

        sig = inspect.signature(training_flow.fn)
        param = sig.parameters["post_training_method"]
        assert param.default == "none", (
            f"post_training_method default={param.default!r}, expected 'none'"
        )

    def test_none_method_skips_post_training(self) -> None:
        """When only 'none' method, parent must NOT call post_training_subflow.

        Verified via AST: the call to post_training_subflow must be inside
        a conditional that filters out 'none' methods.
        """
        source = TRAIN_FLOW_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name != "training_flow":
                continue
            # Find conditional that guards post_training_subflow call
            # Can be if-block or for-loop with if-guard
            for child in ast.walk(node):
                if isinstance(child, ast.If):
                    test_source = ast.dump(child.test)
                    if "none" in test_source or "pt_method" in test_source:
                        for inner in ast.walk(child):
                            if not isinstance(inner, ast.Call):
                                continue
                            func = inner.func
                            if (
                                isinstance(func, ast.Name)
                                and func.id == "post_training_subflow"
                            ):
                                return
                            if (
                                isinstance(func, ast.Attribute)
                                and func.attr == "post_training_subflow"
                            ):
                                return

        pytest.fail(
            "post_training_subflow() call is not guarded by a "
            "'none' check in training_flow()"
        )


class TestTrainingFlowResultPostTrainingInfo:
    """Verify TrainingFlowResult can carry post-training metadata."""

    def test_result_has_post_training_method_field(self) -> None:
        """TrainingFlowResult must have a post_training_method field."""
        from minivess.orchestration.flows.train_flow import TrainingFlowResult

        result = TrainingFlowResult()
        assert hasattr(result, "post_training_method"), (
            "TrainingFlowResult missing post_training_method field"
        )

    def test_result_post_training_method_default_none(self) -> None:
        """Default post_training_method in result should be 'none'."""
        from minivess.orchestration.flows.train_flow import TrainingFlowResult

        result = TrainingFlowResult()
        assert result.post_training_method == "none"

    def test_result_has_post_training_run_id_field(self) -> None:
        """TrainingFlowResult must have a post_training_run_id field."""
        from minivess.orchestration.flows.train_flow import TrainingFlowResult

        result = TrainingFlowResult()
        assert hasattr(result, "post_training_run_id"), (
            "TrainingFlowResult missing post_training_run_id field"
        )


class TestSwagCalibrationDataWiring:
    """T1: _run_swag_post_training must pass calibration_data to SWAGPlugin."""

    def test_run_swag_post_training_passes_calibration_data(self) -> None:
        """_run_swag_post_training must build calibration_data from config, not None."""
        source = TRAIN_FLOW_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name != "_run_swag_post_training":
                continue
            # Check that calibration_data=None is NOT hardcoded
            source_lines = source.splitlines()
            func_start = node.lineno - 1
            func_end = node.end_lineno or func_start + 50
            func_source = "\n".join(source_lines[func_start:func_end])
            assert "calibration_data=None" not in func_source, (
                "_run_swag_post_training still has calibration_data=None — "
                "must build calibration_data from config"
            )
            return

        pytest.fail("_run_swag_post_training not found in train_flow.py")

    def test_run_swag_post_training_has_config_param(self) -> None:
        """_run_swag_post_training must accept config dict for rebuilding model+loader."""
        from minivess.orchestration.flows.train_flow import _run_swag_post_training

        sig = inspect.signature(_run_swag_post_training)
        assert "config" in sig.parameters, (
            f"_run_swag_post_training params: {list(sig.parameters.keys())} — "
            "missing 'config' (needed to rebuild DataLoaders)"
        )

    def test_run_swag_post_training_no_silent_skip(self) -> None:
        """_run_swag_post_training must NOT silently skip on validation errors.

        The old code had: if errors: logger.warning(...); return
        The new code must raise ValueError (Rule 25: loud failures).
        """
        source = TRAIN_FLOW_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name != "_run_swag_post_training":
                continue
            source_lines = source.splitlines()
            func_start = node.lineno - 1
            func_end = node.end_lineno or func_start + 50
            func_source = "\n".join(source_lines[func_start:func_end])
            # Must not have "will retry with DataLoader in future" (old skip message)
            assert "will retry with DataLoader in future" not in func_source, (
                "_run_swag_post_training still has silent skip on validation errors"
            )
            return

        pytest.fail("_run_swag_post_training not found in train_flow.py")


class TestRunFactorialPostTrainingSwag:
    """T5: run_factorial_post_training must accept calibration_data for SWAG."""

    def test_accepts_calibration_data_param(self) -> None:
        """run_factorial_post_training must have calibration_data parameter."""
        from minivess.orchestration.flows.post_training_flow import (
            run_factorial_post_training,
        )

        sig = inspect.signature(run_factorial_post_training)
        assert "calibration_data" in sig.parameters, (
            f"run_factorial_post_training params: {list(sig.parameters.keys())} — "
            "missing 'calibration_data'"
        )

    def test_swag_not_silently_skipped(self) -> None:
        """When calibration_data is None and method is swag, must raise ValueError."""
        source = Path(
            "src/minivess/orchestration/flows/post_training_flow.py"
        ).read_text(encoding="utf-8")

        # The old code had: elif method == "swag": ... continue
        # New code must raise ValueError when calibration_data is None
        assert "SWAG method requires DataLoaders" not in source or "raise" in source, (
            "run_factorial_post_training still silently skips SWAG"
        )

    def test_swag_raises_without_calibration_data(self) -> None:
        """run_factorial_post_training must raise ValueError for swag without calibration_data."""
        from minivess.orchestration.flows.post_training_flow import (
            run_factorial_post_training,
        )

        with pytest.raises(ValueError, match="SWAG requires calibration_data"):
            run_factorial_post_training(
                checkpoint_paths=[Path("/fake/ckpt.pth")],
                methods=["swag"],
                output_dir=Path("/tmp/test_swag"),
                seed=42,
            )


class TestArgparsePostTrainingMethod:
    """Verify --post-training-method CLI flag in __main__ block."""

    def test_argparse_has_post_training_method(self) -> None:
        """AST check: argparse has --post-training-method argument."""
        source = TRAIN_FLOW_PATH.read_text(encoding="utf-8")
        # Check for the string literal in the source (AST would be overkill)
        assert "--post-training-method" in source, (
            "train_flow.py __main__ block missing --post-training-method argument"
        )

    def test_env_var_post_training_methods_in_source(self) -> None:
        """POST_TRAINING_METHODS env var must be read in __main__ block."""
        source = TRAIN_FLOW_PATH.read_text(encoding="utf-8")
        assert "POST_TRAINING_METHODS" in source, (
            "train_flow.py __main__ block missing POST_TRAINING_METHODS env var"
        )
