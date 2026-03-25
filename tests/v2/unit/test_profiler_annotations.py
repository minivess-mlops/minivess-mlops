"""Tests for record_function annotations in trainer.py.

TDD RED phase for T1.1 (#645): Verify that torch.profiler.record_function
context managers are present in train_epoch() and validate_epoch() with
correct region names. Uses AST parsing (Rule #16 — no regex).

record_function is a no-op when the profiler is not active, so these
annotations add zero overhead to non-profiled runs.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# Path to trainer.py
TRAINER_PY = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "src"
    / "minivess"
    / "pipeline"
    / "trainer.py"
)

# Expected region names per the execution plan (RC1, RC13)
# Updated for gradient accumulation refactor: "backward_optimizer" split into
# "backward" + "optimizer_step", "zero_grad" is now part of optimizer_step block.
EXPECTED_TRAIN_REGIONS = {
    "data_to_device",
    "forward",
    "loss_compute",
    "backward",
    "optimizer_step",
    "metrics_update",
}

EXPECTED_VAL_REGIONS = {
    "val_data_to_device",
    "val_forward",
    "val_loss_compute",
    "val_extended_metrics",
}

BANNED_REGION_NAMES = {"data_loading"}


def _extract_record_function_names_from_method(
    tree: ast.Module, method_name: str
) -> set[str]:
    """Extract record_function string arguments from a specific method.

    Walks the AST looking for:
    - `with record_function("name"):`  (ast.With + ast.Call)
    - `record_function("name")`  (direct call in any context)

    Only searches inside methods matching `method_name`.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        # Find method definitions with the right name
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if node.name != method_name:
                continue
            # Walk all nodes inside this method
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    func = child.func
                    # record_function("name") — direct call
                    if (
                        isinstance(func, ast.Name)
                        and func.id == "record_function"
                        and child.args
                        and isinstance(child.args[0], ast.Constant)
                    ):
                        names.add(child.args[0].value)
                    # torch.profiler.record_function("name") — qualified call
                    if (
                        isinstance(func, ast.Attribute)
                        and func.attr == "record_function"
                        and child.args
                        and isinstance(child.args[0], ast.Constant)
                    ):
                        names.add(child.args[0].value)
    return names


@pytest.fixture(scope="module")
def trainer_ast() -> ast.Module:
    """Parse trainer.py into an AST tree."""
    source = TRAINER_PY.read_text(encoding="utf-8")
    return ast.parse(source, filename=str(TRAINER_PY))


class TestTrainEpochAnnotations:
    """Verify record_function annotations in train_epoch()."""

    def test_train_epoch_has_record_function_regions(
        self, trainer_ast: ast.Module
    ) -> None:
        names = _extract_record_function_names_from_method(trainer_ast, "train_epoch")
        assert len(names) >= 6, (
            f"train_epoch must have at least 6 record_function regions, "
            f"found {len(names)}: {names}"
        )

    def test_record_function_region_names_train(self, trainer_ast: ast.Module) -> None:
        names = _extract_record_function_names_from_method(trainer_ast, "train_epoch")
        missing = EXPECTED_TRAIN_REGIONS - names
        assert not missing, (
            f"train_epoch missing record_function regions: {missing}. Found: {names}"
        )


class TestValidateEpochAnnotations:
    """Verify record_function annotations in validate_epoch()."""

    def test_validate_epoch_has_record_function_regions(
        self, trainer_ast: ast.Module
    ) -> None:
        names = _extract_record_function_names_from_method(
            trainer_ast, "validate_epoch"
        )
        assert len(names) >= 4, (
            f"validate_epoch must have at least 4 record_function regions, "
            f"found {len(names)}: {names}"
        )

    def test_record_function_region_names_val(self, trainer_ast: ast.Module) -> None:
        names = _extract_record_function_names_from_method(
            trainer_ast, "validate_epoch"
        )
        missing = EXPECTED_VAL_REGIONS - names
        assert not missing, (
            f"validate_epoch missing record_function regions: {missing}. Found: {names}"
        )


class TestBannedRegionNames:
    """Verify banned region names are NOT used."""

    def test_data_to_device_not_named_data_loading(
        self, trainer_ast: ast.Module
    ) -> None:
        """No record_function('data_loading') anywhere in trainer.py."""
        all_names: set[str] = set()
        for method_name in ("train_epoch", "validate_epoch"):
            all_names |= _extract_record_function_names_from_method(
                trainer_ast, method_name
            )
        found_banned = all_names & BANNED_REGION_NAMES
        assert not found_banned, (
            f"Banned record_function region names found: {found_banned}. "
            "Use 'data_to_device' instead of 'data_loading'."
        )
