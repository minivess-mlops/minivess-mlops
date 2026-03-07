"""Tests for T-01: regex ban fix in analysis_flow.py.

Verifies that _parse_model_name() uses str.rsplit() instead of re.compile(),
and that 'import re' is removed from analysis_flow.py.
"""

from __future__ import annotations

import ast
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ANALYSIS_FLOW_SRC = Path("src/minivess/orchestration/flows/analysis_flow.py")


def _load_analysis_source() -> str:
    return _ANALYSIS_FLOW_SRC.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Tests for parse_fold_metric helper (the replacement for _FOLD_RE)
# ---------------------------------------------------------------------------


class TestParseFoldMetric:
    """Tests for the parse_fold_metric() helper that replaces _FOLD_RE."""

    def test_fold_name_parse_basic(self) -> None:
        from minivess.orchestration.flows.analysis_flow import parse_fold_metric

        loss_fn, fold_id = parse_fold_metric("val_dice_fold0")
        assert loss_fn == "val_dice"
        assert fold_id == 0

    def test_fold_name_parse_multi_fold(self) -> None:
        from minivess.orchestration.flows.analysis_flow import parse_fold_metric

        loss_fn, fold_id = parse_fold_metric("compound_masd_cldice_fold2")
        assert loss_fn == "compound_masd_cldice"
        assert fold_id == 2

    def test_fold_name_parse_fold10(self) -> None:
        from minivess.orchestration.flows.analysis_flow import parse_fold_metric

        loss_fn, fold_id = parse_fold_metric("dice_ce_fold10")
        assert loss_fn == "dice_ce"
        assert fold_id == 10

    def test_fold_name_parse_no_fold_raises(self) -> None:
        import pytest

        from minivess.orchestration.flows.analysis_flow import parse_fold_metric

        with pytest.raises(ValueError, match="_fold"):
            parse_fold_metric("val_loss")

    def test_fold_name_parse_ensemble_name_raises(self) -> None:
        import pytest

        from minivess.orchestration.flows.analysis_flow import parse_fold_metric

        with pytest.raises(ValueError):
            parse_fold_metric("per_loss_single_best")

    def test_fold_name_fold_id_is_int(self) -> None:
        from minivess.orchestration.flows.analysis_flow import parse_fold_metric

        _, fold_id = parse_fold_metric("cbdice_cldice_fold1")
        assert isinstance(fold_id, int)

    def test_fold_name_loss_fn_is_str(self) -> None:
        from minivess.orchestration.flows.analysis_flow import parse_fold_metric

        loss_fn, _ = parse_fold_metric("dice_ce_cldice_fold0")
        assert isinstance(loss_fn, str)


# ---------------------------------------------------------------------------
# Tests for _parse_model_name backward compatibility
# ---------------------------------------------------------------------------


class TestParseModelNameBackwardCompat:
    """_parse_model_name must still return (None, None) for non-fold names."""

    def test_parse_model_name_fold_name(self) -> None:
        from minivess.orchestration.flows.analysis_flow import _parse_model_name

        loss_fn, fold_id = _parse_model_name("dice_ce_fold0")
        assert loss_fn == "dice_ce"
        assert fold_id == 0

    def test_parse_model_name_ensemble_returns_none_pair(self) -> None:
        from minivess.orchestration.flows.analysis_flow import _parse_model_name

        loss_fn, fold_id = _parse_model_name("per_loss_single_best")
        assert loss_fn is None
        assert fold_id is None

    def test_parse_model_name_no_fold_returns_none_pair(self) -> None:
        from minivess.orchestration.flows.analysis_flow import _parse_model_name

        loss_fn, fold_id = _parse_model_name("val_loss")
        assert loss_fn is None
        assert fold_id is None


# ---------------------------------------------------------------------------
# AST-level check: no 'import re' in analysis_flow.py
# ---------------------------------------------------------------------------


class TestNoRegexImport:
    """Verify analysis_flow.py contains no 'import re' at the AST level."""

    def test_analysis_flow_no_re_import(self) -> None:
        source = _load_analysis_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "re", (
                        f"Found banned 'import re' in analysis_flow.py at line {node.lineno}. "
                        "Use str.rsplit() instead of re.compile() for structured string parsing."
                    )
            if isinstance(node, ast.ImportFrom):
                assert node.module != "re", (
                    f"Found banned 'from re import ...' in analysis_flow.py at line {node.lineno}."
                )

    def test_analysis_flow_no_re_compile(self) -> None:
        source = _load_analysis_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                # Check for re.compile(...)
                if isinstance(func, ast.Attribute) and (
                    func.attr == "compile"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "re"
                ):
                    raise AssertionError(
                        "Found banned 're.compile()' call in analysis_flow.py. "
                        "Use str.rsplit() for parsing structured metric names."
                    )

    def test_analysis_flow_no_fold_re_module_level(self) -> None:
        """_FOLD_RE module-level regex constant must not exist."""
        source = _load_analysis_source()
        assert "_FOLD_RE" not in source, (
            "Found '_FOLD_RE' module-level regex constant in analysis_flow.py. "
            "It must be replaced by parse_fold_metric() using str.rsplit()."
        )
