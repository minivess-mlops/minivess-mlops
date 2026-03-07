"""Tests for T-02: regex ban fix in duckdb_extraction.py.

Verifies that eval_fold metric name parsing uses str.partition()
instead of re.match(), and that 'import re' is removed.
"""

from __future__ import annotations

import ast
from pathlib import Path

_DUCKDB_SRC = Path("src/minivess/pipeline/duckdb_extraction.py")


def _load_source() -> str:
    return _DUCKDB_SRC.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Tests for parse_eval_fold_metric() helper
# ---------------------------------------------------------------------------


class TestParseEvalFoldMetric:
    """Tests for parse_eval_fold_metric() that replaces re.match()."""

    def test_metric_parse_eval_fold_basic(self) -> None:
        from minivess.pipeline.duckdb_extraction import parse_eval_fold_metric

        result = parse_eval_fold_metric("eval_fold0_val_dice")
        assert result is not None
        fold_id, metric = result
        assert fold_id == 0
        assert metric == "val_dice"

    def test_metric_parse_multi_digit_fold(self) -> None:
        from minivess.pipeline.duckdb_extraction import parse_eval_fold_metric

        result = parse_eval_fold_metric("eval_fold12_compound_masd")
        assert result is not None
        fold_id, metric = result
        assert fold_id == 12
        assert metric == "compound_masd"

    def test_metric_parse_underscore_in_metric_name(self) -> None:
        from minivess.pipeline.duckdb_extraction import parse_eval_fold_metric

        result = parse_eval_fold_metric("eval_fold0_val_dice_ce_cldice")
        assert result is not None
        fold_id, metric = result
        assert fold_id == 0
        assert metric == "val_dice_ce_cldice"

    def test_metric_parse_non_eval_fold_returns_none(self) -> None:
        from minivess.pipeline.duckdb_extraction import parse_eval_fold_metric

        assert parse_eval_fold_metric("train_loss") is None

    def test_metric_parse_no_prefix_returns_none(self) -> None:
        from minivess.pipeline.duckdb_extraction import parse_eval_fold_metric

        assert parse_eval_fold_metric("val_dice") is None

    def test_metric_parse_partial_prefix_returns_none(self) -> None:
        from minivess.pipeline.duckdb_extraction import parse_eval_fold_metric

        # "eval_fold" without digit → should return None
        assert parse_eval_fold_metric("eval_fold_val_dice") is None

    def test_fold_id_is_int(self) -> None:
        from minivess.pipeline.duckdb_extraction import parse_eval_fold_metric

        result = parse_eval_fold_metric("eval_fold2_dsc")
        assert result is not None
        fold_id, _ = result
        assert isinstance(fold_id, int)

    def test_metric_name_is_str(self) -> None:
        from minivess.pipeline.duckdb_extraction import parse_eval_fold_metric

        result = parse_eval_fold_metric("eval_fold0_nsd")
        assert result is not None
        _, metric = result
        assert isinstance(metric, str)


# ---------------------------------------------------------------------------
# AST-level: no 'import re' in duckdb_extraction.py
# ---------------------------------------------------------------------------


class TestNoRegexImport:
    def test_duckdb_no_re_import(self) -> None:
        source = _load_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "re", (
                        f"Found banned 'import re' in duckdb_extraction.py at line {node.lineno}."
                    )
            if isinstance(node, ast.ImportFrom):
                assert node.module != "re", (
                    f"Found 'from re import ...' in duckdb_extraction.py at line {node.lineno}."
                )

    def test_duckdb_no_re_match_call(self) -> None:
        source = _load_source()
        assert "re.match" not in source, (
            "Found banned 're.match()' in duckdb_extraction.py. "
            "Use parse_eval_fold_metric() with str.partition() instead."
        )
