"""Tests for regex ban compliance in duckdb_extraction.py.

Verifies that eval fold metric name parsing uses str.split("/")
instead of re.match(), and that 'import re' is removed.

Slash-prefix convention (#790): eval/{fold}/{metric}
"""

from __future__ import annotations

import ast
from pathlib import Path

_DUCKDB_SRC = Path("src/minivess/pipeline/duckdb_extraction.py")


def _load_source() -> str:
    return _DUCKDB_SRC.read_text(encoding="utf-8")


class TestParseEvalFoldMetric:
    def test_metric_parse_eval_fold_basic(self) -> None:
        from minivess.pipeline.duckdb_extraction import parse_eval_fold_metric

        result = parse_eval_fold_metric("eval/0/dsc")
        assert result is not None
        assert result[0] == 0
        assert result[1] == "dsc"

    def test_metric_parse_multi_digit_fold(self) -> None:
        from minivess.pipeline.duckdb_extraction import parse_eval_fold_metric

        result = parse_eval_fold_metric("eval/12/compound_masd")
        assert result is not None
        assert result[0] == 12
        assert result[1] == "compound_masd"

    def test_metric_parse_complex_metric_name(self) -> None:
        from minivess.pipeline.duckdb_extraction import parse_eval_fold_metric

        result = parse_eval_fold_metric("eval/0/compound_masd_cldice")
        assert result is not None
        assert result[0] == 0
        assert result[1] == "compound_masd_cldice"

    def test_metric_parse_non_eval_returns_none(self) -> None:
        from minivess.pipeline.duckdb_extraction import parse_eval_fold_metric

        assert parse_eval_fold_metric("train/loss") is None

    def test_metric_parse_no_prefix_returns_none(self) -> None:
        from minivess.pipeline.duckdb_extraction import parse_eval_fold_metric

        assert parse_eval_fold_metric("val/dice") is None

    def test_metric_parse_non_digit_fold_returns_none(self) -> None:
        from minivess.pipeline.duckdb_extraction import parse_eval_fold_metric

        assert parse_eval_fold_metric("eval/abc/dsc") is None

    def test_metric_parse_old_format_returns_none(self) -> None:
        from minivess.pipeline.duckdb_extraction import parse_eval_fold_metric

        assert parse_eval_fold_metric("eval_fold0_val_dice") is None

    def test_fold_id_is_int(self) -> None:
        from minivess.pipeline.duckdb_extraction import parse_eval_fold_metric

        result = parse_eval_fold_metric("eval/2/dsc")
        assert result is not None
        assert isinstance(result[0], int)

    def test_metric_name_is_str(self) -> None:
        from minivess.pipeline.duckdb_extraction import parse_eval_fold_metric

        result = parse_eval_fold_metric("eval/0/nsd")
        assert result is not None
        assert isinstance(result[1], str)


class TestNoRegexImport:
    def test_duckdb_no_re_import(self) -> None:
        source = _load_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "re"
            if isinstance(node, ast.ImportFrom):
                assert node.module != "re"

    def test_duckdb_no_re_match_call(self) -> None:
        source = _load_source()
        assert "re.match" not in source
