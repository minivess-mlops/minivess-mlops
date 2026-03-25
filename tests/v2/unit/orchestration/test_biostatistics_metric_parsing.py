"""Biostatistics metric parsing — Phase 6 Task 6.2.

Verifies that biostatistics_duckdb.py correctly parses eval metrics in
the slash format (eval/{fold}/{metric}) produced by tracking.py.

Plan: run-debug-factorial-experiment-report-6th-pass-post-run-fix-2.xml
"""

from __future__ import annotations


class TestBiostatisticsMetricParsing:
    """Biostatistics parsers must handle slash-format eval metric keys."""

    def test_is_eval_fold_metric_slash_format(self) -> None:
        """_is_eval_fold_metric must recognize eval/0/dsc format."""
        from minivess.pipeline.biostatistics_duckdb import _is_eval_fold_metric

        assert _is_eval_fold_metric("eval/0/dsc") is True
        assert _is_eval_fold_metric("eval/1/cldice") is True
        assert _is_eval_fold_metric("eval/2/measured_masd") is True

    def test_is_eval_fold_metric_rejects_non_eval(self) -> None:
        """_is_eval_fold_metric must reject non-eval keys."""
        from minivess.pipeline.biostatistics_duckdb import _is_eval_fold_metric

        assert _is_eval_fold_metric("train/loss") is False
        assert _is_eval_fold_metric("val/dice") is False
        assert _is_eval_fold_metric("test/deepvess/all/dsc") is False

    def test_parse_eval_fold_metric(self) -> None:
        """_parse_eval_fold_metric must extract (fold_id, base_metric)."""
        from minivess.pipeline.biostatistics_duckdb import _parse_eval_fold_metric

        result = _parse_eval_fold_metric("eval/0/dsc")
        assert result is not None
        fold_id, metric_name = result
        assert fold_id == 0
        assert metric_name == "dsc"

    def test_parse_eval_fold_metric_fold2(self) -> None:
        """Must parse fold 2 correctly (the gate key used by builder.py)."""
        from minivess.pipeline.biostatistics_duckdb import _parse_eval_fold_metric

        result = _parse_eval_fold_metric("eval/2/dsc")
        assert result is not None
        assert result[0] == 2
        assert result[1] == "dsc"

    def test_is_per_volume_metric(self) -> None:
        """_is_per_volume_metric must recognize eval/{fold}/vol/{id}/{metric}."""
        from minivess.pipeline.biostatistics_duckdb import _is_per_volume_metric

        assert _is_per_volume_metric("eval/0/vol/3/dsc") is True
        assert _is_per_volume_metric("eval/1/vol/mv01/cldice") is True

    def test_is_per_volume_metric_rejects_fold_level(self) -> None:
        """_is_per_volume_metric must reject fold-level metrics."""
        from minivess.pipeline.biostatistics_duckdb import _is_per_volume_metric

        assert _is_per_volume_metric("eval/0/dsc") is False

    def test_parse_per_volume_metric(self) -> None:
        """_parse_per_volume_metric must extract (fold_id, volume_id, base_metric)."""
        from minivess.pipeline.biostatistics_duckdb import _parse_per_volume_metric

        fold_id, vol_id, metric = _parse_per_volume_metric("eval/0/vol/mv01/dsc")
        assert fold_id == 0
        assert vol_id == "mv01"
        assert metric == "dsc"
