"""Tests for JSON sidecar models and R data export (Phase 5.3).

Validates:
- FigureSidecar and TableSidecar Pydantic models
- write_sidecar/load_sidecar round-trip
- Python → JSON export for R consumption
- NumpyEncoder handles all numpy types
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from minivess.pipeline.biostatistics_sidecar import (
    FigureSidecar,
    TableSidecar,
    load_sidecar,
    write_sidecar,
)


class TestFigureSidecar:
    def test_creation(self) -> None:
        s = FigureSidecar(figure_id="F4_forest", title="Forest Plot")
        assert s.figure_id == "F4_forest"
        assert s.title == "Forest Plot"

    def test_has_required_fields(self) -> None:
        s = FigureSidecar(figure_id="test", title="test")
        assert hasattr(s, "duckdb_sha256")
        assert hasattr(s, "git_sha")
        assert hasattr(s, "config_hash")
        assert hasattr(s, "data")
        assert hasattr(s, "output_files")

    def test_write_and_load_roundtrip(self, tmp_path: Path) -> None:
        s = FigureSidecar(
            figure_id="F1",
            title="Test Figure",
            data={"values": [1.0, 2.0, 3.0]},
            output_files=["F1.pdf", "F1.png"],
        )
        path = write_sidecar(s, tmp_path / "F1.json")
        loaded = load_sidecar(path)
        assert loaded["figure_id"] == "F1"
        assert loaded["data"]["values"] == [1.0, 2.0, 3.0]
        assert loaded["output_files"] == ["F1.pdf", "F1.png"]

    def test_json_is_valid(self, tmp_path: Path) -> None:
        s = FigureSidecar(figure_id="test", title="test")
        path = write_sidecar(s, tmp_path / "test.json")
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)


class TestTableSidecar:
    def test_creation(self) -> None:
        s = TableSidecar(table_id="T1", title="Comparison Table")
        assert s.table_id == "T1"

    def test_write_and_load(self, tmp_path: Path) -> None:
        s = TableSidecar(
            table_id="T1",
            title="Test",
            data={"rows": [{"metric": "dsc", "value": 0.8}]},
            output_files=["T1.tex"],
        )
        path = write_sidecar(s, tmp_path / "T1.json")
        loaded = load_sidecar(path)
        assert loaded["table_id"] == "T1"


class TestRDataExport:
    def test_export_pairwise_results(self, tmp_path: Path) -> None:
        from minivess.pipeline.biostatistics_r_export import export_pairwise_results
        from minivess.pipeline.biostatistics_types import PairwiseResult

        results = [
            PairwiseResult(
                condition_a="a", condition_b="b", metric="dsc",
                p_value=0.01, p_adjusted=0.02, correction_method="holm",
                significant=True, cohens_d=0.5, cliffs_delta=0.3, vda=0.65,
            ),
        ]
        path = export_pairwise_results(results, tmp_path)
        assert path.exists()
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["condition_a"] == "a"

    def test_export_per_volume_data(self, tmp_path: Path) -> None:
        from minivess.pipeline.biostatistics_r_export import export_per_volume_data

        pvd = {
            "dsc": {
                "cond_a": {0: np.array([0.8, 0.85, 0.79])},
                "cond_b": {0: np.array([0.75, 0.78, 0.76])},
            }
        }
        path = export_per_volume_data(pvd, tmp_path)
        assert path.exists()
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 6  # 2 conditions × 3 volumes

    def test_export_all_r_data(self, tmp_path: Path) -> None:
        from minivess.pipeline.biostatistics_r_export import export_all_r_data
        from minivess.pipeline.biostatistics_types import (
            PairwiseResult,
            RankingResult,
            VarianceDecompositionResult,
        )

        paths = export_all_r_data(
            pairwise=[
                PairwiseResult(
                    condition_a="a", condition_b="b", metric="dsc",
                    p_value=0.01, p_adjusted=0.02, correction_method="holm",
                    significant=True, cohens_d=0.5, cliffs_delta=0.3, vda=0.65,
                ),
            ],
            variance=[
                VarianceDecompositionResult(
                    metric="dsc", friedman_statistic=5.0, friedman_p=0.02,
                    nemenyi_matrix=None, icc_value=0.8,
                    icc_ci_lower=0.5, icc_ci_upper=0.95, icc_type="ICC2",
                ),
            ],
            rankings=[
                RankingResult(
                    metric="dsc",
                    condition_ranks={"a": 1, "b": 2},
                    mean_ranks={"a": 1.0, "b": 2.0},
                    cd_value=1.5,
                ),
            ],
            per_volume_data={
                "dsc": {
                    "a": {0: np.array([0.8])},
                    "b": {0: np.array([0.75])},
                }
            },
            output_dir=tmp_path,
        )
        assert len(paths) == 4
        for p in paths:
            assert p.exists()

    def test_numpy_types_serializable(self, tmp_path: Path) -> None:
        from minivess.pipeline.biostatistics_r_export import export_per_volume_data

        pvd = {
            "dsc": {
                "a": {np.int64(0): np.array([np.float64(0.8)], dtype=np.float64)},
            }
        }
        path = export_per_volume_data(pvd, tmp_path)
        assert path.exists()
