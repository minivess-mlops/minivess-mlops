"""Tests for Nature Protocols compliance artifacts (Phase 8).

Validates CONSORT data, TRIPOD mapping, limitations paragraph,
and provenance chain generation.
"""

from __future__ import annotations

import json
from pathlib import Path

from minivess.pipeline.biostatistics_compliance import (
    generate_consort_data,
    generate_limitations_paragraph,
    generate_provenance_chain_mermaid,
    generate_tripod_mapping,
)


class TestConsortData:
    def test_generates_valid_dict(self) -> None:
        data = generate_consort_data()
        assert isinstance(data, dict)
        assert "enrollment" in data
        assert "allocation" in data
        assert "training" in data

    def test_enrollment_has_70_volumes(self) -> None:
        data = generate_consort_data()
        assert data["enrollment"]["total_volumes"] == 70

    def test_allocation_3_folds(self) -> None:
        data = generate_consort_data()
        assert data["allocation"]["n_folds"] == 3

    def test_external_test_deepvess(self) -> None:
        data = generate_consort_data()
        assert data["external_test"]["dataset"] == "DeepVess"
        assert data["external_test"]["n_volumes"] == 7

    def test_writes_to_file(self, tmp_path: Path) -> None:
        generate_consort_data(output_dir=tmp_path)
        path = tmp_path / "consort_data.json"
        assert path.exists()
        with path.open(encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["enrollment"]["total_volumes"] == 70

    def test_training_runs_count(self) -> None:
        data = generate_consort_data(n_losses=2, n_folds=3)
        assert data["training"]["n_training_runs"] == 6


class TestTripodMapping:
    def test_generates_valid_dict(self) -> None:
        mapping = generate_tripod_mapping()
        assert isinstance(mapping, dict)
        assert len(mapping) >= 8

    def test_consort_maps_to_item_6(self) -> None:
        mapping = generate_tripod_mapping()
        assert "6" in mapping["F2_consort"]["tripod_item"]

    def test_power_maps_to_item_9a(self) -> None:
        mapping = generate_tripod_mapping()
        assert "9a" in mapping["T8_power"]["tripod_item"]

    def test_preregistration_maps_to_item_3(self) -> None:
        mapping = generate_tripod_mapping()
        assert "3" in mapping["pre_registration"]["tripod_item"]

    def test_writes_to_file(self, tmp_path: Path) -> None:
        generate_tripod_mapping(output_dir=tmp_path)
        path = tmp_path / "tripod_mapping.json"
        assert path.exists()


class TestLimitationsParagraph:
    def test_generates_string(self) -> None:
        para = generate_limitations_paragraph()
        assert isinstance(para, str)
        assert "K=3" in para
        assert "N=23" in para

    def test_includes_power_results(self) -> None:
        para = generate_limitations_paragraph(
            power_results={"detectable_d": 0.45}
        )
        assert "0.45" in para

    def test_writes_tex_file(self, tmp_path: Path) -> None:
        generate_limitations_paragraph(output_dir=tmp_path)
        path = tmp_path / "limitations_paragraph.tex"
        assert path.exists()


class TestProvenanceChainMermaid:
    def test_generates_mermaid_string(self) -> None:
        mmd = generate_provenance_chain_mermaid(duckdb_sha="abc123def456")
        assert "graph LR" in mmd
        assert "abc123def456" in mmd

    def test_writes_mmd_file(self, tmp_path: Path) -> None:
        generate_provenance_chain_mermaid(
            duckdb_sha="test_sha",
            output_dir=tmp_path,
        )
        path = tmp_path / "provenance_chain.mmd"
        assert path.exists()
