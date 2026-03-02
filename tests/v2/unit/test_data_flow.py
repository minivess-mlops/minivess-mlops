"""Tests for Prefect Data Flow (Flow 1) tasks and orchestrator.

Covers Tasks 3.1 and 3.2 of data-engineering-improvement-plan.xml.
Closes #178.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Task 3.1: Individual @task function tests
# ---------------------------------------------------------------------------


class TestDiscoverDataTask:
    """discover_data_task wraps discover_nifti_pairs."""

    def test_discover_data_task_returns_list(self, tmp_path: Path) -> None:
        from minivess.orchestration.flows.data_flow import discover_data_task

        # Create minimal NIfTI-like directory structure
        (tmp_path / "images").mkdir()
        (tmp_path / "labels").mkdir()
        (tmp_path / "images" / "vol_001.nii.gz").write_bytes(b"fake")
        (tmp_path / "labels" / "vol_001.nii.gz").write_bytes(b"fake")

        result = discover_data_task(data_dir=tmp_path)
        assert isinstance(result, list)

    def test_discover_data_task_returns_pairs(self, tmp_path: Path) -> None:
        from minivess.orchestration.flows.data_flow import discover_data_task

        (tmp_path / "images").mkdir()
        (tmp_path / "labels").mkdir()
        for i in range(3):
            (tmp_path / "images" / f"vol_{i:03d}.nii.gz").write_bytes(b"fake")
            (tmp_path / "labels" / f"vol_{i:03d}.nii.gz").write_bytes(b"fake")

        result = discover_data_task(data_dir=tmp_path)
        assert len(result) == 3
        for pair in result:
            assert "image" in pair
            assert "label" in pair


class TestValidateDataTask:
    """validate_data_task returns DataValidationReport."""

    def test_validate_data_task_returns_report(self) -> None:
        from minivess.orchestration.flows.data_flow import (
            DataValidationReport,
            validate_data_task,
        )

        pairs = [
            {"image": "/fake/img.nii.gz", "label": "/fake/lbl.nii.gz"},
        ]
        report = validate_data_task(pairs=pairs)
        assert isinstance(report, DataValidationReport)

    def test_validate_report_has_counts(self) -> None:
        from minivess.orchestration.flows.data_flow import validate_data_task

        pairs = [
            {"image": "/fake/img1.nii.gz", "label": "/fake/lbl1.nii.gz"},
            {"image": "/fake/img2.nii.gz", "label": "/fake/lbl2.nii.gz"},
        ]
        report = validate_data_task(pairs=pairs)
        assert report.n_pairs == 2
        assert isinstance(report.errors, list)
        assert isinstance(report.warnings, list)


class TestDataQualityGate:
    """data_quality_gate returns bool based on validation report."""

    def test_quality_gate_passes_valid(self) -> None:
        from minivess.orchestration.flows.data_flow import (
            DataValidationReport,
            data_quality_gate,
        )

        report = DataValidationReport(n_pairs=10, n_valid=10, errors=[], warnings=[])
        assert data_quality_gate(report=report) is True

    def test_quality_gate_fails_empty(self) -> None:
        from minivess.orchestration.flows.data_flow import (
            DataValidationReport,
            data_quality_gate,
        )

        report = DataValidationReport(
            n_pairs=0, n_valid=0, errors=["No data found"], warnings=[]
        )
        assert data_quality_gate(report=report) is False


class TestValidateExternalDataTask:
    """validate_external_data_task checks external dataset pairs."""

    def test_validate_external_returns_list(self, tmp_path: Path) -> None:
        from minivess.orchestration.flows.data_flow import (
            validate_external_data_task,
        )

        # Create directory matching expected layout
        (tmp_path / "images").mkdir()
        (tmp_path / "labels").mkdir()
        (tmp_path / "images" / "vol_001.nii.gz").write_bytes(b"fake")
        (tmp_path / "labels" / "vol_001.nii.gz").write_bytes(b"fake")

        result = validate_external_data_task(dataset_name="deepvess", data_dir=tmp_path)
        assert isinstance(result, list)


class TestLogDataProvenanceTask:
    """log_data_provenance_task logs to MLflow-compatible dict."""

    def test_log_provenance_returns_dict(self) -> None:
        from minivess.orchestration.flows.data_flow import (
            log_data_provenance_task,
        )

        result = log_data_provenance_task(
            n_volumes=10,
            n_folds=3,
            dataset_hash="abc123",
        )
        assert isinstance(result, dict)
        assert "data_n_volumes" in result
        assert "data_n_folds" in result
        assert "data_hash" in result


# ---------------------------------------------------------------------------
# Task 3.2: @flow orchestrator tests
# ---------------------------------------------------------------------------


class TestDataFlowResult:
    """DataFlowResult dataclass."""

    def test_data_flow_result_has_required_fields(self) -> None:
        from minivess.orchestration.flows.data_flow import DataFlowResult

        result = DataFlowResult(
            pairs=[],
            validation_report=None,
            quality_passed=True,
            n_folds=3,
            splits=None,
            external_datasets={},
            provenance={},
        )
        assert result.quality_passed is True
        assert result.n_folds == 3


class TestRunDataFlow:
    """run_data_flow orchestrator."""

    def test_run_data_flow_returns_result(self, tmp_path: Path) -> None:
        from minivess.orchestration.flows.data_flow import (
            DataFlowResult,
            run_data_flow,
        )

        # Create minimal dataset
        (tmp_path / "images").mkdir()
        (tmp_path / "labels").mkdir()
        for i in range(4):
            (tmp_path / "images" / f"vol_{i:03d}.nii.gz").write_bytes(b"fake")
            (tmp_path / "labels" / f"vol_{i:03d}.nii.gz").write_bytes(b"fake")

        result = run_data_flow(data_dir=tmp_path, n_folds=2, seed=42)
        assert isinstance(result, DataFlowResult)

    def test_data_flow_result_has_pairs(self, tmp_path: Path) -> None:
        from minivess.orchestration.flows.data_flow import run_data_flow

        (tmp_path / "images").mkdir()
        (tmp_path / "labels").mkdir()
        for i in range(4):
            (tmp_path / "images" / f"vol_{i:03d}.nii.gz").write_bytes(b"fake")
            (tmp_path / "labels" / f"vol_{i:03d}.nii.gz").write_bytes(b"fake")

        result = run_data_flow(data_dir=tmp_path, n_folds=2, seed=42)
        assert len(result.pairs) == 4

    def test_data_flow_result_has_splits(self, tmp_path: Path) -> None:
        from minivess.orchestration.flows.data_flow import run_data_flow

        (tmp_path / "images").mkdir()
        (tmp_path / "labels").mkdir()
        for i in range(4):
            (tmp_path / "images" / f"vol_{i:03d}.nii.gz").write_bytes(b"fake")
            (tmp_path / "labels" / f"vol_{i:03d}.nii.gz").write_bytes(b"fake")

        result = run_data_flow(data_dir=tmp_path, n_folds=2, seed=42)
        assert result.splits is not None
        assert len(result.splits) == 2

    def test_data_flow_result_has_external(self, tmp_path: Path) -> None:
        from minivess.orchestration.flows.data_flow import run_data_flow

        (tmp_path / "images").mkdir()
        (tmp_path / "labels").mkdir()
        for i in range(4):
            (tmp_path / "images" / f"vol_{i:03d}.nii.gz").write_bytes(b"fake")
            (tmp_path / "labels" / f"vol_{i:03d}.nii.gz").write_bytes(b"fake")

        result = run_data_flow(data_dir=tmp_path, n_folds=2, seed=42)
        assert isinstance(result.external_datasets, dict)

    def test_data_flow_quality_gate_runs(self, tmp_path: Path) -> None:
        from minivess.orchestration.flows.data_flow import run_data_flow

        (tmp_path / "images").mkdir()
        (tmp_path / "labels").mkdir()
        for i in range(4):
            (tmp_path / "images" / f"vol_{i:03d}.nii.gz").write_bytes(b"fake")
            (tmp_path / "labels" / f"vol_{i:03d}.nii.gz").write_bytes(b"fake")

        result = run_data_flow(data_dir=tmp_path, n_folds=2, seed=42)
        assert result.quality_passed is True
