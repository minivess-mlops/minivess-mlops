"""Tests for AcquisitionConfig dataclass.

Phase 1, Task 1.1 of flow-data-acquisition-plan.md.
"""

from __future__ import annotations

from pathlib import Path


class TestAcquisitionConfig:
    """AcquisitionConfig stores acquisition flow configuration."""

    def test_default_config(self) -> None:
        from minivess.config.acquisition_config import AcquisitionConfig

        config = AcquisitionConfig()
        assert isinstance(config.datasets, list)
        assert len(config.datasets) == 4
        assert config.skip_existing is True
        assert config.convert_formats is True
        assert config.verify_checksums is True

    def test_custom_datasets(self) -> None:
        from minivess.config.acquisition_config import AcquisitionConfig

        config = AcquisitionConfig(datasets=["vesselnn", "deepvess"])
        assert config.datasets == ["vesselnn", "deepvess"]

    def test_custom_output_dir(self, tmp_path: Path) -> None:
        from minivess.config.acquisition_config import AcquisitionConfig

        config = AcquisitionConfig(output_dir=tmp_path / "raw")
        assert config.output_dir == tmp_path / "raw"

    def test_default_output_dir_is_path(self) -> None:
        from minivess.config.acquisition_config import AcquisitionConfig

        config = AcquisitionConfig()
        assert isinstance(config.output_dir, Path)

    def test_validate_datasets_valid(self) -> None:
        from minivess.config.acquisition_config import AcquisitionConfig

        config = AcquisitionConfig(datasets=["minivess", "vesselnn"])
        errors = config.validate()
        assert errors == []

    def test_validate_datasets_invalid(self) -> None:
        from minivess.config.acquisition_config import AcquisitionConfig

        config = AcquisitionConfig(datasets=["nonexistent_dataset"])
        errors = config.validate()
        assert len(errors) == 1
        assert "nonexistent_dataset" in errors[0]

    def test_validate_empty_datasets(self) -> None:
        from minivess.config.acquisition_config import AcquisitionConfig

        config = AcquisitionConfig(datasets=[])
        errors = config.validate()
        assert len(errors) == 1


class TestDatasetAcquisitionStatus:
    """DatasetAcquisitionStatus enum for tracking per-dataset state."""

    def test_status_values_exist(self) -> None:
        from minivess.config.acquisition_config import DatasetAcquisitionStatus

        assert DatasetAcquisitionStatus.READY is not None
        assert DatasetAcquisitionStatus.DOWNLOADED is not None
        assert DatasetAcquisitionStatus.MANUAL_REQUIRED is not None
        assert DatasetAcquisitionStatus.FAILED is not None

    def test_status_is_string_enum(self) -> None:
        from minivess.config.acquisition_config import DatasetAcquisitionStatus

        assert DatasetAcquisitionStatus.READY.value == "ready"
        assert DatasetAcquisitionStatus.MANUAL_REQUIRED.value == "manual_required"


class TestAcquisitionResult:
    """AcquisitionResult holds output of the acquisition flow."""

    def test_result_construction(self) -> None:
        from minivess.config.acquisition_config import (
            AcquisitionResult,
            DatasetAcquisitionStatus,
        )

        result = AcquisitionResult(
            datasets_acquired={"vesselnn": DatasetAcquisitionStatus.READY},
            total_volumes=12,
            conversion_log=["converted foo.tif → foo.nii.gz"],
            provenance={"acq_timestamp": "2026-03-04"},
        )
        assert result.total_volumes == 12
        assert len(result.conversion_log) == 1
        assert "vesselnn" in result.datasets_acquired

    def test_result_empty(self) -> None:
        from minivess.config.acquisition_config import AcquisitionResult

        result = AcquisitionResult(
            datasets_acquired={},
            total_volumes=0,
            conversion_log=[],
            provenance={},
        )
        assert result.total_volumes == 0
