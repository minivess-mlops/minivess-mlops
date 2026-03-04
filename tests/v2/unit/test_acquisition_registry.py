"""Tests for DatasetAcquisitionEntry registry and availability check.

Phase 2, Tasks 2.1–2.2 of flow-data-acquisition-plan.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class TestDatasetAcquisitionEntry:
    """DatasetAcquisitionEntry holds acquisition-specific metadata."""

    def test_entry_has_required_fields(self) -> None:
        from minivess.data.acquisition_registry import DatasetAcquisitionEntry

        entry = DatasetAcquisitionEntry(
            name="test",
            source_url="https://example.com",
            download_method="manual",
            requires_auth=False,
            source_format="nifti",
            manual_instructions="Download from website.",
            expected_checksums=None,
        )
        assert entry.name == "test"
        assert entry.download_method == "manual"

    def test_entry_is_frozen(self) -> None:
        from minivess.data.acquisition_registry import DatasetAcquisitionEntry

        entry = DatasetAcquisitionEntry(
            name="test",
            source_url="https://example.com",
            download_method="manual",
            requires_auth=False,
            source_format="nifti",
            manual_instructions="Download from website.",
            expected_checksums=None,
        )
        try:
            entry.name = "changed"  # type: ignore[misc]
            raise AssertionError("Should be frozen")  # noqa: TRY301
        except AttributeError:
            pass  # Expected — frozen dataclass


class TestAcquisitionRegistry:
    """ACQUISITION_REGISTRY covers all 4 datasets."""

    def test_registry_has_all_datasets(self) -> None:
        from minivess.data.acquisition_registry import ACQUISITION_REGISTRY

        assert "minivess" in ACQUISITION_REGISTRY
        assert "deepvess" in ACQUISITION_REGISTRY
        assert "tubenet_2pm" in ACQUISITION_REGISTRY
        assert "vesselnn" in ACQUISITION_REGISTRY

    def test_registry_length(self) -> None:
        from minivess.data.acquisition_registry import ACQUISITION_REGISTRY

        assert len(ACQUISITION_REGISTRY) == 4

    def test_vesselnn_is_git_clone(self) -> None:
        from minivess.data.acquisition_registry import ACQUISITION_REGISTRY

        entry = ACQUISITION_REGISTRY["vesselnn"]
        assert entry.download_method == "git_clone"
        assert entry.requires_auth is False

    def test_minivess_requires_auth(self) -> None:
        from minivess.data.acquisition_registry import ACQUISITION_REGISTRY

        entry = ACQUISITION_REGISTRY["minivess"]
        assert entry.requires_auth is True
        assert entry.download_method == "manual"

    def test_all_have_instructions(self) -> None:
        from minivess.data.acquisition_registry import ACQUISITION_REGISTRY

        for name, entry in ACQUISITION_REGISTRY.items():
            assert entry.manual_instructions, f"{name} missing instructions"

    def test_all_have_source_format(self) -> None:
        from minivess.data.acquisition_registry import ACQUISITION_REGISTRY

        valid_formats = {"nifti", "tiff", "ome_tiff"}
        for name, entry in ACQUISITION_REGISTRY.items():
            assert entry.source_format in valid_formats, (
                f"{name} has invalid format: {entry.source_format}"
            )


class TestCheckDatasetAvailability:
    """check_dataset_availability checks if files exist on disk."""

    def test_missing_dir_returns_manual_required(self, tmp_path: Path) -> None:
        from minivess.config.acquisition_config import DatasetAcquisitionStatus
        from minivess.data.acquisition_registry import check_dataset_availability

        status = check_dataset_availability("minivess", tmp_path / "nonexistent")
        assert status == DatasetAcquisitionStatus.MANUAL_REQUIRED

    def test_empty_dir_returns_manual_required(self, tmp_path: Path) -> None:
        from minivess.config.acquisition_config import DatasetAcquisitionStatus
        from minivess.data.acquisition_registry import check_dataset_availability

        output_dir = tmp_path / "minivess"
        output_dir.mkdir()
        status = check_dataset_availability("minivess", output_dir)
        assert status == DatasetAcquisitionStatus.MANUAL_REQUIRED

    def test_populated_dir_returns_ready(self, tmp_path: Path) -> None:
        from minivess.config.acquisition_config import DatasetAcquisitionStatus
        from minivess.data.acquisition_registry import check_dataset_availability

        output_dir = tmp_path / "vesselnn"
        (output_dir / "images").mkdir(parents=True)
        (output_dir / "labels").mkdir(parents=True)
        (output_dir / "images" / "vol_001.nii.gz").write_bytes(b"fake")
        (output_dir / "labels" / "vol_001.nii.gz").write_bytes(b"fake")

        status = check_dataset_availability("vesselnn", output_dir)
        assert status == DatasetAcquisitionStatus.READY

    def test_unknown_dataset_returns_failed(self, tmp_path: Path) -> None:
        from minivess.config.acquisition_config import DatasetAcquisitionStatus
        from minivess.data.acquisition_registry import check_dataset_availability

        status = check_dataset_availability("nonexistent", tmp_path)
        assert status == DatasetAcquisitionStatus.FAILED
