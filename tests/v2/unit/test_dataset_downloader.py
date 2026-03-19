"""Tests for universal dataset downloader (T-A1).

Tests cloud-agnostic download, VesselNN GitHub support,
checksum verification, and resume capability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# T-A1.1: Downloader instantiation and config
# ---------------------------------------------------------------------------


class TestDatasetDownloader:
    """Test DatasetDownloader configuration and validation."""

    def test_downloader_instantiation(self) -> None:
        """Downloader should instantiate with registry datasets."""
        from minivess.data.dataset_downloader import DatasetDownloader

        downloader = DatasetDownloader()
        assert downloader is not None

    def test_available_datasets(self) -> None:
        """Should list all registered external datasets."""
        from minivess.data.dataset_downloader import DatasetDownloader

        downloader = DatasetDownloader()
        available = downloader.available_datasets()
        assert "vesselnn" in available
        assert "deepvess" in available
        # tubenet_2pm excluded: olfactory bulb, different organ, only 1 2PM volume

    def test_get_dataset_info(self) -> None:
        """Should return config for a known dataset."""
        from minivess.data.dataset_downloader import DatasetDownloader

        downloader = DatasetDownloader()
        info = downloader.get_dataset_info("vesselnn")
        assert info.name == "vesselnn"
        assert info.n_volumes == 12
        assert info.source_url.startswith("https://github.com")

    def test_get_dataset_info_unknown_raises(self) -> None:
        """Unknown dataset should raise KeyError."""
        from minivess.data.dataset_downloader import DatasetDownloader

        downloader = DatasetDownloader()
        with pytest.raises(KeyError):
            downloader.get_dataset_info("nonexistent_dataset")


# ---------------------------------------------------------------------------
# T-A1.2: Directory preparation
# ---------------------------------------------------------------------------


class TestDirectoryPreparation:
    """Test download directory setup."""

    def test_prepare_download_dir(self, tmp_path: Path) -> None:
        """Should create images/ and labels/ subdirectories."""
        from minivess.data.dataset_downloader import DatasetDownloader

        downloader = DatasetDownloader()
        target = tmp_path / "vesselnn"
        result = downloader.prepare_download_dir("vesselnn", target)

        assert result.is_dir()
        assert (result / "images").is_dir()
        assert (result / "labels").is_dir()

    def test_prepare_download_dir_idempotent(self, tmp_path: Path) -> None:
        """Calling twice should not fail."""
        from minivess.data.dataset_downloader import DatasetDownloader

        downloader = DatasetDownloader()
        target = tmp_path / "vesselnn"
        downloader.prepare_download_dir("vesselnn", target)
        downloader.prepare_download_dir("vesselnn", target)
        assert (target / "images").is_dir()


# ---------------------------------------------------------------------------
# T-A1.3: Download status tracking
# ---------------------------------------------------------------------------


class TestDownloadStatus:
    """Test download status and resume."""

    def test_download_status_not_started(self, tmp_path: Path) -> None:
        """New directory should report not downloaded."""
        from minivess.data.dataset_downloader import DatasetDownloader

        downloader = DatasetDownloader()
        target = tmp_path / "vesselnn"
        downloader.prepare_download_dir("vesselnn", target)

        status = downloader.get_download_status("vesselnn", target)
        assert status["complete"] is False
        assert status["n_images"] == 0

    def test_download_status_with_files(self, tmp_path: Path) -> None:
        """Directory with files should report correct counts."""
        from minivess.data.dataset_downloader import DatasetDownloader

        downloader = DatasetDownloader()
        target = tmp_path / "vesselnn"
        downloader.prepare_download_dir("vesselnn", target)

        # Simulate some downloaded files
        for i in range(3):
            (target / "images" / f"vol_{i:03d}.nii.gz").write_bytes(b"fake")
            (target / "labels" / f"vol_{i:03d}.nii.gz").write_bytes(b"fake")

        status = downloader.get_download_status("vesselnn", target)
        assert status["n_images"] == 3
        assert status["n_labels"] == 3


# ---------------------------------------------------------------------------
# T-A1.4: Checksum verification
# ---------------------------------------------------------------------------


class TestChecksumVerification:
    """Test file integrity verification."""

    def test_compute_file_checksum(self, tmp_path: Path) -> None:
        """Should compute SHA-256 checksum of a file."""
        from minivess.data.dataset_downloader import compute_file_checksum

        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"hello world")

        checksum = compute_file_checksum(test_file)
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA-256 hex digest

    def test_checksum_deterministic(self, tmp_path: Path) -> None:
        """Same content should produce same checksum."""
        from minivess.data.dataset_downloader import compute_file_checksum

        f1 = tmp_path / "f1.bin"
        f2 = tmp_path / "f2.bin"
        f1.write_bytes(b"identical content")
        f2.write_bytes(b"identical content")

        assert compute_file_checksum(f1) == compute_file_checksum(f2)

    def test_checksum_different_content(self, tmp_path: Path) -> None:
        """Different content should produce different checksums."""
        from minivess.data.dataset_downloader import compute_file_checksum

        f1 = tmp_path / "f1.bin"
        f2 = tmp_path / "f2.bin"
        f1.write_bytes(b"content A")
        f2.write_bytes(b"content B")

        assert compute_file_checksum(f1) != compute_file_checksum(f2)


# ---------------------------------------------------------------------------
# T-A1.5: Download manifest
# ---------------------------------------------------------------------------


class TestDownloadManifest:
    """Test download manifest generation and validation."""

    def test_generate_manifest(self, tmp_path: Path) -> None:
        """Should generate a manifest JSON with checksums."""
        from minivess.data.dataset_downloader import DatasetDownloader

        downloader = DatasetDownloader()
        target = tmp_path / "vesselnn"
        downloader.prepare_download_dir("vesselnn", target)

        # Create fake files
        for i in range(2):
            (target / "images" / f"vol_{i:03d}.nii.gz").write_bytes(f"img{i}".encode())
            (target / "labels" / f"vol_{i:03d}.nii.gz").write_bytes(f"lbl{i}".encode())

        manifest = downloader.generate_manifest(target)
        assert "files" in manifest
        assert len(manifest["files"]) == 4  # 2 images + 2 labels
        for entry in manifest["files"]:
            assert "path" in entry
            assert "checksum" in entry
            assert "size" in entry

    def test_verify_manifest(self, tmp_path: Path) -> None:
        """Manifest verification should pass for correct files."""
        from minivess.data.dataset_downloader import DatasetDownloader

        downloader = DatasetDownloader()
        target = tmp_path / "vesselnn"
        downloader.prepare_download_dir("vesselnn", target)

        (target / "images" / "vol_000.nii.gz").write_bytes(b"image data")
        (target / "labels" / "vol_000.nii.gz").write_bytes(b"label data")

        manifest = downloader.generate_manifest(target)
        errors = downloader.verify_manifest(target, manifest)
        assert len(errors) == 0


# ---------------------------------------------------------------------------
# T-A1.6: Cloud upload abstraction
# ---------------------------------------------------------------------------


class TestCloudUploadAbstraction:
    """Test cloud-agnostic upload interface."""

    def test_upload_target_gcs(self) -> None:
        """GCS upload target should parse correctly."""
        from minivess.data.dataset_downloader import parse_upload_target

        target = parse_upload_target("gs://minivess-mlops-dvc-data/external/vesselnn")
        assert target["provider"] == "gcs"
        assert target["bucket"] == "minivess-mlops-dvc-data"
        assert target["prefix"] == "external/vesselnn"

    def test_upload_target_s3(self) -> None:
        """S3 upload target should parse correctly."""
        from minivess.data.dataset_downloader import parse_upload_target

        target = parse_upload_target("s3://bucket-name/path/to/data")
        assert target["provider"] == "s3"
        assert target["bucket"] == "bucket-name"
        assert target["prefix"] == "path/to/data"

    def test_upload_target_local(self) -> None:
        """Local path should be recognized."""
        from minivess.data.dataset_downloader import parse_upload_target

        target = parse_upload_target("/data/external/vesselnn")
        assert target["provider"] == "local"
        assert target["path"] == "/data/external/vesselnn"

    def test_upload_target_invalid_raises(self) -> None:
        """Invalid scheme should raise ValueError."""
        from minivess.data.dataset_downloader import parse_upload_target

        with pytest.raises(ValueError, match="Unsupported"):
            parse_upload_target("ftp://some-server/path")
