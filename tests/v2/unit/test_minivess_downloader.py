"""Tests for MiniVess downloader module (extracted from scripts/).

Phase 1, Task T-ACQ.1.1 of overnight-child-01-acquisition.xml.
Validates that MiniVess download logic is importable as a module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class TestMinivessDownloaderImportable:
    """downloader.py is an importable module, not just a script."""

    def test_import_is_dataset_ready(self) -> None:
        from minivess.data.downloader import is_dataset_ready

        assert callable(is_dataset_ready)

    def test_import_reorganise_ebrains_to_loader(self) -> None:
        from minivess.data.downloader import reorganise_ebrains_to_loader

        assert callable(reorganise_ebrains_to_loader)

    def test_import_extract_and_reorganise(self) -> None:
        from minivess.data.downloader import extract_and_reorganise

        assert callable(extract_and_reorganise)

    def test_import_download_minivess(self) -> None:
        from minivess.data.downloader import download_minivess

        assert callable(download_minivess)


class TestIsDatasetReady:
    """is_dataset_ready checks both imagesTr/labelsTr and raw/seg layouts."""

    def test_empty_dir_is_not_ready(self, tmp_path: Path) -> None:
        from minivess.data.downloader import is_dataset_ready

        assert is_dataset_ready(tmp_path) is False

    def test_images_tr_layout_ready(self, tmp_path: Path) -> None:
        from minivess.data.downloader import is_dataset_ready

        img_dir = tmp_path / "imagesTr"
        lbl_dir = tmp_path / "labelsTr"
        img_dir.mkdir()
        lbl_dir.mkdir()
        for i in range(70):
            (img_dir / f"mv{i + 1:02d}.nii.gz").write_bytes(b"fake")
            (lbl_dir / f"mv{i + 1:02d}.nii.gz").write_bytes(b"fake")

        assert is_dataset_ready(tmp_path) is True

    def test_raw_seg_layout_ready(self, tmp_path: Path) -> None:
        from minivess.data.downloader import is_dataset_ready

        raw_dir = tmp_path / "raw"
        seg_dir = tmp_path / "seg"
        raw_dir.mkdir()
        seg_dir.mkdir()
        for i in range(70):
            (raw_dir / f"mv{i + 1:02d}.nii.gz").write_bytes(b"fake")
            (seg_dir / f"mv{i + 1:02d}_y.nii.gz").write_bytes(b"fake")

        assert is_dataset_ready(tmp_path) is True

    def test_partial_dataset_is_not_ready(self, tmp_path: Path) -> None:
        from minivess.data.downloader import is_dataset_ready

        img_dir = tmp_path / "imagesTr"
        lbl_dir = tmp_path / "labelsTr"
        img_dir.mkdir()
        lbl_dir.mkdir()
        # Only 10 volumes — not enough
        for i in range(10):
            (img_dir / f"mv{i + 1:02d}.nii.gz").write_bytes(b"fake")
            (lbl_dir / f"mv{i + 1:02d}.nii.gz").write_bytes(b"fake")

        assert is_dataset_ready(tmp_path) is False

    def test_custom_expected_volumes(self, tmp_path: Path) -> None:
        from minivess.data.downloader import is_dataset_ready

        img_dir = tmp_path / "imagesTr"
        lbl_dir = tmp_path / "labelsTr"
        img_dir.mkdir()
        lbl_dir.mkdir()
        for i in range(5):
            (img_dir / f"mv{i + 1:02d}.nii.gz").write_bytes(b"fake")
            (lbl_dir / f"mv{i + 1:02d}.nii.gz").write_bytes(b"fake")

        assert is_dataset_ready(tmp_path, expected_volumes=5) is True


class TestReorganiseEbrainsToLoader:
    """reorganise_ebrains_to_loader converts raw/seg → imagesTr/labelsTr."""

    def test_creates_imagestir_labelstr(self, tmp_path: Path) -> None:
        from minivess.data.downloader import reorganise_ebrains_to_loader

        ebrains_dir = tmp_path / "ebrains"
        raw_dir = ebrains_dir / "raw"
        seg_dir = ebrains_dir / "seg"
        raw_dir.mkdir(parents=True)
        seg_dir.mkdir(parents=True)

        (raw_dir / "mv01.nii.gz").write_bytes(b"image_data")
        (seg_dir / "mv01_y.nii.gz").write_bytes(b"label_data")

        output_dir = tmp_path / "output"
        reorganise_ebrains_to_loader(ebrains_dir, output_dir)

        assert (output_dir / "imagesTr" / "mv01.nii.gz").exists()
        assert (output_dir / "labelsTr" / "mv01.nii.gz").exists()

    def test_strips_y_suffix_from_labels(self, tmp_path: Path) -> None:
        from minivess.data.downloader import reorganise_ebrains_to_loader

        ebrains_dir = tmp_path / "ebrains"
        (ebrains_dir / "raw").mkdir(parents=True)
        (ebrains_dir / "seg").mkdir(parents=True)
        (ebrains_dir / "raw" / "mv05.nii.gz").write_bytes(b"img")
        (ebrains_dir / "seg" / "mv05_y.nii.gz").write_bytes(b"lbl")

        output_dir = tmp_path / "output"
        reorganise_ebrains_to_loader(ebrains_dir, output_dir)

        # Label renamed: mv05_y.nii.gz → mv05.nii.gz
        assert (output_dir / "labelsTr" / "mv05.nii.gz").exists()
        assert not (output_dir / "labelsTr" / "mv05_y.nii.gz").exists()

    def test_raises_on_missing_raw_dir(self, tmp_path: Path) -> None:
        import pytest

        from minivess.data.downloader import reorganise_ebrains_to_loader

        with pytest.raises(FileNotFoundError):
            reorganise_ebrains_to_loader(tmp_path, tmp_path / "out")


class TestDownloadMinivess:
    """download_minivess is the high-level entry point."""

    def test_skips_if_ready(self, tmp_path: Path) -> None:
        from minivess.data.downloader import download_minivess

        # Pre-populate with 70 volumes
        img_dir = tmp_path / "imagesTr"
        lbl_dir = tmp_path / "labelsTr"
        img_dir.mkdir()
        lbl_dir.mkdir()
        for i in range(70):
            (img_dir / f"mv{i + 1:02d}.nii.gz").write_bytes(b"fake")
            (lbl_dir / f"mv{i + 1:02d}.nii.gz").write_bytes(b"fake")

        # Should return without error (idempotent)
        result = download_minivess(tmp_path)
        assert result is True  # Already ready


class TestMinivessDownloaderRegistered:
    """MiniVess downloader is registered in the dispatch dict."""

    def test_minivess_has_downloader(self) -> None:
        from minivess.data.downloaders import get_downloader

        downloader = get_downloader("minivess")
        # minivess needs manual download (EBRAINS auth) — should return None
        # OR return a downloader if we have one for local ZIP extraction
        # The key point: get_downloader("minivess") should not raise
        assert downloader is None or callable(downloader)
