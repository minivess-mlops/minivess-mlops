"""Tests for the MiniVess download and reorganisation script.

Tests the offline reorganisation logic (ZIP extraction → loader-expected
directory structure). Network download is NOT tested here — only the
file reorganisation and integrity checks.
"""

from __future__ import annotations

from pathlib import Path

import pytest


class TestReorganiseEbrainsData:
    """Test reorganising EBRAINS layout (raw/seg/json) into loader layout."""

    def test_reorganise_creates_loader_structure(self, tmp_path: Path) -> None:
        """Given extracted EBRAINS dirs, reorganise into imagesTr/labelsTr."""
        from tests.v2.fixtures.synthetic_nifti import (
            create_synthetic_nifti_dataset_ebrains,
        )

        # Simulate extracted ZIP: raw/ + seg/ + json/
        ebrains_dir = create_synthetic_nifti_dataset_ebrains(
            tmp_path / "extracted", n_volumes=3
        )

        from scripts.download_minivess import reorganise_ebrains_to_loader

        output_dir = tmp_path / "output"
        reorganise_ebrains_to_loader(ebrains_dir, output_dir)

        assert (output_dir / "imagesTr").exists()
        assert (output_dir / "labelsTr").exists()
        assert len(list((output_dir / "imagesTr").glob("*.nii.gz"))) == 3
        assert len(list((output_dir / "labelsTr").glob("*.nii.gz"))) == 3

    def test_reorganise_matching_filenames(self, tmp_path: Path) -> None:
        """Image and label files should have matching names after reorganise."""
        from tests.v2.fixtures.synthetic_nifti import (
            create_synthetic_nifti_dataset_ebrains,
        )

        ebrains_dir = create_synthetic_nifti_dataset_ebrains(
            tmp_path / "extracted", n_volumes=2
        )

        from scripts.download_minivess import reorganise_ebrains_to_loader

        output_dir = tmp_path / "output"
        reorganise_ebrains_to_loader(ebrains_dir, output_dir)

        img_names = sorted(f.name for f in (output_dir / "imagesTr").glob("*.nii.gz"))
        lbl_names = sorted(f.name for f in (output_dir / "labelsTr").glob("*.nii.gz"))

        assert img_names == lbl_names

    def test_reorganise_preserves_metadata(self, tmp_path: Path) -> None:
        """JSON metadata files should be copied to metadata/ dir."""
        from tests.v2.fixtures.synthetic_nifti import (
            create_synthetic_nifti_dataset_ebrains,
        )

        ebrains_dir = create_synthetic_nifti_dataset_ebrains(
            tmp_path / "extracted", n_volumes=2
        )
        # Add json dir with metadata
        json_dir = ebrains_dir / "json"
        json_dir.mkdir(exist_ok=True)
        (json_dir / "mv01.json").write_text('{"z_slices": 22}', encoding="utf-8")
        (json_dir / "mv02.json").write_text('{"z_slices": 30}', encoding="utf-8")

        from scripts.download_minivess import reorganise_ebrains_to_loader

        output_dir = tmp_path / "output"
        reorganise_ebrains_to_loader(ebrains_dir, output_dir)

        assert (output_dir / "metadata").exists()
        assert len(list((output_dir / "metadata").glob("*.json"))) == 2


class TestDownloadIdempotent:
    """Download/reorganise should be a no-op when data already exists."""

    def test_idempotent_when_data_exists(self, tmp_path: Path) -> None:
        from tests.v2.fixtures.synthetic_nifti import create_synthetic_nifti_dataset

        # Pre-populate with valid data
        data_dir = create_synthetic_nifti_dataset(tmp_path / "data", n_volumes=3)

        from scripts.download_minivess import is_dataset_ready

        assert is_dataset_ready(data_dir, expected_volumes=3) is True

    def test_not_ready_when_empty(self, tmp_path: Path) -> None:
        from scripts.download_minivess import is_dataset_ready

        assert is_dataset_ready(tmp_path, expected_volumes=3) is False

    def test_not_ready_when_incomplete(self, tmp_path: Path) -> None:
        from tests.v2.fixtures.synthetic_nifti import create_synthetic_nifti_dataset

        data_dir = create_synthetic_nifti_dataset(tmp_path / "data", n_volumes=2)

        from scripts.download_minivess import is_dataset_ready

        assert is_dataset_ready(data_dir, expected_volumes=3) is False


class TestExtractFromZip:
    """Test extraction from local ZIP file."""

    def test_extract_zip_creates_expected_dirs(self, tmp_path: Path) -> None:
        """Create a mock ZIP and verify extraction."""
        import zipfile

        import nibabel as nib
        import numpy as np

        # Create a minimal ZIP mimicking EBRAINS structure
        zip_path = tmp_path / "minivess.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            affine = np.eye(4)
            for i in range(1, 3):
                # Raw image
                img = nib.Nifti1Image(
                    np.random.rand(8, 8, 4).astype(np.float32), affine
                )
                img_bytes = nib.save(img, str(tmp_path / f"tmp_img_{i}.nii.gz"))
                zf.write(tmp_path / f"tmp_img_{i}.nii.gz", f"raw/mv{i:02d}.nii.gz")

                # Segmentation
                lbl = nib.Nifti1Image(
                    np.random.randint(0, 2, (8, 8, 4), dtype=np.uint8), affine
                )
                nib.save(lbl, str(tmp_path / f"tmp_lbl_{i}.nii.gz"))
                zf.write(
                    tmp_path / f"tmp_lbl_{i}.nii.gz", f"seg/mv{i:02d}_y.nii.gz"
                )

        from scripts.download_minivess import extract_and_reorganise

        output_dir = tmp_path / "output"
        extract_and_reorganise(zip_path, output_dir)

        assert (output_dir / "imagesTr").exists()
        assert (output_dir / "labelsTr").exists()
        assert len(list((output_dir / "imagesTr").glob("*.nii.gz"))) == 2
        assert len(list((output_dir / "labelsTr").glob("*.nii.gz"))) == 2
