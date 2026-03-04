"""Tests for TIFF → NIfTI format conversion.

Phase 3, Tasks 3.1–3.2 of flow-data-acquisition-plan.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


class TestConvertTiffToNifti:
    """convert_tiff_to_nifti converts a single TIFF to NIfTI."""

    def test_basic_roundtrip(self, tmp_path: Path) -> None:
        import nibabel as nib
        import tifffile

        from minivess.data.format_conversion import convert_tiff_to_nifti

        # Create a synthetic 3D TIFF
        data = np.random.default_rng(42).integers(0, 255, (10, 32, 32), dtype=np.uint8)
        input_path = tmp_path / "test_vol.tif"
        tifffile.imwrite(str(input_path), data)

        output_path = tmp_path / "test_vol.nii.gz"
        result = convert_tiff_to_nifti(
            input_path=input_path,
            output_path=output_path,
            voxel_spacing=(0.5, 0.5, 1.0),
        )

        assert result == output_path
        assert output_path.exists()

        # Verify the NIfTI content
        nii = nib.load(str(output_path))
        loaded_data = np.asarray(nii.dataobj)
        np.testing.assert_array_equal(loaded_data, data)

    def test_preserves_voxel_spacing(self, tmp_path: Path) -> None:
        import nibabel as nib
        import tifffile

        from minivess.data.format_conversion import convert_tiff_to_nifti

        data = np.zeros((5, 16, 16), dtype=np.uint8)
        input_path = tmp_path / "spacing_test.tif"
        tifffile.imwrite(str(input_path), data)

        output_path = tmp_path / "spacing_test.nii.gz"
        spacing = (0.31, 0.31, 2.0)
        convert_tiff_to_nifti(
            input_path=input_path,
            output_path=output_path,
            voxel_spacing=spacing,
        )

        nii = nib.load(str(output_path))
        pixdim = nii.header.get_zooms()  # type: ignore[union-attr]
        np.testing.assert_allclose(pixdim[:3], spacing, atol=1e-5)

    def test_skip_existing(self, tmp_path: Path) -> None:
        import tifffile

        from minivess.data.format_conversion import convert_tiff_to_nifti

        data = np.zeros((5, 8, 8), dtype=np.uint8)
        input_path = tmp_path / "skip_test.tif"
        tifffile.imwrite(str(input_path), data)

        output_path = tmp_path / "skip_test.nii.gz"
        output_path.write_bytes(b"existing")

        result = convert_tiff_to_nifti(
            input_path=input_path,
            output_path=output_path,
            voxel_spacing=(1.0, 1.0, 1.0),
            skip_existing=True,
        )

        assert result == output_path
        # File should not have been overwritten
        assert output_path.read_bytes() == b"existing"

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        import tifffile

        from minivess.data.format_conversion import convert_tiff_to_nifti

        data = np.zeros((5, 8, 8), dtype=np.uint8)
        input_path = tmp_path / "subdir_test.tif"
        tifffile.imwrite(str(input_path), data)

        output_path = tmp_path / "nested" / "deep" / "output.nii.gz"
        result = convert_tiff_to_nifti(
            input_path=input_path,
            output_path=output_path,
            voxel_spacing=(1.0, 1.0, 1.0),
        )

        assert result == output_path
        assert output_path.exists()


class TestConvertDatasetFormats:
    """convert_dataset_formats batch-converts all TIFFs in a directory."""

    def test_converts_tiff_files(self, tmp_path: Path) -> None:
        import tifffile

        from minivess.data.format_conversion import convert_dataset_formats

        # Set up input directory with TIFF files
        input_dir = tmp_path / "input"
        (input_dir / "images").mkdir(parents=True)
        (input_dir / "labels").mkdir(parents=True)

        data = np.zeros((5, 8, 8), dtype=np.uint8)
        tifffile.imwrite(str(input_dir / "images" / "vol_001.tif"), data)
        tifffile.imwrite(str(input_dir / "labels" / "vol_001.tif"), data)

        output_dir = tmp_path / "output"
        log = convert_dataset_formats(
            dataset_name="deepvess",
            input_dir=input_dir,
            output_dir=output_dir,
            voxel_spacing=(1.0, 1.0, 1.7),
        )

        assert len(log) == 2
        assert (output_dir / "images" / "vol_001.nii.gz").exists()
        assert (output_dir / "labels" / "vol_001.nii.gz").exists()

    def test_skips_nifti_files(self, tmp_path: Path) -> None:
        from minivess.data.format_conversion import convert_dataset_formats

        input_dir = tmp_path / "input"
        (input_dir / "images").mkdir(parents=True)
        (input_dir / "labels").mkdir(parents=True)

        # NIfTI files don't need conversion
        (input_dir / "images" / "vol_001.nii.gz").write_bytes(b"fake")
        (input_dir / "labels" / "vol_001.nii.gz").write_bytes(b"fake")

        output_dir = tmp_path / "output"
        log = convert_dataset_formats(
            dataset_name="minivess",
            input_dir=input_dir,
            output_dir=output_dir,
            voxel_spacing=(1.0, 1.0, 1.0),
        )

        assert len(log) == 0  # Nothing to convert

    def test_missing_input_dir_returns_empty(self, tmp_path: Path) -> None:
        from minivess.data.format_conversion import convert_dataset_formats

        log = convert_dataset_formats(
            dataset_name="test",
            input_dir=tmp_path / "nonexistent",
            output_dir=tmp_path / "output",
            voxel_spacing=(1.0, 1.0, 1.0),
        )

        assert log == []
