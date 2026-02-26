"""Tests for the debug dataset generator.

RED phase: These tests verify that create_debug_dataset() produces valid
synthetic NIfTI volumes in EBRAINS layout (raw/ + seg/) suitable for
debugging/development while waiting for real external test sets.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


class TestCreateDebugDataset:
    """Verify create_debug_dataset creates valid EBRAINS-layout NIfTI data."""

    def test_create_debug_dataset_creates_4_volumes(self, tmp_path: Path) -> None:
        """Default call creates exactly 4 image/label pairs."""
        from minivess.data.debug_dataset import create_debug_dataset

        output_dir = tmp_path / "debug_ds"
        result = create_debug_dataset(output_dir)

        raw_files = sorted((result / "raw").glob("*.nii.gz"))
        seg_files = sorted((result / "seg").glob("*.nii.gz"))

        assert len(raw_files) == 4
        assert len(seg_files) == 4

    def test_debug_volume_shapes_valid(self, tmp_path: Path) -> None:
        """Each volume is a valid 3D NIfTI with the requested shape."""
        from minivess.data.debug_dataset import create_debug_dataset

        shape = (48, 48, 12)
        output_dir = tmp_path / "debug_ds"
        result = create_debug_dataset(output_dir, volume_shape=shape)

        for nii_path in (result / "raw").glob("*.nii.gz"):
            img = nib.load(str(nii_path))
            data = img.get_fdata()
            assert data.ndim == 3, f"Expected 3D, got {data.ndim}D for {nii_path.name}"
            assert data.shape == shape, f"Expected {shape}, got {data.shape}"

        for nii_path in (result / "seg").glob("*.nii.gz"):
            lbl = nib.load(str(nii_path))
            data = lbl.get_fdata()
            assert data.ndim == 3
            assert data.shape == shape

    def test_debug_labels_are_binary(self, tmp_path: Path) -> None:
        """Label volumes contain only values 0 and 1."""
        from minivess.data.debug_dataset import create_debug_dataset

        output_dir = tmp_path / "debug_ds"
        result = create_debug_dataset(output_dir)

        for nii_path in (result / "seg").glob("*.nii.gz"):
            lbl = nib.load(str(nii_path))
            data = lbl.get_fdata()
            unique_vals = set(np.unique(data).astype(int))
            assert unique_vals <= {0, 1}, (
                f"Label {nii_path.name} has non-binary values: {unique_vals}"
            )

    def test_debug_dataset_has_image_and_label_keys(self, tmp_path: Path) -> None:
        """Debug dataset is discoverable by discover_nifti_pairs (EBRAINS layout)."""
        from minivess.data.debug_dataset import create_debug_dataset
        from minivess.data.loader import discover_nifti_pairs

        output_dir = tmp_path / "debug_ds"
        result = create_debug_dataset(output_dir, n_volumes=3)

        pairs = discover_nifti_pairs(result)
        assert len(pairs) == 3
        for pair in pairs:
            assert "image" in pair
            assert "label" in pair
            assert Path(pair["image"]).exists()
            assert Path(pair["label"]).exists()

    def test_debug_dataset_is_deterministic(self, tmp_path: Path) -> None:
        """Same seed produces identical data; different seeds differ."""
        from minivess.data.debug_dataset import create_debug_dataset

        dir_a = tmp_path / "run_a"
        dir_b = tmp_path / "run_b"
        dir_c = tmp_path / "run_c"

        create_debug_dataset(dir_a, seed=42)
        create_debug_dataset(dir_b, seed=42)
        create_debug_dataset(dir_c, seed=99)

        # Same seed -> same data
        for fname in sorted((dir_a / "raw").glob("*.nii.gz")):
            data_a = nib.load(str(fname)).get_fdata()
            data_b = nib.load(str(dir_b / "raw" / fname.name)).get_fdata()
            np.testing.assert_array_equal(data_a, data_b)

        # Different seed -> different data
        first_file = sorted((dir_a / "raw").glob("*.nii.gz"))[0]
        data_a = nib.load(str(first_file)).get_fdata()
        data_c = nib.load(str(dir_c / "raw" / first_file.name)).get_fdata()
        assert not np.array_equal(data_a, data_c), "Different seeds should produce different data"

    def test_debug_dataset_directory_structure(self, tmp_path: Path) -> None:
        """Creates raw/ and seg/ directories in the output path."""
        from minivess.data.debug_dataset import create_debug_dataset

        output_dir = tmp_path / "debug_ds"
        result = create_debug_dataset(output_dir, n_volumes=2)

        assert result == output_dir
        assert (output_dir / "raw").is_dir()
        assert (output_dir / "seg").is_dir()

        # Verify naming convention matches EBRAINS pattern
        raw_names = sorted(f.name for f in (output_dir / "raw").glob("*.nii.gz"))
        seg_names = sorted(f.name for f in (output_dir / "seg").glob("*.nii.gz"))

        assert len(raw_names) == 2
        assert len(seg_names) == 2

        # Label names should have _y suffix matching image stems
        for raw_name, seg_name in zip(raw_names, seg_names, strict=True):
            stem = raw_name.replace(".nii.gz", "")
            expected_seg = f"{stem}_y.nii.gz"
            assert seg_name == expected_seg, (
                f"Expected seg name '{expected_seg}', got '{seg_name}'"
            )
