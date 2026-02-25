"""Tests for NIfTI file-based data loading.

RED phase: These tests exercise discover_nifti_pairs() and MONAI loaders
with actual NIfTI files on disk (synthetic), covering gaps that
tensor-only fixtures miss.
"""

from __future__ import annotations

from pathlib import Path

import pytest


class TestSyntheticNiftiCreation:
    """Verify the synthetic NIfTI fixture creates valid files."""

    def test_create_synthetic_nifti_creates_files(self, tmp_path: Path) -> None:
        from tests.v2.fixtures.synthetic_nifti import create_synthetic_nifti_dataset

        data_dir = create_synthetic_nifti_dataset(
            tmp_path, n_volumes=3, spatial_size=(16, 16, 8)
        )

        img_dir = data_dir / "imagesTr"
        lbl_dir = data_dir / "labelsTr"

        assert img_dir.exists()
        assert lbl_dir.exists()
        assert len(list(img_dir.glob("*.nii.gz"))) == 3
        assert len(list(lbl_dir.glob("*.nii.gz"))) == 3

    def test_synthetic_nifti_matching_names(self, tmp_path: Path) -> None:
        from tests.v2.fixtures.synthetic_nifti import create_synthetic_nifti_dataset

        data_dir = create_synthetic_nifti_dataset(tmp_path, n_volumes=2)

        img_names = sorted(f.name for f in (data_dir / "imagesTr").glob("*.nii.gz"))
        lbl_names = sorted(f.name for f in (data_dir / "labelsTr").glob("*.nii.gz"))

        assert img_names == lbl_names

    def test_synthetic_nifti_has_foreground(self, tmp_path: Path) -> None:
        import nibabel as nib
        import numpy as np

        from tests.v2.fixtures.synthetic_nifti import create_synthetic_nifti_dataset

        data_dir = create_synthetic_nifti_dataset(
            tmp_path, n_volumes=1, spatial_size=(16, 16, 8)
        )

        lbl_file = next((data_dir / "labelsTr").glob("*.nii.gz"))
        lbl_data = nib.load(lbl_file).get_fdata()

        # Labels should have some foreground voxels (not all zeros)
        assert np.any(lbl_data > 0), "Label should contain foreground voxels"
        # But not all foreground (sparse vessels)
        foreground_ratio = np.mean(lbl_data > 0)
        assert foreground_ratio < 0.5, f"Too much foreground: {foreground_ratio:.2%}"


class TestDiscoverNiftiPairs:
    """Test discover_nifti_pairs() with real NIfTI files on disk."""

    def test_discover_nifti_pairs_synthetic(self, tmp_path: Path) -> None:
        from minivess.data.loader import discover_nifti_pairs
        from tests.v2.fixtures.synthetic_nifti import create_synthetic_nifti_dataset

        data_dir = create_synthetic_nifti_dataset(tmp_path, n_volumes=3)
        pairs = discover_nifti_pairs(data_dir)

        assert len(pairs) == 3
        for pair in pairs:
            assert "image" in pair
            assert "label" in pair
            assert Path(pair["image"]).exists()
            assert Path(pair["label"]).exists()

    def test_discover_ebrains_layout(self, tmp_path: Path) -> None:
        """EBRAINS uses raw/ + seg/ with _y suffix labels."""
        from minivess.data.loader import discover_nifti_pairs
        from tests.v2.fixtures.synthetic_nifti import (
            create_synthetic_nifti_dataset_ebrains,
        )

        data_dir = create_synthetic_nifti_dataset_ebrains(tmp_path, n_volumes=3)
        pairs = discover_nifti_pairs(data_dir)

        assert len(pairs) == 3

    def test_discover_suffix_stripping(self, tmp_path: Path) -> None:
        """Labels with _y suffix should match images without suffix."""
        from minivess.data.loader import discover_nifti_pairs
        from tests.v2.fixtures.synthetic_nifti import (
            create_synthetic_nifti_dataset_ebrains,
        )

        data_dir = create_synthetic_nifti_dataset_ebrains(tmp_path, n_volumes=2)
        pairs = discover_nifti_pairs(data_dir)

        for pair in pairs:
            img_stem = Path(pair["image"]).name.replace(".nii.gz", "")
            lbl_stem = Path(pair["label"]).name.replace(".nii.gz", "")
            # Label should have _y suffix that was matched to image
            assert lbl_stem == f"{img_stem}_y"

    def test_discover_backward_compatible(self, tmp_path: Path) -> None:
        """imagesTr/ + labelsTr/ layout still works."""
        from minivess.data.loader import discover_nifti_pairs
        from tests.v2.fixtures.synthetic_nifti import create_synthetic_nifti_dataset

        data_dir = create_synthetic_nifti_dataset(tmp_path, n_volumes=2)
        pairs = discover_nifti_pairs(data_dir)

        assert len(pairs) == 2

    def test_discover_empty_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            from minivess.data.loader import discover_nifti_pairs

            discover_nifti_pairs(tmp_path)


class TestMonaiLoaderWithNifti:
    """Test MONAI CacheDataset + DataLoader with NIfTI files."""

    def test_build_train_loader_from_nifti_files(self, tmp_path: Path) -> None:
        from minivess.config.models import DataConfig
        from minivess.data.loader import build_train_loader, discover_nifti_pairs
        from tests.v2.fixtures.synthetic_nifti import create_synthetic_nifti_dataset

        data_dir = create_synthetic_nifti_dataset(
            tmp_path, n_volumes=2, spatial_size=(32, 32, 16)
        )
        pairs = discover_nifti_pairs(data_dir)

        config = DataConfig(
            dataset_name="test",
            data_dir=data_dir,
            patch_size=(16, 16, 8),
            voxel_spacing=(1.0, 1.0, 1.0),
            num_workers=0,
        )

        loader = build_train_loader(pairs, config, batch_size=1, cache_rate=1.0)
        batch = next(iter(loader))

        assert "image" in batch
        assert "label" in batch
        # After RandCropByPosNegLabeld with num_samples=4, batch dim grows
        assert batch["image"].ndim == 5  # (B, C, D, H, W)

    def test_build_val_loader_from_nifti_files(self, tmp_path: Path) -> None:
        from minivess.config.models import DataConfig
        from minivess.data.loader import build_val_loader, discover_nifti_pairs
        from tests.v2.fixtures.synthetic_nifti import create_synthetic_nifti_dataset

        data_dir = create_synthetic_nifti_dataset(
            tmp_path, n_volumes=2, spatial_size=(32, 32, 16)
        )
        pairs = discover_nifti_pairs(data_dir)

        config = DataConfig(
            dataset_name="test",
            data_dir=data_dir,
            patch_size=(16, 16, 8),
            voxel_spacing=(1.0, 1.0, 1.0),
            num_workers=0,
        )

        loader = build_val_loader(pairs, config, cache_rate=1.0)
        batch = next(iter(loader))

        assert "image" in batch
        assert "label" in batch
        assert batch["image"].ndim == 5  # (B, C, D, H, W)
