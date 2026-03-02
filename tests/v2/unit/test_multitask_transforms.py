"""Tests for multi-task target loading in MONAI transform pipeline (T7)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


def _make_test_data(tmp_path: Path) -> tuple[dict[str, str], Path]:
    """Create synthetic NIfTI image + label + precomputed SDF."""
    import nibabel as nib

    affine = np.eye(4)

    # Image (random noise)
    img = np.random.rand(32, 32, 16).astype(np.float32)
    nib.save(nib.Nifti1Image(img, affine), str(tmp_path / "test_img.nii.gz"))

    # Label (binary sphere)
    coords = np.mgrid[:32, :32, :16]
    center = np.array([16, 16, 8])
    dist = np.sqrt(sum((c - center[i]) ** 2 for i, c in enumerate(coords)))
    mask = (dist < 6).astype(np.float32)
    nib.save(nib.Nifti1Image(mask, affine), str(tmp_path / "test_label.nii.gz"))

    # Precomputed SDF
    from scipy.ndimage import distance_transform_edt

    sdf = (distance_transform_edt(mask == 0) - distance_transform_edt(mask > 0)).astype(
        np.float32
    )
    precomp_dir = tmp_path / "precomputed"
    precomp_dir.mkdir()
    nib.save(nib.Nifti1Image(sdf, affine), str(precomp_dir / "test01_sdf.nii.gz"))

    data_dict = {
        "image": str(tmp_path / "test_img.nii.gz"),
        "label": str(tmp_path / "test_label.nii.gz"),
        "volume_id": "test01",
    }
    return data_dict, precomp_dir


class TestBuildTrainTransformsWithAux:
    """Test that build_train_transforms supports auxiliary target loading."""

    def test_aux_targets_loaded_in_output(self, tmp_path: Path) -> None:
        """Training transforms include LoadAuxiliaryTargetsd when aux_configs given."""
        from scipy.ndimage import distance_transform_edt

        from minivess.config.models import DataConfig
        from minivess.data.multitask_targets import AuxTargetConfig
        from minivess.data.transforms import build_train_transforms

        def _compute_sdf(m: np.ndarray) -> np.ndarray:
            return (
                distance_transform_edt(m == 0) - distance_transform_edt(m > 0)
            ).astype(np.float32)

        data_dict, precomp_dir = _make_test_data(tmp_path)
        aux_configs = [
            AuxTargetConfig(name="sdf", suffix="sdf", compute_fn=_compute_sdf),
        ]

        config = DataConfig(
            dataset_name="test",
            data_dir=tmp_path,
            patch_size=(16, 16, 8),
            voxel_spacing=(0, 0, 0),
        )

        transforms = build_train_transforms(
            config, aux_configs=aux_configs, precomputed_dir=precomp_dir
        )
        result = transforms(data_dict)

        # RandCropByPosNegLabeld with num_samples=4 returns a list of patches
        if isinstance(result, list):
            result = result[0]

        # Auxiliary target should be present in the output
        assert "sdf" in result

    def test_no_aux_when_not_configured(self, tmp_path: Path) -> None:
        """Without aux_configs, no extra keys are added."""
        from minivess.config.models import DataConfig
        from minivess.data.transforms import build_train_transforms

        data_dict, _ = _make_test_data(tmp_path)

        config = DataConfig(
            dataset_name="test",
            data_dir=tmp_path,
            patch_size=(16, 16, 8),
            voxel_spacing=(0, 0, 0),
        )

        transforms = build_train_transforms(config)
        result = transforms(data_dict)

        if isinstance(result, list):
            result = result[0]

        # No sdf key should be present
        assert "sdf" not in result


class TestBuildValTransformsWithAux:
    """Test that build_val_transforms supports auxiliary target loading."""

    def test_aux_targets_loaded_in_val(self, tmp_path: Path) -> None:
        """Validation transforms include LoadAuxiliaryTargetsd."""
        from scipy.ndimage import distance_transform_edt

        from minivess.config.models import DataConfig
        from minivess.data.multitask_targets import AuxTargetConfig
        from minivess.data.transforms import build_val_transforms

        def _compute_sdf(m: np.ndarray) -> np.ndarray:
            return (
                distance_transform_edt(m == 0) - distance_transform_edt(m > 0)
            ).astype(np.float32)

        data_dict, precomp_dir = _make_test_data(tmp_path)
        aux_configs = [
            AuxTargetConfig(name="sdf", suffix="sdf", compute_fn=_compute_sdf),
        ]

        config = DataConfig(
            dataset_name="test",
            data_dir=tmp_path,
            patch_size=(16, 16, 8),
            voxel_spacing=(0, 0, 0),
        )

        transforms = build_val_transforms(
            config, aux_configs=aux_configs, precomputed_dir=precomp_dir
        )
        result = transforms(data_dict)

        assert "sdf" in result
