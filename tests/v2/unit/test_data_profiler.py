"""Tests for dataset profiler: volume scanning and safe patch computation.

RED phase: Tests for scan_volume, scan_dataset, compute_safe_patch_sizes.
These must FAIL before the implementation exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from minivess.data.profiler import (
    DatasetProfile,
    VolumeStats,
    compute_safe_patch_sizes,
    scan_dataset,
    scan_volume,
)


def _create_nifti(path: Path, shape: tuple, spacing: tuple = (1.0, 1.0, 1.0)) -> None:
    """Helper to create a synthetic NIfTI file."""
    data = np.random.rand(*shape).astype(np.float32)
    affine = np.diag([*spacing, 1.0])
    img = nib.Nifti1Image(data, affine)
    nib.save(img, str(path))


class TestVolumeScanner:
    def test_scan_single_volume_returns_stats(self, tmp_path):
        """scan_volume returns VolumeStats with shape, spacing, intensity range."""
        nifti_path = tmp_path / "test.nii.gz"
        _create_nifti(nifti_path, (64, 64, 16), spacing=(0.5, 0.5, 1.0))
        stats = scan_volume(nifti_path)
        assert isinstance(stats, VolumeStats)
        assert stats.shape == (64, 64, 16)
        assert stats.spacing == pytest.approx((0.5, 0.5, 1.0), abs=1e-5)
        assert stats.size_bytes > 0

    def test_scan_dataset_computes_min_max_shape(self, tmp_path):
        """scan_dataset finds min/max shapes across all volumes."""
        img_dir = tmp_path / "imagesTr"
        lbl_dir = tmp_path / "labelsTr"
        img_dir.mkdir()
        lbl_dir.mkdir()
        # Volume 1: small Z
        _create_nifti(img_dir / "vol01.nii.gz", (64, 64, 5))
        _create_nifti(lbl_dir / "vol01.nii.gz", (64, 64, 5))
        # Volume 2: larger Z
        _create_nifti(img_dir / "vol02.nii.gz", (128, 128, 20))
        _create_nifti(lbl_dir / "vol02.nii.gz", (128, 128, 20))

        profile = scan_dataset(tmp_path)
        assert isinstance(profile, DatasetProfile)
        assert profile.min_shape[2] == 5
        assert profile.max_shape[2] == 20
        assert profile.num_volumes == 2

    def test_anisotropy_detected(self, tmp_path):
        """Flag anisotropic spacing (varies >2x across axes)."""
        nifti_path = tmp_path / "aniso.nii.gz"
        _create_nifti(nifti_path, (64, 64, 16), spacing=(0.5, 0.5, 3.0))
        stats = scan_volume(nifti_path)
        assert stats.is_anisotropic  # spacing ratio > 2x

    def test_safe_patch_z_lte_min_z(self, tmp_path):
        """Patch Z must be <= min volume Z in dataset."""
        img_dir = tmp_path / "imagesTr"
        lbl_dir = tmp_path / "labelsTr"
        img_dir.mkdir()
        lbl_dir.mkdir()
        _create_nifti(img_dir / "vol01.nii.gz", (512, 512, 5))
        _create_nifti(lbl_dir / "vol01.nii.gz", (512, 512, 5))
        _create_nifti(img_dir / "vol02.nii.gz", (512, 512, 30))
        _create_nifti(lbl_dir / "vol02.nii.gz", (512, 512, 30))

        profile = scan_dataset(tmp_path)
        patches = compute_safe_patch_sizes(profile, model_divisor=8)
        assert patches[2] <= 5  # Z dimension constrained by smallest volume

    def test_total_size_bytes_accurate(self, tmp_path):
        """Total dataset size computed from all volumes."""
        nifti_path = tmp_path / "test.nii.gz"
        shape = (64, 64, 16)
        _create_nifti(nifti_path, shape)
        stats = scan_volume(nifti_path)
        expected = 64 * 64 * 16 * 4  # float32
        assert stats.size_bytes == expected


class TestSafePatchComputation:
    def test_dynunet_patch_divisible_by_8(self, tmp_path):
        """DynUNet with 4 levels needs patches divisible by 8."""
        img_dir = tmp_path / "imagesTr"
        lbl_dir = tmp_path / "labelsTr"
        img_dir.mkdir()
        lbl_dir.mkdir()
        _create_nifti(img_dir / "v1.nii.gz", (256, 256, 32))
        _create_nifti(lbl_dir / "v1.nii.gz", (256, 256, 32))
        profile = scan_dataset(tmp_path)
        patches = compute_safe_patch_sizes(profile, model_divisor=8)
        assert patches[0] % 8 == 0
        assert patches[1] % 8 == 0
        assert patches[2] % 8 == 0

    def test_patch_reduced_for_small_z(self, tmp_path):
        """Z=5 → patch_z should be rounded down to nearest divisor (4 for div-8)."""
        img_dir = tmp_path / "imagesTr"
        lbl_dir = tmp_path / "labelsTr"
        img_dir.mkdir()
        lbl_dir.mkdir()
        _create_nifti(img_dir / "v1.nii.gz", (512, 512, 5))
        _create_nifti(lbl_dir / "v1.nii.gz", (512, 512, 5))
        profile = scan_dataset(tmp_path)
        patches = compute_safe_patch_sizes(profile, model_divisor=8)
        # With min_z=5 and divisor=8, largest valid z is 5 - but 5 is not divisible by 8
        # So we need patch_z = floor(5/8)*8 = 0... that's wrong
        # Actually with divisor=8, we need to find the largest multiple of 8 that fits
        # But 5 < 8, so we need to allow smaller divisors or use the value directly
        # The function should handle this gracefully - minimum patch_z should be
        # max(divisor, floor_to_divisor(min_z))  or use a fallback
        assert patches[2] <= 5
        assert patches[2] > 0

    def test_patch_xy_unconstrained_for_large_volumes(self, tmp_path):
        """XY dimensions can be larger when volumes have large X,Y."""
        img_dir = tmp_path / "imagesTr"
        lbl_dir = tmp_path / "labelsTr"
        img_dir.mkdir()
        lbl_dir.mkdir()
        _create_nifti(img_dir / "v1.nii.gz", (512, 512, 32))
        _create_nifti(lbl_dir / "v1.nii.gz", (512, 512, 32))
        profile = scan_dataset(tmp_path)
        patches = compute_safe_patch_sizes(profile, model_divisor=8)
        # With 512x512, patch XY should be reasonable (e.g. 96 or 128)
        assert patches[0] >= 64
        assert patches[1] >= 64

    def test_outlier_spacing_flagged(self, tmp_path):
        """Volumes with outlier spacing (>3x median) should be flagged."""
        img_dir = tmp_path / "imagesTr"
        lbl_dir = tmp_path / "labelsTr"
        img_dir.mkdir()
        lbl_dir.mkdir()
        # Normal spacing
        _create_nifti(img_dir / "v1.nii.gz", (64, 64, 16), spacing=(0.5, 0.5, 0.5))
        _create_nifti(lbl_dir / "v1.nii.gz", (64, 64, 16), spacing=(0.5, 0.5, 0.5))
        _create_nifti(img_dir / "v2.nii.gz", (64, 64, 16), spacing=(0.5, 0.5, 0.5))
        _create_nifti(lbl_dir / "v2.nii.gz", (64, 64, 16), spacing=(0.5, 0.5, 0.5))
        # Outlier spacing (10x normal)
        _create_nifti(img_dir / "v3.nii.gz", (64, 64, 16), spacing=(5.0, 5.0, 5.0))
        _create_nifti(lbl_dir / "v3.nii.gz", (64, 64, 16), spacing=(5.0, 5.0, 5.0))

        profile = scan_dataset(tmp_path)
        assert len(profile.outlier_volumes) > 0
