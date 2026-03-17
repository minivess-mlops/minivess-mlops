"""Tests for DeepChecks 3D-to-2D slice adapter (T5).

Uses synthetic numpy arrays — no DeepChecks import needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


class TestExtractRepresentativeSlices:
    """extract_representative_slices produces 2D slices from 3D volumes."""

    def test_returns_list_of_2d_arrays(self) -> None:
        """T5-R1: Returns list of 2D arrays."""
        from minivess.validation.deepchecks_3d_adapter import (
            extract_representative_slices,
        )

        volume = (
            np.random.default_rng(42).uniform(0, 1, (64, 64, 32)).astype(np.float32)
        )
        slices = extract_representative_slices(volume)
        assert isinstance(slices, list)
        assert len(slices) > 0
        for s in slices:
            assert s.ndim == 2

    def test_default_strategy_extracts_middle_slice(self) -> None:
        """T5-R2: Default strategy extracts middle axial slice."""
        from minivess.validation.deepchecks_3d_adapter import (
            extract_representative_slices,
        )

        volume = np.zeros((64, 64, 32), dtype=np.float32)
        # Mark the middle axial slice (z=16) with a known value
        volume[:, :, 16] = 1.0

        slices = extract_representative_slices(volume, strategy="middle")
        assert len(slices) >= 1
        # At least one slice should be the middle one
        found_middle = any(np.allclose(s, 1.0) for s in slices)
        assert found_middle

    def test_max_foreground_strategy(self) -> None:
        """T5-R3: strategy='max_foreground' finds slice with most label voxels."""
        from minivess.validation.deepchecks_3d_adapter import (
            extract_representative_slices,
        )

        volume = np.zeros((64, 64, 32), dtype=np.float32)
        label = np.zeros((64, 64, 32), dtype=np.int16)
        # Concentrate foreground in slice z=5
        label[:, :, 5] = 1

        slices = extract_representative_slices(
            volume, label=label, strategy="max_foreground"
        )
        assert len(slices) >= 1

    def test_returned_slices_are_2d(self) -> None:
        """T5-R4: Returned slices have shape (H, W)."""
        from minivess.validation.deepchecks_3d_adapter import (
            extract_representative_slices,
        )

        volume = (
            np.random.default_rng(42).uniform(0, 1, (48, 96, 16)).astype(np.float32)
        )
        slices = extract_representative_slices(volume)
        for s in slices:
            assert s.ndim == 2
            assert s.shape[0] == 48
            assert s.shape[1] == 96

    def test_empty_volume_returns_empty_list(self) -> None:
        """T5-R5: Empty volume returns empty list."""
        from minivess.validation.deepchecks_3d_adapter import (
            extract_representative_slices,
        )

        volume = np.zeros((0, 0, 0), dtype=np.float32)
        slices = extract_representative_slices(volume)
        assert slices == []


class TestBuildDeepChecksDataset:
    """build_deepchecks_dataset creates list of dicts with image/label keys."""

    def test_builds_list_of_dicts(self, tmp_path: Path) -> None:
        """T5-R6: Returns list of dicts with 'image' and 'label' keys."""
        import nibabel as nib

        from minivess.validation.deepchecks_3d_adapter import (
            build_deepchecks_dataset,
        )

        # Create synthetic NIfTI files
        shape = (32, 32, 16)
        img_data = np.random.default_rng(42).uniform(0, 1, shape).astype(np.float32)
        lbl_data = np.zeros(shape, dtype=np.int16)
        lbl_data[10:20, 10:20, 5:10] = 1

        img_path = tmp_path / "img.nii.gz"
        lbl_path = tmp_path / "lbl.nii.gz"
        nib.save(nib.Nifti1Image(img_data, np.eye(4)), str(img_path))
        nib.save(nib.Nifti1Image(lbl_data, np.eye(4)), str(lbl_path))

        pairs = [{"image": str(img_path), "label": str(lbl_path)}]
        dataset = build_deepchecks_dataset(pairs)

        assert isinstance(dataset, list)
        assert len(dataset) > 0
        for item in dataset:
            assert "image" in item
            assert "label" in item
            assert isinstance(item["image"], np.ndarray)
