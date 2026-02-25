"""Data loading edge case tests (Issue #54 — R5.10).

Tests discover_nifti_pairs with empty directories, mismatched counts,
and malformed directory structures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from minivess.data.loader import discover_nifti_pairs


class TestDiscoverNiftiPairsEdgeCases:
    """Test discover_nifti_pairs with edge-case directory structures."""

    def test_empty_imagesTr_directory(self, tmp_path: Path) -> None:
        """Empty imagesTr dir (no .nii.gz files) should raise FileNotFoundError."""
        img_dir = tmp_path / "imagesTr"
        lbl_dir = tmp_path / "labelsTr"
        img_dir.mkdir()
        lbl_dir.mkdir()
        # Both directories exist but are empty
        with pytest.raises(FileNotFoundError, match="No matching"):
            discover_nifti_pairs(tmp_path)

    def test_missing_label_dir(self, tmp_path: Path) -> None:
        """imagesTr exists but labelsTr does not should raise FileNotFoundError."""
        img_dir = tmp_path / "imagesTr"
        img_dir.mkdir()
        # Create a dummy nifti file
        (img_dir / "vol_001.nii.gz").touch()
        # No labelsTr directory
        with pytest.raises(FileNotFoundError):
            discover_nifti_pairs(tmp_path)

    def test_mismatched_image_label_counts(self, tmp_path: Path) -> None:
        """Extra images without matching labels should be silently skipped."""
        img_dir = tmp_path / "imagesTr"
        lbl_dir = tmp_path / "labelsTr"
        img_dir.mkdir()
        lbl_dir.mkdir()

        # 3 images, only 2 matching labels
        (img_dir / "vol_001.nii.gz").touch()
        (img_dir / "vol_002.nii.gz").touch()
        (img_dir / "vol_003.nii.gz").touch()
        (lbl_dir / "vol_001.nii.gz").touch()
        (lbl_dir / "vol_002.nii.gz").touch()
        # vol_003 has no matching label

        pairs = discover_nifti_pairs(tmp_path)
        assert len(pairs) == 2
        names = {Path(p["image"]).name for p in pairs}
        assert "vol_003.nii.gz" not in names

    def test_completely_empty_root(self, tmp_path: Path) -> None:
        """Empty root directory with no recognized layout should raise."""
        with pytest.raises(FileNotFoundError, match="No image directory"):
            discover_nifti_pairs(tmp_path)

    def test_ebrains_layout_empty_seg(self, tmp_path: Path) -> None:
        """EBRAINS raw/ exists but seg/ is empty should raise."""
        raw_dir = tmp_path / "raw"
        seg_dir = tmp_path / "seg"
        raw_dir.mkdir()
        seg_dir.mkdir()
        (raw_dir / "mv01.nii.gz").touch()
        # seg/ is empty — no matching labels
        with pytest.raises(FileNotFoundError, match="No matching"):
            discover_nifti_pairs(tmp_path)

    def test_non_nifti_files_ignored(self, tmp_path: Path) -> None:
        """Non .nii.gz files should be ignored by discover."""
        img_dir = tmp_path / "imagesTr"
        lbl_dir = tmp_path / "labelsTr"
        img_dir.mkdir()
        lbl_dir.mkdir()

        # Only non-nifti files
        (img_dir / "readme.txt").touch()
        (img_dir / "data.csv").touch()
        (lbl_dir / "readme.txt").touch()

        with pytest.raises(FileNotFoundError, match="No matching"):
            discover_nifti_pairs(tmp_path)
