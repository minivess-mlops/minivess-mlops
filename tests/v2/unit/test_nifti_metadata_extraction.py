"""Tests for NIfTI metadata extraction task (T2).

Uses synthetic NIfTI files created via nibabel — no real data required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path


def _create_synthetic_nifti(
    tmp_path: Path,
    name: str,
    shape: tuple[int, int, int] = (64, 64, 32),
    voxel_spacing: tuple[float, float, float] = (0.5, 0.5, 1.0),
    *,
    is_label: bool = False,
) -> Path:
    """Create a synthetic NIfTI file for testing."""
    import nibabel as nib

    if is_label:
        data = np.zeros(shape, dtype=np.int16)
        # Create a small foreground region
        data[20:40, 20:40, 10:20] = 1
    else:
        rng = np.random.default_rng(42)
        data = rng.uniform(0.0, 1000.0, size=shape).astype(np.float32)

    affine = np.diag([*voxel_spacing, 1.0])
    img = nib.Nifti1Image(data, affine)
    filepath = tmp_path / f"{name}.nii.gz"
    nib.save(img, str(filepath))
    return filepath


class TestExtractNiftiMetadataTask:
    """extract_nifti_metadata_task produces NIfTI metadata DataFrame."""

    def test_returns_dataframe(self, tmp_path: Path) -> None:
        """T2-R1: Returns a pd.DataFrame."""
        from minivess.orchestration.flows.data_flow import (
            extract_nifti_metadata_task,
        )

        img_path = _create_synthetic_nifti(tmp_path, "img_001")
        lbl_path = _create_synthetic_nifti(tmp_path, "lbl_001", is_label=True)
        pairs = [{"image": str(img_path), "label": str(lbl_path)}]

        result = extract_nifti_metadata_task(pairs)
        assert isinstance(result, pd.DataFrame)

    def test_has_all_schema_columns(self, tmp_path: Path) -> None:
        """T2-R2: DataFrame has all NiftiMetadataSchema columns."""
        from minivess.orchestration.flows.data_flow import (
            extract_nifti_metadata_task,
        )

        img_path = _create_synthetic_nifti(tmp_path, "img_001")
        lbl_path = _create_synthetic_nifti(tmp_path, "lbl_001", is_label=True)
        pairs = [{"image": str(img_path), "label": str(lbl_path)}]

        df = extract_nifti_metadata_task(pairs)
        expected_columns = {
            "file_path",
            "shape_x",
            "shape_y",
            "shape_z",
            "voxel_spacing_x",
            "voxel_spacing_y",
            "voxel_spacing_z",
            "intensity_min",
            "intensity_max",
            "num_foreground_voxels",
            "has_valid_affine",
        }
        assert expected_columns.issubset(set(df.columns))

    def test_shape_values_match_nifti(self, tmp_path: Path) -> None:
        """T2-R3: Shape values match actual NIfTI dimensions."""
        from minivess.orchestration.flows.data_flow import (
            extract_nifti_metadata_task,
        )

        shape = (48, 96, 16)
        img_path = _create_synthetic_nifti(tmp_path, "img_001", shape=shape)
        lbl_path = _create_synthetic_nifti(
            tmp_path, "lbl_001", shape=shape, is_label=True
        )
        pairs = [{"image": str(img_path), "label": str(lbl_path)}]

        df = extract_nifti_metadata_task(pairs)
        assert df.iloc[0]["shape_x"] == 48
        assert df.iloc[0]["shape_y"] == 96
        assert df.iloc[0]["shape_z"] == 16

    def test_voxel_spacing_matches_affine(self, tmp_path: Path) -> None:
        """T2-R4: Voxel spacing values match NIfTI affine."""
        from minivess.orchestration.flows.data_flow import (
            extract_nifti_metadata_task,
        )

        spacing = (0.3, 0.4, 2.0)
        img_path = _create_synthetic_nifti(tmp_path, "img_001", voxel_spacing=spacing)
        lbl_path = _create_synthetic_nifti(
            tmp_path, "lbl_001", voxel_spacing=spacing, is_label=True
        )
        pairs = [{"image": str(img_path), "label": str(lbl_path)}]

        df = extract_nifti_metadata_task(pairs)
        assert abs(df.iloc[0]["voxel_spacing_x"] - 0.3) < 1e-6
        assert abs(df.iloc[0]["voxel_spacing_y"] - 0.4) < 1e-6
        assert abs(df.iloc[0]["voxel_spacing_z"] - 2.0) < 1e-6

    def test_intensity_range_matches_data(self, tmp_path: Path) -> None:
        """T2-R5: intensity_min / intensity_max match actual data."""
        from minivess.orchestration.flows.data_flow import (
            extract_nifti_metadata_task,
        )

        img_path = _create_synthetic_nifti(tmp_path, "img_001")
        lbl_path = _create_synthetic_nifti(tmp_path, "lbl_001", is_label=True)
        pairs = [{"image": str(img_path), "label": str(lbl_path)}]

        df = extract_nifti_metadata_task(pairs)
        # Synthetic data is uniform [0, 1000)
        assert df.iloc[0]["intensity_min"] >= 0.0
        assert df.iloc[0]["intensity_max"] <= 1000.0
        assert df.iloc[0]["intensity_max"] > df.iloc[0]["intensity_min"]

    def test_has_valid_affine_true_for_standard(self, tmp_path: Path) -> None:
        """T2-R6: has_valid_affine is True for standard affines."""
        from minivess.orchestration.flows.data_flow import (
            extract_nifti_metadata_task,
        )

        img_path = _create_synthetic_nifti(tmp_path, "img_001")
        lbl_path = _create_synthetic_nifti(tmp_path, "lbl_001", is_label=True)
        pairs = [{"image": str(img_path), "label": str(lbl_path)}]

        df = extract_nifti_metadata_task(pairs)
        assert bool(df.iloc[0]["has_valid_affine"]) is True

    def test_num_foreground_voxels(self, tmp_path: Path) -> None:
        """T2-R7: num_foreground_voxels counts nonzero label voxels."""
        from minivess.orchestration.flows.data_flow import (
            extract_nifti_metadata_task,
        )

        img_path = _create_synthetic_nifti(tmp_path, "img_001")
        lbl_path = _create_synthetic_nifti(tmp_path, "lbl_001", is_label=True)
        pairs = [{"image": str(img_path), "label": str(lbl_path)}]

        df = extract_nifti_metadata_task(pairs)
        # Label has foreground in region [20:40, 20:40, 10:20] = 20*20*10 = 4000
        assert df.iloc[0]["num_foreground_voxels"] == 4000

    def test_empty_pairs_returns_empty_dataframe(self) -> None:
        """T2-R8: Empty pairs list returns empty DataFrame with correct columns."""
        from minivess.orchestration.flows.data_flow import (
            extract_nifti_metadata_task,
        )

        df = extract_nifti_metadata_task([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert "file_path" in df.columns
        assert "shape_x" in df.columns
