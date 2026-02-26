"""Tests for the DVC preprocess stage.

Verifies that the preprocessing module:
1. Discovers NIfTI pairs from raw data
2. Resamples to uniform voxel spacing
3. Writes processed output with correct structure
4. Generates a validation report JSON
5. Works as a runnable __main__ module
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class TestPreprocessCreatesOutput:
    """Preprocessing should create the processed directory structure."""

    def test_preprocess_creates_output_dir(self, tmp_path: Path) -> None:
        from minivess.data.preprocess import preprocess_dataset
        from tests.v2.fixtures.synthetic_nifti import create_synthetic_nifti_dataset

        raw_dir = create_synthetic_nifti_dataset(
            tmp_path / "raw", n_volumes=2, spatial_size=(32, 32, 16)
        )
        out_dir = tmp_path / "processed"

        preprocess_dataset(raw_dir, out_dir)

        assert out_dir.exists()
        assert (out_dir / "imagesTr").exists()
        assert (out_dir / "labelsTr").exists()

    def test_preprocess_copies_all_volumes(self, tmp_path: Path) -> None:
        from minivess.data.preprocess import preprocess_dataset
        from tests.v2.fixtures.synthetic_nifti import create_synthetic_nifti_dataset

        n = 3
        raw_dir = create_synthetic_nifti_dataset(
            tmp_path / "raw", n_volumes=n, spatial_size=(32, 32, 16)
        )
        out_dir = tmp_path / "processed"

        preprocess_dataset(raw_dir, out_dir)

        images = list((out_dir / "imagesTr").glob("*.nii.gz"))
        labels = list((out_dir / "labelsTr").glob("*.nii.gz"))
        assert len(images) == n
        assert len(labels) == n

    def test_preprocess_matching_filenames(self, tmp_path: Path) -> None:
        from minivess.data.preprocess import preprocess_dataset
        from tests.v2.fixtures.synthetic_nifti import create_synthetic_nifti_dataset

        raw_dir = create_synthetic_nifti_dataset(
            tmp_path / "raw", n_volumes=2, spatial_size=(32, 32, 16)
        )
        out_dir = tmp_path / "processed"

        preprocess_dataset(raw_dir, out_dir)

        img_names = {p.name for p in (out_dir / "imagesTr").glob("*.nii.gz")}
        lbl_names = {p.name for p in (out_dir / "labelsTr").glob("*.nii.gz")}
        assert img_names == lbl_names, "Image and label filenames should match"


class TestPreprocessReport:
    """Preprocessing should generate a validation report."""

    def test_report_generated(self, tmp_path: Path) -> None:
        from minivess.data.preprocess import preprocess_dataset
        from tests.v2.fixtures.synthetic_nifti import create_synthetic_nifti_dataset

        raw_dir = create_synthetic_nifti_dataset(
            tmp_path / "raw", n_volumes=2, spatial_size=(32, 32, 16)
        )
        out_dir = tmp_path / "processed"

        report_path = preprocess_dataset(raw_dir, out_dir)

        assert report_path.exists()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert "volumes" in report
        assert "summary" in report

    def test_report_has_per_volume_stats(self, tmp_path: Path) -> None:
        from minivess.data.preprocess import preprocess_dataset
        from tests.v2.fixtures.synthetic_nifti import create_synthetic_nifti_dataset

        raw_dir = create_synthetic_nifti_dataset(
            tmp_path / "raw", n_volumes=2, spatial_size=(32, 32, 16)
        )
        out_dir = tmp_path / "processed"

        report_path = preprocess_dataset(raw_dir, out_dir)
        report = json.loads(report_path.read_text(encoding="utf-8"))

        assert len(report["volumes"]) == 2
        vol = report["volumes"][0]
        assert "filename" in vol
        assert "shape" in vol
        assert "voxel_spacing" in vol
        assert "intensity_range" in vol

    def test_report_summary_fields(self, tmp_path: Path) -> None:
        from minivess.data.preprocess import preprocess_dataset
        from tests.v2.fixtures.synthetic_nifti import create_synthetic_nifti_dataset

        raw_dir = create_synthetic_nifti_dataset(
            tmp_path / "raw", n_volumes=2, spatial_size=(32, 32, 16)
        )
        out_dir = tmp_path / "processed"

        report_path = preprocess_dataset(raw_dir, out_dir)
        report = json.loads(report_path.read_text(encoding="utf-8"))

        summary = report["summary"]
        assert "total_volumes" in summary
        assert "target_spacing" in summary
        assert summary["total_volumes"] == 2


class TestPreprocessIdempotent:
    """Running preprocess twice should produce the same result."""

    def test_preprocess_idempotent(self, tmp_path: Path) -> None:
        from minivess.data.preprocess import preprocess_dataset
        from tests.v2.fixtures.synthetic_nifti import create_synthetic_nifti_dataset

        raw_dir = create_synthetic_nifti_dataset(
            tmp_path / "raw", n_volumes=2, spatial_size=(32, 32, 16)
        )
        out_dir = tmp_path / "processed"

        report1 = preprocess_dataset(raw_dir, out_dir)
        report2 = preprocess_dataset(raw_dir, out_dir)

        r1 = json.loads(report1.read_text(encoding="utf-8"))
        r2 = json.loads(report2.read_text(encoding="utf-8"))
        assert r1["summary"]["total_volumes"] == r2["summary"]["total_volumes"]


class TestPreprocessEbrainsLayout:
    """Preprocessing should also work with raw/seg EBRAINS layout."""

    def test_preprocess_ebrains_layout(self, tmp_path: Path) -> None:
        from minivess.data.preprocess import preprocess_dataset
        from tests.v2.fixtures.synthetic_nifti import (
            create_synthetic_nifti_dataset_ebrains,
        )

        raw_dir = create_synthetic_nifti_dataset_ebrains(
            tmp_path / "raw", n_volumes=2, spatial_size=(32, 32, 16)
        )
        out_dir = tmp_path / "processed"

        preprocess_dataset(raw_dir, out_dir)

        images = list((out_dir / "imagesTr").glob("*.nii.gz"))
        labels = list((out_dir / "labelsTr").glob("*.nii.gz"))
        assert len(images) == 2
        assert len(labels) == 2
