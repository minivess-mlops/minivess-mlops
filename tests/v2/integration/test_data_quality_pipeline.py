"""Integration test for the full data quality pipeline (T7).

Exercises: discover -> extract metadata -> Pandera -> GE -> DATA-CARE ->
DeepChecks -> enforce -> split. Uses synthetic NIfTI files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path


def _create_synthetic_dataset(
    tmp_path: Path,
    n_volumes: int = 4,
    shape: tuple[int, int, int] = (32, 32, 16),
    voxel_spacing: tuple[float, float, float] = (0.5, 0.5, 1.0),
) -> Path:
    """Create a synthetic NIfTI dataset in images/labels layout."""
    import nibabel as nib

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    for i in range(n_volumes):
        # Image
        rng = np.random.default_rng(42 + i)
        img_data = rng.uniform(0.0, 1000.0, size=shape).astype(np.float32)
        affine = np.diag([*voxel_spacing, 1.0])
        nib.save(
            nib.Nifti1Image(img_data, affine), str(images_dir / f"vol_{i:03d}.nii.gz")
        )

        # Label
        lbl_data = np.zeros(shape, dtype=np.int16)
        lbl_data[10:20, 10:20, 5:10] = 1
        nib.save(
            nib.Nifti1Image(lbl_data, affine), str(labels_dir / f"vol_{i:03d}.nii.gz")
        )

    return tmp_path


@pytest.mark.integration
class TestFullQualityPipeline:
    """End-to-end data quality pipeline integration tests."""

    def test_full_pipeline_valid_data_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """T7-R1: Full pipeline with valid synthetic NIfTI passes all gates."""
        monkeypatch.setenv("MINIVESS_ALLOW_HOST", "1")
        monkeypatch.setenv("SPLITS_OUTPUT_DIR", str(tmp_path / "splits"))

        from minivess.orchestration.flows.data_flow import (
            DataFlowResult,
            run_data_flow,
        )

        data_dir = _create_synthetic_dataset(tmp_path / "data")
        result = run_data_flow(data_dir=data_dir, n_folds=2, seed=42)

        assert isinstance(result, DataFlowResult)
        assert result.quality_passed is True
        assert len(result.pairs) == 4

    def test_pipeline_with_bad_spacing_triggers_pandera(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """T7-R2: Pipeline with bad spacing triggers Pandera error gate."""
        monkeypatch.setenv("MINIVESS_ALLOW_HOST", "1")
        monkeypatch.setenv("SPLITS_OUTPUT_DIR", str(tmp_path / "splits"))

        from minivess.validation.enforcement import DataQualityError

        # Create data with out-of-range voxel spacing
        data_dir = _create_synthetic_dataset(
            tmp_path / "data",
            voxel_spacing=(99.0, 0.5, 1.0),  # x spacing out of range
        )

        from minivess.orchestration.flows.data_flow import run_data_flow

        # Pandera is configured as 'error' severity, so it should raise
        with pytest.raises(DataQualityError):
            run_data_flow(data_dir=data_dir, n_folds=2, seed=42)

    def test_skip_quality_gate_bypasses_all(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """T7-R4: MINIVESS_SKIP_QUALITY_GATE=1 bypasses all gates."""
        monkeypatch.setenv("MINIVESS_ALLOW_HOST", "1")
        monkeypatch.setenv("MINIVESS_SKIP_QUALITY_GATE", "1")
        monkeypatch.setenv("SPLITS_OUTPUT_DIR", str(tmp_path / "splits"))

        from minivess.orchestration.flows.data_flow import run_data_flow

        # Create data with bad spacing — should NOT fail due to skip
        data_dir = _create_synthetic_dataset(
            tmp_path / "data",
            voxel_spacing=(99.0, 0.5, 1.0),
        )

        result = run_data_flow(data_dir=data_dir, n_folds=2, seed=42)
        assert result.quality_passed is True

    def test_quality_gate_results_in_provenance(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """T7-R5: DataFlowResult includes quality gate results in provenance."""
        monkeypatch.setenv("MINIVESS_ALLOW_HOST", "1")
        monkeypatch.setenv("SPLITS_OUTPUT_DIR", str(tmp_path / "splits"))

        from minivess.orchestration.flows.data_flow import run_data_flow

        data_dir = _create_synthetic_dataset(tmp_path / "data")
        result = run_data_flow(data_dir=data_dir, n_folds=2, seed=42)

        assert result.provenance is not None
        assert "data_n_volumes" in result.provenance
