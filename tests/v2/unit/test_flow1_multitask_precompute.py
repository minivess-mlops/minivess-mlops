"""Tests for Flow 1 generic auxiliary GT precomputation (T18 — #245)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from minivess.data.multitask_targets import AuxTargetConfig
from minivess.pipeline.sdf_generation import compute_sdf_from_mask


def _make_volume_pair(
    tmpdir: Path,
    vol_id: str = "vol001",
    shape: tuple[int, int, int] = (8, 16, 16),
) -> dict[str, str]:
    """Create a synthetic NIfTI volume + label pair."""
    img_dir = tmpdir / "images"
    lbl_dir = tmpdir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    img = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    lbl = np.zeros(shape, dtype=np.uint8)
    lbl[2:6, 4:12, 4:12] = 1

    img_path = img_dir / f"{vol_id}.nii.gz"
    lbl_path = lbl_dir / f"{vol_id}.nii.gz"
    nib.save(nib.Nifti1Image(img, np.eye(4)), str(img_path))
    nib.save(nib.Nifti1Image(lbl, np.eye(4)), str(lbl_path))

    return {"image": str(img_path), "label": str(lbl_path), "volume_id": vol_id}


def _dummy_compute(mask: np.ndarray) -> np.ndarray:
    """Dummy compute function for testing."""
    return np.ones_like(mask, dtype=np.float32) * 99.0


class TestFlow1MultitaskPrecompute:
    """Tests for generic auxiliary GT precomputation."""

    def test_precompute_creates_configured_files(self) -> None:
        """NIfTI files created per config."""
        from minivess.orchestration.precompute import precompute_auxiliary_targets

        configs = [
            AuxTargetConfig(name="sdf", suffix="sdf", compute_fn=compute_sdf_from_mask),
        ]
        with tempfile.TemporaryDirectory() as tmp_str:
            tmpdir = Path(tmp_str)
            pair = _make_volume_pair(tmpdir)
            output_dir = tmpdir / "precomputed"
            precompute_auxiliary_targets([pair], output_dir, configs)
            expected = output_dir / "vol001_sdf.nii.gz"
            assert expected.exists()

    def test_precompute_generic_two_targets(self) -> None:
        """Works with 2 different target configs."""
        from minivess.orchestration.precompute import precompute_auxiliary_targets

        configs = [
            AuxTargetConfig(name="sdf", suffix="sdf", compute_fn=compute_sdf_from_mask),
            AuxTargetConfig(name="dummy", suffix="dummy", compute_fn=_dummy_compute),
        ]
        with tempfile.TemporaryDirectory() as tmp_str:
            tmpdir = Path(tmp_str)
            pair = _make_volume_pair(tmpdir)
            output_dir = tmpdir / "precomputed"
            precompute_auxiliary_targets([pair], output_dir, configs)
            assert (output_dir / "vol001_sdf.nii.gz").exists()
            assert (output_dir / "vol001_dummy.nii.gz").exists()

    def test_precompute_idempotent(self) -> None:
        """Second run skips existing files."""
        from minivess.orchestration.precompute import precompute_auxiliary_targets

        configs = [
            AuxTargetConfig(name="sdf", suffix="sdf", compute_fn=compute_sdf_from_mask),
        ]
        with tempfile.TemporaryDirectory() as tmp_str:
            tmpdir = Path(tmp_str)
            pair = _make_volume_pair(tmpdir)
            output_dir = tmpdir / "precomputed"

            # Run twice
            result1 = precompute_auxiliary_targets([pair], output_dir, configs)
            result2 = precompute_auxiliary_targets([pair], output_dir, configs)

            assert result1["computed"] == 1
            assert result2["skipped"] == 1

    def test_precompute_force_recomputes(self) -> None:
        """--force flag regenerates files."""
        from minivess.orchestration.precompute import precompute_auxiliary_targets

        configs = [
            AuxTargetConfig(name="sdf", suffix="sdf", compute_fn=compute_sdf_from_mask),
        ]
        with tempfile.TemporaryDirectory() as tmp_str:
            tmpdir = Path(tmp_str)
            pair = _make_volume_pair(tmpdir)
            output_dir = tmpdir / "precomputed"

            precompute_auxiliary_targets([pair], output_dir, configs)
            result = precompute_auxiliary_targets(
                [pair], output_dir, configs, force=True
            )
            assert result["computed"] == 1

    def test_precompute_shape_matches_label(self) -> None:
        """Aux target spatial dims match label."""
        from minivess.orchestration.precompute import precompute_auxiliary_targets

        shape = (8, 16, 16)
        configs = [
            AuxTargetConfig(name="sdf", suffix="sdf", compute_fn=compute_sdf_from_mask),
        ]
        with tempfile.TemporaryDirectory() as tmp_str:
            tmpdir = Path(tmp_str)
            pair = _make_volume_pair(tmpdir, shape=shape)
            output_dir = tmpdir / "precomputed"
            precompute_auxiliary_targets([pair], output_dir, configs)

            sdf_path = output_dir / "vol001_sdf.nii.gz"
            sdf = np.asarray(nib.load(str(sdf_path)).dataobj)
            assert sdf.shape == shape

    def test_precompute_custom_compute_fn(self) -> None:
        """Works with arbitrary compute function."""
        from minivess.orchestration.precompute import precompute_auxiliary_targets

        configs = [
            AuxTargetConfig(name="custom", suffix="custom", compute_fn=_dummy_compute),
        ]
        with tempfile.TemporaryDirectory() as tmp_str:
            tmpdir = Path(tmp_str)
            pair = _make_volume_pair(tmpdir)
            output_dir = tmpdir / "precomputed"
            precompute_auxiliary_targets([pair], output_dir, configs)

            custom_path = output_dir / "vol001_custom.nii.gz"
            data = np.asarray(nib.load(str(custom_path)).dataobj)
            assert data.max() == 99.0
