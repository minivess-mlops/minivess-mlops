"""Tests for auxiliary GT precomputation (T2 — topology real-data plan)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import nibabel as nib
import numpy as np

from minivess.data.multitask_targets import AuxTargetConfig, LoadAuxiliaryTargetsd

if TYPE_CHECKING:
    from pathlib import Path
from minivess.orchestration.precompute import precompute_auxiliary_targets


def _make_sphere_mask(shape: tuple[int, int, int] = (32, 32, 32)) -> np.ndarray:
    """Create a binary sphere mask for testing."""
    coords = np.mgrid[: shape[0], : shape[1], : shape[2]]
    center = np.array(shape) / 2
    dist = np.sqrt(sum((c - center[i]) ** 2 for i, c in enumerate(coords)))
    return (dist < min(shape) / 4).astype(np.float32)


def _compute_sdf(mask: np.ndarray) -> np.ndarray:
    """Signed distance function: negative inside, positive outside."""
    from scipy.ndimage import distance_transform_edt

    pos_dist = distance_transform_edt(mask == 0)
    neg_dist = distance_transform_edt(mask > 0)
    return (pos_dist - neg_dist).astype(np.float32)


def _compute_centreline_dist(mask: np.ndarray) -> np.ndarray:
    """Distance from each voxel to nearest skeleton voxel."""
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize

    binary = (mask > 0).astype(np.uint8)
    skeleton = skeletonize(binary)
    return distance_transform_edt(skeleton == 0).astype(np.float32)


class TestPrecomputeAuxiliaryTargets:
    """Tests for precompute_auxiliary_targets function."""

    def test_computes_sdf_for_synthetic_volume(self, tmp_path: Path) -> None:
        """SDF has correct shape and sign convention."""
        mask = _make_sphere_mask()
        affine = np.eye(4)
        label_path = tmp_path / "test_label.nii.gz"
        nib.save(nib.Nifti1Image(mask, affine), str(label_path))

        volumes = [{"label": str(label_path), "volume_id": "test01"}]
        configs = [AuxTargetConfig(name="sdf", suffix="sdf", compute_fn=_compute_sdf)]

        result = precompute_auxiliary_targets(volumes, tmp_path / "output", configs)

        assert result["computed"] == 1
        assert result["skipped"] == 0

        out_file = tmp_path / "output" / "test01_sdf.nii.gz"
        assert out_file.exists()

        sdf_img = nib.load(str(out_file))
        sdf_data = np.asarray(sdf_img.dataobj)
        assert sdf_data.shape == mask.shape
        # Inside the sphere should be negative
        center = tuple(s // 2 for s in mask.shape)
        assert sdf_data[center] < 0  # negative inside

    def test_computes_centreline_distance(self, tmp_path: Path) -> None:
        """Centreline distance has correct shape, zero on skeleton."""
        mask = _make_sphere_mask()
        affine = np.eye(4)
        label_path = tmp_path / "test_label.nii.gz"
        nib.save(nib.Nifti1Image(mask, affine), str(label_path))

        volumes = [{"label": str(label_path), "volume_id": "test02"}]
        configs = [
            AuxTargetConfig(
                name="centerline_dist",
                suffix="centerline_dist",
                compute_fn=_compute_centreline_dist,
            )
        ]

        result = precompute_auxiliary_targets(volumes, tmp_path / "output", configs)

        assert result["computed"] == 1
        out_file = tmp_path / "output" / "test02_centerline_dist.nii.gz"
        assert out_file.exists()

        cl_img = nib.load(str(out_file))
        cl_data = np.asarray(cl_img.dataobj)
        assert cl_data.shape == mask.shape
        assert cl_data.min() >= 0  # distance is non-negative

    def test_idempotency_skips_existing(self, tmp_path: Path) -> None:
        """Skip computation when output file already exists."""
        mask = _make_sphere_mask()
        affine = np.eye(4)
        label_path = tmp_path / "test_label.nii.gz"
        nib.save(nib.Nifti1Image(mask, affine), str(label_path))

        volumes = [{"label": str(label_path), "volume_id": "test03"}]
        configs = [AuxTargetConfig(name="sdf", suffix="sdf", compute_fn=_compute_sdf)]
        output_dir = tmp_path / "output"

        # First run — should compute
        result1 = precompute_auxiliary_targets(volumes, output_dir, configs)
        assert result1["computed"] == 1

        # Second run — should skip
        result2 = precompute_auxiliary_targets(volumes, output_dir, configs)
        assert result2["skipped"] == 1
        assert result2["computed"] == 0

    def test_force_recomputes(self, tmp_path: Path) -> None:
        """force=True recomputes even when file exists."""
        mask = _make_sphere_mask()
        affine = np.eye(4)
        label_path = tmp_path / "test_label.nii.gz"
        nib.save(nib.Nifti1Image(mask, affine), str(label_path))

        volumes = [{"label": str(label_path), "volume_id": "test04"}]
        configs = [AuxTargetConfig(name="sdf", suffix="sdf", compute_fn=_compute_sdf)]
        output_dir = tmp_path / "output"

        # First run
        precompute_auxiliary_targets(volumes, output_dir, configs)

        # Second run with force
        result = precompute_auxiliary_targets(volumes, output_dir, configs, force=True)
        assert result["computed"] == 1
        assert result["skipped"] == 0

    def test_multiple_volumes_and_targets(self, tmp_path: Path) -> None:
        """Handles multiple volumes x multiple targets."""
        affine = np.eye(4)
        volumes = []
        for i in range(3):
            mask = _make_sphere_mask()
            label_path = tmp_path / f"vol{i:02d}_label.nii.gz"
            nib.save(nib.Nifti1Image(mask, affine), str(label_path))
            volumes.append({"label": str(label_path), "volume_id": f"vol{i:02d}"})

        configs = [
            AuxTargetConfig(name="sdf", suffix="sdf", compute_fn=_compute_sdf),
            AuxTargetConfig(
                name="centerline_dist",
                suffix="centerline_dist",
                compute_fn=_compute_centreline_dist,
            ),
        ]

        result = precompute_auxiliary_targets(volumes, tmp_path / "output", configs)

        # 3 volumes x 2 targets = 6
        assert result["computed"] == 6


class TestLoadAuxiliaryTargetsd:
    """Tests for the MONAI-compatible transform."""

    def test_loads_precomputed_nifti(self, tmp_path: Path) -> None:
        """Loads precomputed target from NIfTI file."""
        # Create a precomputed SDF file
        sdf_data = np.random.randn(16, 16, 16).astype(np.float32)
        nib.save(
            nib.Nifti1Image(sdf_data, np.eye(4)),
            str(tmp_path / "vol01_sdf.nii.gz"),
        )

        config = AuxTargetConfig(name="sdf", suffix="sdf", compute_fn=_compute_sdf)
        transform = LoadAuxiliaryTargetsd(
            label_key="label",
            aux_configs=[config],
            precomputed_dir=tmp_path,
        )

        data: dict[str, Any] = {
            "label": np.zeros((16, 16, 16), dtype=np.float32),
            "volume_id": "vol01",
        }
        result = transform(data)
        assert "sdf" in result
        np.testing.assert_allclose(result["sdf"], sdf_data, atol=1e-5)

    def test_fallback_to_on_the_fly(self) -> None:
        """Computes on-the-fly when no precomputed file."""
        mask = _make_sphere_mask((16, 16, 16))
        config = AuxTargetConfig(name="sdf", suffix="sdf", compute_fn=_compute_sdf)
        transform = LoadAuxiliaryTargetsd(
            label_key="label",
            aux_configs=[config],
            precomputed_dir=None,
        )

        data: dict[str, Any] = {"label": mask, "volume_id": "vol99"}
        result = transform(data)
        assert "sdf" in result
        assert result["sdf"].shape == mask.shape
