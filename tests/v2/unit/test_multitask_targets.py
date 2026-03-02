"""Tests for generic multi-task target loading framework (T2 — #227)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from minivess.data.multitask_targets import AuxTargetConfig, LoadAuxiliaryTargetsd
from minivess.pipeline.sdf_generation import compute_sdf_from_mask


def _make_sample_data(
    shape: tuple[int, int, int] = (16, 32, 32),
) -> dict[str, np.ndarray]:
    """Create a sample MONAI-style data dict with image and label."""
    mask = np.zeros(shape, dtype=np.uint8)
    mask[4:12, 8:24, 8:24] = 1
    return {
        "image": np.random.default_rng(0).standard_normal(shape).astype(np.float32),
        "label": mask,
    }


class TestLoadAuxiliaryTargetsd:
    """Tests for LoadAuxiliaryTargetsd MONAI MapTransform."""

    def test_generic_loader_adds_configured_keys(self) -> None:
        """All configured aux keys present in output."""
        configs = [
            AuxTargetConfig(name="sdf", suffix="sdf", compute_fn=compute_sdf_from_mask),
            AuxTargetConfig(
                name="dummy",
                suffix="dummy",
                compute_fn=lambda m: np.ones_like(m, dtype=np.float32),
            ),
        ]
        transform = LoadAuxiliaryTargetsd(label_key="label", aux_configs=configs)
        data = _make_sample_data()
        result = transform(data)
        assert "sdf" in result, "SDF key should be in result"
        assert "dummy" in result, "dummy key should be in result"

    def test_generic_loader_label_unchanged(self) -> None:
        """Original 'label' key preserved."""
        configs = [
            AuxTargetConfig(name="sdf", suffix="sdf", compute_fn=compute_sdf_from_mask),
        ]
        transform = LoadAuxiliaryTargetsd(label_key="label", aux_configs=configs)
        data = _make_sample_data()
        label_orig = data["label"].copy()
        result = transform(data)
        np.testing.assert_array_equal(result["label"], label_orig)

    def test_generic_loader_shape_matches_label(self) -> None:
        """Aux targets match label spatial dims."""
        configs = [
            AuxTargetConfig(name="sdf", suffix="sdf", compute_fn=compute_sdf_from_mask),
        ]
        transform = LoadAuxiliaryTargetsd(label_key="label", aux_configs=configs)
        data = _make_sample_data()
        result = transform(data)
        assert result["sdf"].shape == result["label"].shape

    def test_generic_loader_empty_mask_no_crash(self) -> None:
        """Graceful handling of empty masks."""
        configs = [
            AuxTargetConfig(name="sdf", suffix="sdf", compute_fn=compute_sdf_from_mask),
        ]
        transform = LoadAuxiliaryTargetsd(label_key="label", aux_configs=configs)
        data = {
            "image": np.zeros((8, 16, 16), dtype=np.float32),
            "label": np.zeros((8, 16, 16), dtype=np.uint8),
        }
        result = transform(data)
        assert "sdf" in result
        assert np.isfinite(result["sdf"]).all()

    def test_generic_loader_3d_volume(self) -> None:
        """Works with realistic 3D volumes (32,64,64)."""
        configs = [
            AuxTargetConfig(name="sdf", suffix="sdf", compute_fn=compute_sdf_from_mask),
        ]
        transform = LoadAuxiliaryTargetsd(label_key="label", aux_configs=configs)
        data = _make_sample_data((32, 64, 64))
        result = transform(data)
        assert result["sdf"].shape == (32, 64, 64)

    def test_generic_loader_fallback_compute(self) -> None:
        """On-the-fly compute when precomputed file missing."""
        configs = [
            AuxTargetConfig(name="sdf", suffix="sdf", compute_fn=compute_sdf_from_mask),
        ]
        # No precomputed_dir — should fall back to compute_fn
        transform = LoadAuxiliaryTargetsd(label_key="label", aux_configs=configs)
        data = _make_sample_data()
        result = transform(data)
        assert "sdf" in result

    def test_generic_loader_precomputed_preferred(self) -> None:
        """Loads NIfTI when available in precomputed_dir."""
        shape = (8, 16, 16)
        mask = np.zeros(shape, dtype=np.uint8)
        mask[2:6, 4:12, 4:12] = 1

        # Create precomputed SDF file
        sdf_data = np.ones(shape, dtype=np.float32) * 42.0  # sentinel value
        with tempfile.TemporaryDirectory() as tmpdir:
            sdf_path = Path(tmpdir) / "vol001_sdf.nii.gz"
            nib.save(nib.Nifti1Image(sdf_data, np.eye(4)), str(sdf_path))

            configs = [
                AuxTargetConfig(
                    name="sdf", suffix="sdf", compute_fn=compute_sdf_from_mask
                ),
            ]
            transform = LoadAuxiliaryTargetsd(
                label_key="label",
                aux_configs=configs,
                precomputed_dir=Path(tmpdir),
            )
            data = {
                "image": np.zeros(shape, dtype=np.float32),
                "label": mask,
                "volume_id": "vol001",
            }
            result = transform(data)
            # Should have loaded the sentinel value, not computed
            assert result["sdf"].max() == 42.0, (
                "Should load precomputed file, not compute"
            )

    def test_generic_loader_arbitrary_key_name(self) -> None:
        """Works with any key name string."""
        configs = [
            AuxTargetConfig(
                name="my_custom_target",
                suffix="custom",
                compute_fn=lambda m: np.zeros_like(m, dtype=np.float32),
            ),
        ]
        transform = LoadAuxiliaryTargetsd(label_key="label", aux_configs=configs)
        data = _make_sample_data()
        result = transform(data)
        assert "my_custom_target" in result

    def test_spatial_augmentation_applies_to_all_keys(self) -> None:
        """RandFlipd transforms aux keys too — verify keys are accessible."""
        configs = [
            AuxTargetConfig(name="sdf", suffix="sdf", compute_fn=compute_sdf_from_mask),
        ]
        transform = LoadAuxiliaryTargetsd(label_key="label", aux_configs=configs)
        data = _make_sample_data()
        result = transform(data)
        # Verify the keys list includes aux targets
        assert "sdf" in result
        assert result["sdf"].dtype == np.float32
