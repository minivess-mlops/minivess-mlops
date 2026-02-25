from __future__ import annotations

import pytest
import yaml

from minivess.config.model_profiles import (
    ModelProfile,
    estimate_vram_mb,
    list_available_profiles,
    load_model_profile,
)


class TestModelProfile:
    def test_load_dynunet_profile(self):
        """DynUNet profile loads with correct divisor."""
        profile = load_model_profile("dynunet")
        assert isinstance(profile, ModelProfile)
        assert profile.name == "dynunet"
        assert profile.divisor == 8
        assert profile.model_overhead_mb > 0

    def test_load_segresnet_profile(self):
        """SegResNet profile loads."""
        profile = load_model_profile("segresnet")
        assert profile.name == "segresnet"
        assert profile.divisor == 8

    def test_load_vista3d_profile(self):
        """VISTA-3D profile loads with higher divisor."""
        profile = load_model_profile("vista3d")
        assert profile.name == "vista3d"
        assert profile.divisor >= 16

    def test_load_custom_profile(self, tmp_path):
        """Custom YAML profile can be loaded from arbitrary path."""
        custom = {
            "name": "my_model",
            "divisor": 4,
            "model_overhead_mb": 200,
            "bytes_per_voxel_amp": 8,
            "bytes_per_voxel_fp32": 16,
            "max_batch_size": {"gpu_low": 2, "gpu_high": 4},
            "default_patch_xy": 96,
            "notes": "Custom test model",
        }
        yaml_path = tmp_path / "my_model.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(custom, f)

        profile = load_model_profile("my_model", search_dirs=[tmp_path])
        assert profile.name == "my_model"
        assert profile.divisor == 4

    def test_unknown_profile_raises(self):
        """Unknown profile name raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_model_profile("nonexistent_model_xyz")

    def test_list_available_profiles(self):
        """Lists at least dynunet, segresnet, vista3d."""
        profiles = list_available_profiles()
        assert "dynunet" in profiles
        assert "segresnet" in profiles
        assert "vista3d" in profiles


class TestVRAMEstimation:
    def test_estimate_dynunet_small_patch(self):
        """Small patch + batch=2 -> reasonable VRAM estimate."""
        profile = load_model_profile("dynunet")
        vram = estimate_vram_mb(
            profile=profile,
            batch_size=2,
            patch_size=(96, 96, 24),
            mixed_precision=True,
        )
        assert 500 < vram < 8000  # Reasonable range

    def test_estimate_scales_with_batch(self):
        """Doubling batch size roughly doubles activation memory."""
        profile = load_model_profile("dynunet")
        vram_b1 = estimate_vram_mb(
            profile, batch_size=1, patch_size=(96, 96, 24), mixed_precision=True
        )
        vram_b2 = estimate_vram_mb(
            profile, batch_size=2, patch_size=(96, 96, 24), mixed_precision=True
        )
        # Batch 2 should be more than batch 1 (but not exactly 2x due to overhead)
        assert vram_b2 > vram_b1

    def test_fp32_uses_more_vram(self):
        """FP32 uses more VRAM than AMP."""
        profile = load_model_profile("dynunet")
        vram_amp = estimate_vram_mb(
            profile, batch_size=2, patch_size=(96, 96, 24), mixed_precision=True
        )
        vram_fp32 = estimate_vram_mb(
            profile, batch_size=2, patch_size=(96, 96, 24), mixed_precision=False
        )
        assert vram_fp32 > vram_amp


class TestProfileYAMLStructure:
    def test_dynunet_yaml_has_required_fields(self):
        """DynUNet YAML has all required fields."""
        profile = load_model_profile("dynunet")
        assert profile.divisor > 0
        assert profile.model_overhead_mb > 0
        assert profile.bytes_per_voxel_amp > 0
        assert profile.bytes_per_voxel_fp32 > 0

    def test_example_custom_yaml_exists(self):
        """Example custom profile exists as a template."""
        profiles = list_available_profiles()
        assert "example_custom" in profiles
