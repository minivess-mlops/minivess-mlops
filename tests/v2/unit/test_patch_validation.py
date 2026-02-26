from __future__ import annotations

import pytest

from minivess.data.profiler import DatasetProfile
from minivess.data.validation import (
    MemoryBudgetError,
    PatchValidationError,
    validate_cache_fits_ram,
    validate_no_default_resampling,
    validate_patch_divisibility,
    validate_patch_fits_dataset,
    validate_vram_budget,
)


def _make_profile(
    min_shape: tuple = (512, 512, 5),
    total_size_bytes: int = 4_500_000_000,
) -> DatasetProfile:
    """Helper to create test DatasetProfile."""
    return DatasetProfile(
        num_volumes=70,
        min_shape=min_shape,
        max_shape=(512, 512, 30),
        median_shape=(512, 512, 15),
        min_spacing=(0.31, 0.31, 0.31),
        max_spacing=(4.97, 4.97, 4.97),
        median_spacing=(0.5, 0.5, 0.5),
        total_size_bytes=total_size_bytes,
        volume_stats=[],
        outlier_volumes=[],
    )


class TestPatchDatasetCompatibility:
    def test_patch_fits_all_volumes(self):
        """Valid patch size passes validation."""
        profile = _make_profile(min_shape=(512, 512, 5))
        # (64, 64, 4) fits within (512, 512, 5)
        validate_patch_fits_dataset(patch_size=(64, 64, 4), profile=profile)
        # Should not raise

    def test_rejects_oversized_patch_z(self):
        """patch_z=16 with min_z=5 -> PatchValidationError."""
        profile = _make_profile(min_shape=(512, 512, 5))
        with pytest.raises(PatchValidationError, match="z"):
            validate_patch_fits_dataset(patch_size=(64, 64, 16), profile=profile)

    def test_rejects_oversized_patch_xy(self):
        """patch_x=1024 with min_x=512 -> PatchValidationError."""
        profile = _make_profile(min_shape=(512, 512, 5))
        with pytest.raises(PatchValidationError):
            validate_patch_fits_dataset(patch_size=(1024, 64, 4), profile=profile)

    def test_patch_divisible_by_model(self):
        """DynUNet patches divisible by 8 pass."""
        validate_patch_divisibility(patch_size=(96, 96, 8), model_divisor=8)
        # Should not raise

    def test_rejects_non_divisible_patch(self):
        """96x96x23 -> error (23 % 8 != 0)."""
        with pytest.raises(PatchValidationError, match="divisible"):
            validate_patch_divisibility(patch_size=(96, 96, 23), model_divisor=8)


class TestMemoryBudget:
    def test_cache_fits_ram(self):
        """cached_size < 70% available RAM passes."""
        profile = _make_profile(total_size_bytes=4_500_000_000)  # 4.5 GB
        validate_cache_fits_ram(
            profile=profile,
            cache_rate=1.0,
            available_ram_mb=24000,  # 24 GB
        )
        # Should not raise

    def test_rejects_oversized_cache(self):
        """100 GB dataset, rate=1.0, 32 GB RAM -> MemoryBudgetError."""
        profile = _make_profile(total_size_bytes=100_000_000_000)  # 100 GB
        with pytest.raises(MemoryBudgetError, match="cache"):
            validate_cache_fits_ram(
                profile=profile,
                cache_rate=1.0,
                available_ram_mb=24000,
            )

    def test_vram_estimate_within_budget(self):
        """Small batch + small patch on 8 GB GPU passes."""
        validate_vram_budget(
            batch_size=2,
            patch_size=(96, 96, 24),
            gpu_vram_mb=8192,
            model_name="dynunet",
        )
        # Should not raise

    def test_rejects_large_batch_on_small_gpu(self):
        """batch=8 with large patches on 8 GB GPU -> MemoryBudgetError."""
        with pytest.raises(MemoryBudgetError, match="VRAM"):
            validate_vram_budget(
                batch_size=8,
                patch_size=(128, 128, 64),
                gpu_vram_mb=8192,
                model_name="dynunet",
            )


class TestDefaults:
    def test_no_default_resampling_passes(self):
        """voxel_spacing == (0, 0, 0) passes validation."""
        validate_no_default_resampling(voxel_spacing=(0.0, 0.0, 0.0))
        # Should not raise

    def test_resampling_detected_warns(self):
        """Non-zero voxel_spacing raises warning (not error by default)."""
        # With strict=True, it should raise
        with pytest.raises(PatchValidationError, match="resampling"):
            validate_no_default_resampling(
                voxel_spacing=(1.0, 1.0, 1.0),
                strict=True,
            )

    def test_resampling_non_strict_returns_warning(self):
        """Non-zero spacing with strict=False returns warning string."""
        result = validate_no_default_resampling(
            voxel_spacing=(1.0, 1.0, 1.0),
            strict=False,
        )
        assert result is not None
        assert "resampling" in result.lower()
