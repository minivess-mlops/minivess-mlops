from __future__ import annotations

from unittest.mock import patch

from minivess.config.adaptive_profiles import (
    HardwareBudget,
    compute_adaptive_profile,
    detect_hardware,
)
from minivess.data.profiler import DatasetProfile


def _make_dataset_profile(
    num_volumes: int = 70,
    min_shape: tuple = (512, 512, 5),
    max_shape: tuple = (512, 512, 30),
    total_size_bytes: int = 4_500_000_000,  # ~4.5 GB
) -> DatasetProfile:
    """Helper to create test DatasetProfile."""
    return DatasetProfile(
        num_volumes=num_volumes,
        min_shape=min_shape,
        max_shape=max_shape,
        median_shape=(
            (min_shape[0] + max_shape[0]) // 2,
            (min_shape[1] + max_shape[1]) // 2,
            (min_shape[2] + max_shape[2]) // 2,
        ),
        min_spacing=(0.31, 0.31, 0.31),
        max_spacing=(4.97, 4.97, 4.97),
        median_spacing=(0.5, 0.5, 0.5),
        total_size_bytes=total_size_bytes,
        volume_stats=[],
        outlier_volumes=[],
    )


class TestHardwareDetection:
    def test_detect_returns_budget(self):
        """detect_hardware returns HardwareBudget with all fields populated."""
        budget = detect_hardware()
        assert isinstance(budget, HardwareBudget)
        assert budget.ram_total_mb > 0
        assert budget.ram_available_mb > 0
        assert budget.cpu_count > 0

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_no_gpu_detected(self, mock_run):
        """Mock nvidia-smi failure → gpu_vram_mb=0."""
        budget = detect_hardware()
        assert budget.gpu_vram_mb == 0
        assert budget.gpu_name == ""

    def test_8gb_gpu_returns_gpu_low_tier(self):
        """8 GB GPU (8192 MB) maps to 'gpu_low' tier."""
        budget = HardwareBudget(
            ram_total_mb=32768,
            ram_available_mb=24000,
            gpu_vram_mb=8192,
            gpu_name="RTX 2070 Super",
            cpu_count=8,
            swap_used_mb=0,
        )
        assert budget.gpu_tier == "gpu_low"


class TestAdaptiveComputation:
    def test_patch_constrained_by_dataset(self):
        """Patch Z dimension constrained by smallest volume Z."""
        profile = _make_dataset_profile(min_shape=(512, 512, 5))
        budget = HardwareBudget(
            ram_total_mb=32768,
            ram_available_mb=24000,
            gpu_vram_mb=8192,
            gpu_name="RTX 2070",
            cpu_count=8,
            swap_used_mb=0,
        )
        result = compute_adaptive_profile(budget, profile, model_name="dynunet")
        assert result.patch_size[2] <= 5

    def test_patch_divisible_by_model_divisor(self):
        """DynUNet patches must be divisible by 8."""
        profile = _make_dataset_profile(min_shape=(512, 512, 32))
        budget = HardwareBudget(
            ram_total_mb=32768,
            ram_available_mb=24000,
            gpu_vram_mb=8192,
            gpu_name="RTX 2070",
            cpu_count=8,
            swap_used_mb=0,
        )
        result = compute_adaptive_profile(budget, profile, model_name="dynunet")
        assert result.patch_size[0] % 8 == 0
        assert result.patch_size[1] % 8 == 0
        assert result.patch_size[2] % 8 == 0

    def test_cache_rate_adaptive_to_ram(self):
        """32 GB RAM → cache_rate=1.0 for ~4.5 GB MiniVess dataset."""
        profile = _make_dataset_profile(total_size_bytes=4_500_000_000)
        budget = HardwareBudget(
            ram_total_mb=32768,
            ram_available_mb=24000,
            gpu_vram_mb=8192,
            gpu_name="RTX 2070",
            cpu_count=8,
            swap_used_mb=0,
        )
        result = compute_adaptive_profile(budget, profile, model_name="dynunet")
        assert (
            result.cache_rate >= 0.9
        )  # Should be 1.0 for 4.5 GB dataset with 24 GB RAM

    def test_cache_rate_reduced_for_large_dataset(self):
        """100 GB dataset → cache_rate < 0.3."""
        profile = _make_dataset_profile(total_size_bytes=100_000_000_000)
        budget = HardwareBudget(
            ram_total_mb=32768,
            ram_available_mb=24000,
            gpu_vram_mb=8192,
            gpu_name="RTX 2070",
            cpu_count=8,
            swap_used_mb=0,
        )
        result = compute_adaptive_profile(budget, profile, model_name="dynunet")
        assert result.cache_rate < 0.3

    def test_batch_size_reduced_when_vram_tight(self):
        """8 GB GPU → batch_size <= 2."""
        profile = _make_dataset_profile()
        budget = HardwareBudget(
            ram_total_mb=32768,
            ram_available_mb=24000,
            gpu_vram_mb=8192,
            gpu_name="RTX 2070",
            cpu_count=8,
            swap_used_mb=0,
        )
        result = compute_adaptive_profile(budget, profile, model_name="dynunet")
        assert result.batch_size <= 2

    def test_auto_profile_name(self):
        """Returns descriptive auto profile name."""
        profile = _make_dataset_profile()
        budget = HardwareBudget(
            ram_total_mb=32768,
            ram_available_mb=24000,
            gpu_vram_mb=8192,
            gpu_name="RTX 2070",
            cpu_count=8,
            swap_used_mb=0,
        )
        result = compute_adaptive_profile(budget, profile, model_name="dynunet")
        assert "auto" in result.name
        assert "dynunet" in result.name


class TestManualOverride:
    def test_explicit_profile_bypasses_auto(self):
        """When using get_compute_profile("gpu_low"), it returns the static profile."""
        from minivess.config.compute_profiles import get_compute_profile

        profile = get_compute_profile("gpu_low")
        assert profile.name == "gpu_low"
        assert profile.batch_size == 2

    def test_explicit_patch_size_overrides(self):
        """User can override patch_size in AdaptiveComputeProfile."""
        profile = _make_dataset_profile()
        budget = HardwareBudget(
            ram_total_mb=32768,
            ram_available_mb=24000,
            gpu_vram_mb=8192,
            gpu_name="RTX 2070",
            cpu_count=8,
            swap_used_mb=0,
        )
        result = compute_adaptive_profile(
            budget,
            profile,
            model_name="dynunet",
            override_patch_size=(64, 64, 8),
        )
        assert result.patch_size == (64, 64, 8)


class TestCPUProfile:
    def test_cpu_only_profile(self):
        """No GPU → CPU profile with no mixed precision."""
        profile = _make_dataset_profile()
        budget = HardwareBudget(
            ram_total_mb=32768,
            ram_available_mb=24000,
            gpu_vram_mb=0,
            gpu_name="",
            cpu_count=8,
            swap_used_mb=0,
        )
        result = compute_adaptive_profile(budget, profile, model_name="dynunet")
        assert result.mixed_precision is False
        assert result.batch_size >= 1
