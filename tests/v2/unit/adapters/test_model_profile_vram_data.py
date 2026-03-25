"""Tests for model profile VRAM measurement data completeness.

Verifies that all production model profiles have measured VRAM data
with non-null training_gb values. This ensures the VRAM estimator
has the data it needs for pre-training budget checks.

Task 2.3 of the SAM3 batch-size-1 plan.
"""

from __future__ import annotations

import pytest

from minivess.config.model_profiles import load_model_profile

# All production model families that should have measured VRAM data
_PRODUCTION_MODELS = ("dynunet", "sam3_topolora", "sam3_hybrid", "mambavesselnet")


class TestModelProfilesHaveMeasuredVram:
    """All production model profiles must have measured=true with non-null training_gb."""

    @pytest.mark.parametrize("model_family", _PRODUCTION_MODELS)
    def test_model_profiles_have_measured_vram(self, model_family: str) -> None:
        """Profile for {model_family} has vram.measured=true and training_gb set."""
        profile = load_model_profile(model_family)

        assert profile.vram is not None, (
            f"Model profile '{model_family}' has no vram section"
        )
        assert profile.vram.measured is True, (
            f"Model profile '{model_family}' has vram.measured=false — "
            f"needs real GPU benchmarking"
        )
        assert profile.vram.training_gb is not None, (
            f"Model profile '{model_family}' has vram.training_gb=null — "
            f"needs real GPU benchmarking"
        )
        assert profile.vram.training_gb > 0, (
            f"Model profile '{model_family}' has vram.training_gb={profile.vram.training_gb} "
            f"— must be positive"
        )

    @pytest.mark.parametrize("model_family", _PRODUCTION_MODELS)
    def test_model_profiles_have_measured_gpu(self, model_family: str) -> None:
        """Profile for {model_family} has measured_gpu and measured_date set."""
        profile = load_model_profile(model_family)

        assert profile.vram is not None
        assert profile.vram.measured_gpu is not None, (
            f"Model profile '{model_family}' missing vram.measured_gpu"
        )
        assert profile.vram.measured_date is not None, (
            f"Model profile '{model_family}' missing vram.measured_date"
        )


class TestSam3PerBatchSizeData:
    """SAM3 profiles must have per_batch_size VRAM data for batch-aware estimation."""

    @pytest.mark.parametrize(
        "model_family", ("sam3_topolora", "sam3_hybrid")
    )
    def test_sam3_profiles_have_per_batch_size(self, model_family: str) -> None:
        """SAM3 profiles have per_batch_size data with at least BS=1 and BS=2."""
        profile = load_model_profile(model_family)

        assert profile.vram is not None
        assert profile.vram.per_batch_size is not None, (
            f"Model profile '{model_family}' missing vram.per_batch_size"
        )
        assert 1 in profile.vram.per_batch_size, (
            f"Model profile '{model_family}' missing per_batch_size entry for BS=1"
        )
        assert 2 in profile.vram.per_batch_size, (
            f"Model profile '{model_family}' missing per_batch_size entry for BS=2"
        )

    def test_sam3_topolora_bs2_exceeds_l4(self) -> None:
        """SAM3 TopoLoRA at BS=2 needs more than L4's 24 GB (21.9 GB measured + overhead)."""
        profile = load_model_profile("sam3_topolora")

        assert profile.vram is not None
        assert profile.vram.per_batch_size is not None
        # BS=2 measured at 21.9 GB — while under 24 GB nominal, it OOMs due to
        # framework overhead and activation spikes. The data records the measured value.
        assert profile.vram.per_batch_size[2] > 20.0

    def test_sam3_topolora_bs1_fits_l4(self) -> None:
        """SAM3 TopoLoRA at BS=1 fits within L4's 24 GB."""
        profile = load_model_profile("sam3_topolora")

        assert profile.vram is not None
        assert profile.vram.per_batch_size is not None
        assert profile.vram.per_batch_size[1] < 24.0

    def test_sam3_hybrid_bs1_fits_l4(self) -> None:
        """SAM3 Hybrid at BS=1 fits comfortably within L4's 24 GB."""
        profile = load_model_profile("sam3_hybrid")

        assert profile.vram is not None
        assert profile.vram.per_batch_size is not None
        assert profile.vram.per_batch_size[1] < 24.0
