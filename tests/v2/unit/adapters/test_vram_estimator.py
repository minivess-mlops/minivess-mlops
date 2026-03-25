"""Tests for pre-training VRAM estimation and budget checking.

Task 2.2 of the SAM3 batch-size-1 plan.

Uses real model profile YAMLs from configs/model_profiles/ — no mocking
of YAML data since these tests validate the integration between the
estimator and the actual measured VRAM data.
"""

from __future__ import annotations

import pytest

from minivess.adapters.vram_estimator import check_vram_budget, estimate_training_vram


class TestEstimateTrainingVram:
    """estimate_training_vram reads model profiles and returns VRAM in GB."""

    def test_sam3_topolora_bs1(self) -> None:
        """SAM3 TopoLoRA at BS=1 returns the measured 13.0 GB."""
        vram = estimate_training_vram("sam3_topolora", batch_size=1)
        assert vram == pytest.approx(13.0, abs=0.1)

    def test_sam3_topolora_bs2(self) -> None:
        """SAM3 TopoLoRA at BS=2 returns the measured 21.9 GB."""
        vram = estimate_training_vram("sam3_topolora", batch_size=2)
        assert vram == pytest.approx(21.9, abs=0.1)

    def test_sam3_hybrid_bs1(self) -> None:
        """SAM3 Hybrid at BS=1 returns the measured 7.2 GB."""
        vram = estimate_training_vram("sam3_hybrid", batch_size=1)
        assert vram == pytest.approx(7.2, abs=0.1)

    def test_sam3_hybrid_bs2(self) -> None:
        """SAM3 Hybrid at BS=2 returns the estimated 14.4 GB."""
        vram = estimate_training_vram("sam3_hybrid", batch_size=2)
        assert vram == pytest.approx(14.4, abs=0.1)

    def test_dynunet_bs2(self) -> None:
        """DynUNet at BS=2 returns a positive VRAM estimate."""
        vram = estimate_training_vram("dynunet", batch_size=2)
        assert vram > 0

    def test_unknown_model_raises_file_not_found(self) -> None:
        """Unknown model family raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            estimate_training_vram("nonexistent_model", batch_size=1)

    def test_sam3_topolora_extrapolates_bs3(self) -> None:
        """SAM3 TopoLoRA extrapolates linearly for BS=3 from BS=1 and BS=2 data."""
        # Linear extrapolation: slope = (21.9 - 13.0) / (2 - 1) = 8.9
        # BS=3: 13.0 + (3 - 1) * 8.9 = 30.8
        vram = estimate_training_vram("sam3_topolora", batch_size=3)
        assert vram == pytest.approx(30.8, abs=0.1)


class TestCheckVramBudget:
    """check_vram_budget raises RuntimeError when estimated > available."""

    def test_sam3_topolora_bs2_exceeds_l4(self) -> None:
        """SAM3 TopoLoRA at BS=2 (21.9 GB) exceeds a simulated 20 GB GPU."""
        with pytest.raises(RuntimeError, match="batch_size=2"):
            check_vram_budget("sam3_topolora", batch_size=2, available_vram_gb=20.0)

    def test_sam3_topolora_bs1_fits_l4(self) -> None:
        """SAM3 TopoLoRA at BS=1 (13.0 GB) fits within L4's 24 GB."""
        # Should not raise
        check_vram_budget("sam3_topolora", batch_size=1, available_vram_gb=24.0)

    def test_dynunet_bs2_fits_l4(self) -> None:
        """DynUNet at BS=2 fits within L4's 24 GB."""
        # DynUNet training_gb is 3.5 at BS=2 — well within 24 GB
        check_vram_budget("dynunet", batch_size=2, available_vram_gb=24.0)

    def test_raises_with_model_family_in_message(self) -> None:
        """Error message includes model family name."""
        with pytest.raises(RuntimeError, match="sam3_topolora"):
            check_vram_budget("sam3_topolora", batch_size=2, available_vram_gb=20.0)

    def test_raises_with_available_vram_in_message(self) -> None:
        """Error message includes available VRAM."""
        with pytest.raises(RuntimeError, match="20.0 GB available"):
            check_vram_budget("sam3_topolora", batch_size=2, available_vram_gb=20.0)

    def test_sam3_hybrid_bs2_fits_l4(self) -> None:
        """SAM3 Hybrid at BS=2 (14.4 GB) fits within L4's 24 GB."""
        check_vram_budget("sam3_hybrid", batch_size=2, available_vram_gb=24.0)

    def test_sam3_hybrid_bs2_exceeds_16gb_gpu(self) -> None:
        """SAM3 Hybrid at BS=2 (14.4 GB) does not exceed a 16 GB GPU."""
        # 14.4 < 16.0, so it should pass
        check_vram_budget("sam3_hybrid", batch_size=2, available_vram_gb=16.0)
