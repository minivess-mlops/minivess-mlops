"""Tests for SAM3 GPU VRAM enforcement check (T-02, updated for mode split).

Verifies that check_sam3_vram() raises RuntimeError when GPU VRAM is below
the mode-appropriate minimum:
- mode="training"  → 16 GB minimum (LoRA / fine-tuning)
- mode="inference" → 6 GB minimum (single-image BF16 forward pass)

Sources for thresholds: GitHub Issues #200, #235, #307 at facebookresearch/sam3;
debuggercafe.com SAM3 memory benchmarks.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _make_hw(gpu_vram_mb: int, gpu_name: str = "Test GPU") -> MagicMock:
    """Create a mock HardwareBudget with given GPU VRAM."""
    hw = MagicMock()
    hw.gpu_vram_mb = gpu_vram_mb
    hw.gpu_name = gpu_name
    return hw


class TestCheckSam3VramTraining:
    """check_sam3_vram(mode='training') — 16 GB minimum."""

    def test_raises_when_vram_below_16gb(self) -> None:
        """RuntimeError raised when VRAM is 8 GB (below 16 GB training minimum)."""
        from minivess.adapters.sam3_vram_check import check_sam3_vram

        hw = _make_hw(gpu_vram_mb=8192, gpu_name="RTX 2070 Super")
        with (
            patch(
                "minivess.adapters.sam3_vram_check.detect_hardware",
                return_value=hw,
            ),
            pytest.raises(RuntimeError),
        ):
            check_sam3_vram(mode="training")

    def test_passes_when_vram_meets_16gb(self) -> None:
        """No exception raised when VRAM is 20 GB (above 16 GB minimum)."""
        from minivess.adapters.sam3_vram_check import check_sam3_vram

        hw = _make_hw(gpu_vram_mb=20480, gpu_name="A100-40GB")
        with patch(
            "minivess.adapters.sam3_vram_check.detect_hardware",
            return_value=hw,
        ):
            check_sam3_vram(mode="training")  # Must not raise

    def test_passes_at_exactly_16gb_boundary(self) -> None:
        """No exception raised at exactly 16384 MB (boundary is inclusive)."""
        from minivess.adapters.sam3_vram_check import check_sam3_vram

        hw = _make_hw(gpu_vram_mb=16384, gpu_name="RTX 4090")
        with patch(
            "minivess.adapters.sam3_vram_check.detect_hardware",
            return_value=hw,
        ):
            check_sam3_vram(mode="training")  # Must not raise

    def test_error_message_includes_detected_vram(self) -> None:
        """Error message must include detected VRAM in GB."""
        from minivess.adapters.sam3_vram_check import check_sam3_vram

        hw = _make_hw(gpu_vram_mb=8192, gpu_name="RTX 2070 Super")
        with (
            patch(
                "minivess.adapters.sam3_vram_check.detect_hardware",
                return_value=hw,
            ),
            pytest.raises(RuntimeError, match="8.0 GB"),
        ):
            check_sam3_vram(mode="training")

    def test_error_message_includes_variant_name(self) -> None:
        """Error message must include the variant name."""
        from minivess.adapters.sam3_vram_check import check_sam3_vram

        hw = _make_hw(gpu_vram_mb=4096, gpu_name="GTX 1080")
        with (
            patch(
                "minivess.adapters.sam3_vram_check.detect_hardware",
                return_value=hw,
            ),
            pytest.raises(RuntimeError, match="sam3_topolora"),
        ):
            check_sam3_vram(variant="sam3_topolora", mode="training")

    def test_raises_when_no_gpu_detected(self) -> None:
        """RuntimeError raised when VRAM is 0 (no GPU)."""
        from minivess.adapters.sam3_vram_check import check_sam3_vram

        hw = _make_hw(gpu_vram_mb=0, gpu_name="")
        with (
            patch(
                "minivess.adapters.sam3_vram_check.detect_hardware",
                return_value=hw,
            ),
            pytest.raises(RuntimeError),
        ):
            check_sam3_vram(mode="training")

    def test_default_mode_is_training(self) -> None:
        """Default mode is 'training' — 8 GB raises."""
        from minivess.adapters.sam3_vram_check import check_sam3_vram

        hw = _make_hw(gpu_vram_mb=8192)
        with (
            patch(
                "minivess.adapters.sam3_vram_check.detect_hardware",
                return_value=hw,
            ),
            pytest.raises(RuntimeError),
        ):
            check_sam3_vram()  # No mode arg → training → 8 GB fails


class TestCheckSam3VramInference:
    """check_sam3_vram(mode='inference') — 6 GB minimum."""

    def test_8gb_passes_for_inference(self) -> None:
        """RTX 2070 Super (8 GB) passes the inference gate."""
        from minivess.adapters.sam3_vram_check import check_sam3_vram

        hw = _make_hw(gpu_vram_mb=8192, gpu_name="RTX 2070 Super")
        with patch(
            "minivess.adapters.sam3_vram_check.detect_hardware",
            return_value=hw,
        ):
            check_sam3_vram(mode="inference")  # Must not raise

    def test_passes_at_exactly_6gb_boundary(self) -> None:
        """No exception at exactly 6144 MB (inclusive boundary)."""
        from minivess.adapters.sam3_vram_check import check_sam3_vram

        hw = _make_hw(gpu_vram_mb=6144, gpu_name="RTX 3060")
        with patch(
            "minivess.adapters.sam3_vram_check.detect_hardware",
            return_value=hw,
        ):
            check_sam3_vram(mode="inference")  # Must not raise

    def test_raises_below_6gb_for_inference(self) -> None:
        """RuntimeError when VRAM is 4 GB (below 6 GB inference minimum)."""
        from minivess.adapters.sam3_vram_check import check_sam3_vram

        hw = _make_hw(gpu_vram_mb=4096, gpu_name="GTX 1650")
        with (
            patch(
                "minivess.adapters.sam3_vram_check.detect_hardware",
                return_value=hw,
            ),
            pytest.raises(RuntimeError),
        ):
            check_sam3_vram(mode="inference")


class TestCheckSam3VramBatchSize:
    """check_sam3_vram with batch_size parameter (Task 2.4)."""

    def test_vram_check_batch_size_1_passes_on_l4(self) -> None:
        """L4 (24 GB) passes at batch_size=1 for SAM3 TopoLoRA."""
        from minivess.adapters.sam3_vram_check import check_sam3_vram

        hw = _make_hw(gpu_vram_mb=24576, gpu_name="NVIDIA L4")
        with patch(
            "minivess.adapters.sam3_vram_check.detect_hardware",
            return_value=hw,
        ):
            # Should not raise — 24 GB > 16 GB base threshold, BS=1 fits
            check_sam3_vram(variant="sam3_topolora", mode="training", batch_size=1)

    def test_vram_check_batch_size_2_raises_on_limited_gpu(self) -> None:
        """SAM3 TopoLoRA at BS=2 (~21.9 GB) raises on a 20 GB GPU."""
        from minivess.adapters.sam3_vram_check import check_sam3_vram

        # 20 GB GPU: passes base 16 GB threshold but fails batch-size check
        hw = _make_hw(gpu_vram_mb=20480, gpu_name="Test 20GB GPU")
        with (
            patch(
                "minivess.adapters.sam3_vram_check.detect_hardware",
                return_value=hw,
            ),
            pytest.raises(RuntimeError, match="batch_size=2"),
        ):
            check_sam3_vram(variant="sam3_topolora", mode="training", batch_size=2)

    def test_vram_check_backward_compatible(self) -> None:
        """Calling check_sam3_vram() without batch_size still works (default=1)."""
        from minivess.adapters.sam3_vram_check import check_sam3_vram

        hw = _make_hw(gpu_vram_mb=24576, gpu_name="NVIDIA L4")
        with patch(
            "minivess.adapters.sam3_vram_check.detect_hardware",
            return_value=hw,
        ):
            # Default batch_size=1 — should not raise on 24 GB GPU
            check_sam3_vram(variant="sam3_topolora", mode="training")

    def test_batch_size_ignored_for_inference_mode(self) -> None:
        """batch_size parameter has no effect in inference mode."""
        from minivess.adapters.sam3_vram_check import check_sam3_vram

        hw = _make_hw(gpu_vram_mb=8192, gpu_name="RTX 2070 Super")
        with patch(
            "minivess.adapters.sam3_vram_check.detect_hardware",
            return_value=hw,
        ):
            # Inference only needs 6 GB — 8 GB passes regardless of batch_size
            check_sam3_vram(mode="inference", batch_size=4)

    def test_batch_size_1_same_as_no_batch_size(self) -> None:
        """batch_size=1 produces the same result as omitting batch_size."""
        from minivess.adapters.sam3_vram_check import check_sam3_vram

        hw = _make_hw(gpu_vram_mb=24576, gpu_name="NVIDIA L4")
        with patch(
            "minivess.adapters.sam3_vram_check.detect_hardware",
            return_value=hw,
        ):
            # Both should pass without raising
            check_sam3_vram(variant="sam3_topolora", mode="training")
            check_sam3_vram(variant="sam3_topolora", mode="training", batch_size=1)


class TestCheckSam3VramEdgeCases:
    """Edge cases and invalid inputs."""

    def test_invalid_mode_raises_value_error(self) -> None:
        """ValueError raised for unknown mode."""
        from minivess.adapters.sam3_vram_check import check_sam3_vram

        hw = _make_hw(gpu_vram_mb=40960)
        with (
            patch(
                "minivess.adapters.sam3_vram_check.detect_hardware",
                return_value=hw,
            ),
            pytest.raises(ValueError, match="mode must be"),
        ):
            check_sam3_vram(mode="finetuning")  # Invalid string
