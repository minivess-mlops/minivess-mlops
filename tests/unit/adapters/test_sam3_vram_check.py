"""Tests for SAM3 GPU VRAM enforcement check (T-02).

Verifies that check_sam3_vram() raises RuntimeError when GPU VRAM
is below the 16 GB minimum, and passes when VRAM meets or exceeds it.
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


class TestCheckSam3Vram:
    """check_sam3_vram() raises when GPU VRAM < 16 GB."""

    def test_raises_when_vram_below_16gb(self) -> None:
        """RuntimeError raised when VRAM is 8 GB (below 16 GB minimum)."""
        from minivess.adapters.sam3_vram_check import check_sam3_vram

        hw = _make_hw(gpu_vram_mb=8192, gpu_name="RTX 3070")
        with (
            patch(
                "minivess.adapters.sam3_vram_check.detect_hardware",
                return_value=hw,
            ),
            pytest.raises(RuntimeError),
        ):
            check_sam3_vram()

    def test_passes_when_vram_meets_16gb(self) -> None:
        """No exception raised when VRAM is 20 GB (above minimum)."""
        from minivess.adapters.sam3_vram_check import check_sam3_vram

        hw = _make_hw(gpu_vram_mb=20480, gpu_name="A100-40GB")
        with patch(
            "minivess.adapters.sam3_vram_check.detect_hardware",
            return_value=hw,
        ):
            check_sam3_vram()  # Must not raise

    def test_passes_at_exactly_16gb_boundary(self) -> None:
        """No exception raised at exactly 16384 MB (boundary is inclusive)."""
        from minivess.adapters.sam3_vram_check import check_sam3_vram

        hw = _make_hw(gpu_vram_mb=16384, gpu_name="RTX 4090")
        with patch(
            "minivess.adapters.sam3_vram_check.detect_hardware",
            return_value=hw,
        ):
            check_sam3_vram()  # Must not raise

    def test_error_message_includes_detected_vram(self) -> None:
        """Error message must include detected VRAM in GB."""
        from minivess.adapters.sam3_vram_check import check_sam3_vram

        hw = _make_hw(gpu_vram_mb=8192, gpu_name="RTX 3070")
        with (
            patch(
                "minivess.adapters.sam3_vram_check.detect_hardware",
                return_value=hw,
            ),
            pytest.raises(RuntimeError, match="8.0 GB"),
        ):
            check_sam3_vram()

    def test_error_message_includes_variant_name(self) -> None:
        """Error message must include the variant name passed to check_sam3_vram()."""
        from minivess.adapters.sam3_vram_check import check_sam3_vram

        hw = _make_hw(gpu_vram_mb=4096, gpu_name="GTX 1080")
        with (
            patch(
                "minivess.adapters.sam3_vram_check.detect_hardware",
                return_value=hw,
            ),
            pytest.raises(RuntimeError, match="sam3_topolora"),
        ):
            check_sam3_vram(variant="sam3_topolora")

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
            check_sam3_vram()
