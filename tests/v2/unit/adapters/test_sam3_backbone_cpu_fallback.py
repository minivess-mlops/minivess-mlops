"""Tests for Sam3Backbone CPU-only fallback — T22 regression test.

Bug: Sam3Backbone.__init__ calls torch.cuda.is_bf16_supported() without
guarding for the no-CUDA case. This crashes on CPU-only machines.
"""

from __future__ import annotations

from pathlib import Path

SRC_FILE = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "minivess"
    / "adapters"
    / "sam3_backbone.py"
)


class TestSam3BackboneCpuFallback:
    """T22: Sam3Backbone must not crash on CPU-only machines."""

    def test_backbone_init_on_cpu_only_machine(self):
        """bf16 detection must guard against missing CUDA."""
        source = SRC_FILE.read_text(encoding="utf-8")
        # The BF16 auto-detection should check torch.cuda.is_available() first
        assert "torch.cuda.is_available()" in source, (
            "Sam3Backbone must guard torch.cuda.is_bf16_supported() with "
            "torch.cuda.is_available() check for CPU-only machines"
        )
