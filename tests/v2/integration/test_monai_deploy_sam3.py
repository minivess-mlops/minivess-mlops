"""Tests for MONAI Deploy MAP operator with SAM3 ONNX models (T12).

Verifies that MiniVessInferenceOperator and MiniVessSegApp work
with SAM3-exported ONNX models. CI-compatible (duck-typed, no SDK).

IMPORTANT: SAM3 tests require real pretrained weights (GPU ≥16 GB).
They are skipped in CI where SAM3 is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from minivess.adapters.model_builder import _sam3_package_available

if TYPE_CHECKING:
    from pathlib import Path


def _insufficient_vram() -> bool:
    """Check if GPU VRAM is too small for SAM3 (needs >= 16 GB)."""
    if not torch.cuda.is_available():
        return True
    return torch.cuda.get_device_properties(0).total_memory < 16 * 1024**3


_sam3_skip = pytest.mark.skipif(
    not _sam3_package_available(), reason="SAM3 not installed"
)
_vram_skip = pytest.mark.skipif(
    _insufficient_vram(), reason="SAM3 requires >= 16 GB VRAM"
)


@_sam3_skip
@_vram_skip
class TestMonaiDeploySam3:
    """Test MAP operator with SAM3 ONNX models."""

    def _export_sam3_onnx(self, tmp_path: Path) -> Path:
        """Export a SAM3 vanilla adapter to ONNX."""
        from minivess.adapters.model_builder import build_adapter
        from minivess.config.models import ModelConfig, ModelFamily

        config = ModelConfig(
            family=ModelFamily.SAM3_VANILLA,
            name="sam3_vanilla",
            in_channels=1,
            out_channels=2,
        )
        adapter = build_adapter(config)
        onnx_path = tmp_path / "sam3_map_test.onnx"
        example_input = torch.randn(1, 1, 4, 32, 32)
        adapter.export_onnx(onnx_path, example_input)
        return onnx_path

    def test_inference_operator_creates(self, tmp_path: Path) -> None:
        from minivess.serving.monai_deploy_app import MiniVessInferenceOperator

        onnx_path = self._export_sam3_onnx(tmp_path)
        operator = MiniVessInferenceOperator(model_path=onnx_path)
        assert operator is not None
        assert operator.model_path == onnx_path

    def test_inference_operator_process(self, tmp_path: Path) -> None:
        from minivess.serving.monai_deploy_app import MiniVessInferenceOperator

        onnx_path = self._export_sam3_onnx(tmp_path)
        operator = MiniVessInferenceOperator(model_path=onnx_path)

        # 3D input: (B, C, D, H, W)
        volume = np.random.rand(1, 1, 4, 32, 32).astype(np.float32)
        result = operator.process(volume)

        assert "segmentation" in result
        assert "probabilities" in result
        assert result["segmentation"].shape[0] == 1  # batch
        assert result["probabilities"].shape[1] == 2  # 2 classes

    def test_seg_app_compose(self, tmp_path: Path) -> None:
        from minivess.serving.monai_deploy_app import MiniVessSegApp

        onnx_path = self._export_sam3_onnx(tmp_path)
        app = MiniVessSegApp(model_path=onnx_path)
        operators = app.compose()
        assert len(operators) >= 1
        assert hasattr(operators[0], "process")
