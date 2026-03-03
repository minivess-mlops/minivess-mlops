"""Tests for model-agnostic ONNX export including SAM3 (T10).

Verifies that deploy_onnx_export.py works with SAM3 adapters,
not just DynUNet. CI-compatible (uses stub encoders).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from pathlib import Path


class TestModelAgnosticCheckpointLoading:
    """Test _load_model_from_checkpoint with different model families."""

    def _make_champion(
        self,
        tmp_path: Path,
        model_family: str,
    ) -> Any:
        """Create a mock ChampionModel with a real checkpoint."""
        from minivess.adapters.model_builder import build_adapter
        from minivess.config.models import ModelConfig, ModelFamily
        from minivess.pipeline.deploy_champion_discovery import ChampionModel

        config = ModelConfig(
            family=ModelFamily(model_family),
            name=model_family,
            in_channels=1,
            out_channels=2,
        )
        adapter = build_adapter(config, use_stub=True)

        # Save checkpoint
        ckpt_path = tmp_path / f"{model_family}_ckpt.pth"
        adapter.save_checkpoint(ckpt_path)

        return ChampionModel(
            run_id="test_run",
            experiment_id="test_exp",
            category="balanced",
            metrics={"dsc": 0.5},
            checkpoint_path=ckpt_path,
            model_config={
                "model_family": model_family,
                "in_channels": "1",
                "out_channels": "2",
            },
        )

    def test_load_sam3_vanilla_checkpoint(self, tmp_path: Path) -> None:
        from minivess.pipeline.deploy_onnx_export import _load_model_from_checkpoint

        champion = self._make_champion(tmp_path, "sam3_vanilla")
        model = _load_model_from_checkpoint(champion)
        assert model is not None

        # Verify forward works with 3D input
        x = torch.randn(1, 1, 16, 64, 64)
        with torch.no_grad():
            output = model(x)
        assert output is not None

    def test_load_sam3_topolora_checkpoint(self, tmp_path: Path) -> None:
        from minivess.pipeline.deploy_onnx_export import _load_model_from_checkpoint

        champion = self._make_champion(tmp_path, "sam3_topolora")
        model = _load_model_from_checkpoint(champion)
        assert model is not None

    def test_load_sam3_hybrid_checkpoint(self, tmp_path: Path) -> None:
        from minivess.pipeline.deploy_onnx_export import _load_model_from_checkpoint

        champion = self._make_champion(tmp_path, "sam3_hybrid")
        model = _load_model_from_checkpoint(champion)
        assert model is not None

    def test_load_dynunet_still_works(self, tmp_path: Path) -> None:
        """Regression: DynUNet loading must still work."""
        from minivess.pipeline.deploy_onnx_export import _load_model_from_checkpoint

        champion = self._make_champion(tmp_path, "dynunet")
        model = _load_model_from_checkpoint(champion)
        assert model is not None


class TestSam3OnnxExport:
    """Test ONNX export for SAM3 adapters."""

    def test_sam3_vanilla_onnx_export(self, tmp_path: Path) -> None:
        from minivess.adapters.model_builder import build_adapter
        from minivess.config.models import ModelConfig, ModelFamily

        config = ModelConfig(
            family=ModelFamily.SAM3_VANILLA,
            name="sam3_vanilla",
            in_channels=1,
            out_channels=2,
        )
        adapter = build_adapter(config, use_stub=True)
        onnx_path = tmp_path / "sam3_vanilla.onnx"
        example_input = torch.randn(1, 1, 16, 64, 64)
        adapter.export_onnx(onnx_path, example_input)
        assert onnx_path.exists()
        assert onnx_path.stat().st_size > 0

    def test_sam3_onnx_valid_with_onnxruntime(self, tmp_path: Path) -> None:
        """Validate exported ONNX with onnxruntime inference."""
        import numpy as np

        try:
            import onnxruntime as ort
        except ImportError:
            import pytest

            pytest.skip("onnxruntime not installed")

        from minivess.adapters.model_builder import build_adapter
        from minivess.config.models import ModelConfig, ModelFamily

        config = ModelConfig(
            family=ModelFamily.SAM3_VANILLA,
            name="sam3_vanilla",
            in_channels=1,
            out_channels=2,
        )
        adapter = build_adapter(config, use_stub=True)
        onnx_path = tmp_path / "sam3_test.onnx"
        example_input = torch.randn(1, 1, 16, 64, 64)
        adapter.export_onnx(onnx_path, example_input)

        session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name
        dummy = np.random.rand(1, 1, 16, 64, 64).astype(np.float32)
        outputs = session.run(None, {input_name: dummy})
        assert len(outputs) > 0
        assert outputs[0].shape[0] == 1
