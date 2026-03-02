"""Tests for multi-task ONNX export support (T19 — #246)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from minivess.adapters.base import SegmentationOutput
from minivess.adapters.multitask_adapter import AuxHeadConfig, MultiTaskAdapter
from minivess.config.deploy_config import DeployConfig


class _StubNet(torch.nn.Module):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()
        self.decoder_conv = torch.nn.Conv3d(1, 16, 3, padding=1)
        self.output_conv = torch.nn.Conv3d(16, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_conv(self.decoder_conv(x))


class _StubModel(torch.nn.Module):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()
        self.net = _StubNet()

    def forward(self, x: torch.Tensor) -> SegmentationOutput:
        logits = self.net(x)
        return SegmentationOutput(
            prediction=torch.softmax(logits, dim=1),
            logits=logits,
            metadata={},
        )

    def export_onnx(self, path: Path, example_input: torch.Tensor) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            self.net,
            example_input,
            str(path),
            input_names=["images"],
            output_names=["logits"],
            opset_version=17,
            dynamo=False,
        )


class TestMultitaskOnnxExport:
    """Tests for multi-task ONNX export support."""

    def test_multitask_onnx_mask_only(self) -> None:
        """Default export has single output (mask only)."""
        base = _StubModel()
        aux_configs = [
            AuxHeadConfig(name="sdf", head_type="regression", out_channels=1),
        ]
        adapter = MultiTaskAdapter(base_model=base, aux_head_configs=aux_configs)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.onnx"
            x = torch.randn(1, 1, 4, 8, 8)
            # Default export delegates to base (mask-only)
            adapter.export_onnx(path, x)
            assert path.exists()

    def test_multitask_onnx_validates(self) -> None:
        """ONNX Runtime inference succeeds on exported model."""
        import onnxruntime as ort

        base = _StubModel()
        aux_configs = [
            AuxHeadConfig(name="sdf", head_type="regression", out_channels=1),
        ]
        adapter = MultiTaskAdapter(base_model=base, aux_head_configs=aux_configs)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.onnx"
            x = torch.randn(1, 1, 4, 8, 8)
            adapter.export_onnx(path, x)

            session = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
            input_name = session.get_inputs()[0].name
            result = session.run(None, {input_name: x.numpy()})
            assert result[0].shape == (1, 2, 4, 8, 8)

    def test_standard_model_export_unchanged(self) -> None:
        """Non-multitask ONNX export untouched."""
        base = _StubModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.onnx"
            x = torch.randn(1, 1, 4, 8, 8)
            base.export_onnx(path, x)
            assert path.exists()

    def test_deploy_config_export_aux_heads(self) -> None:
        """DeployConfig has export_aux_heads field defaulting to False."""
        config = DeployConfig(
            mlruns_dir=Path("mlruns"),
            output_dir=Path("deploy"),
        )
        assert config.export_aux_heads is False

    def test_deploy_config_export_aux_heads_enabled(self) -> None:
        """DeployConfig supports export_aux_heads=True."""
        config = DeployConfig(
            mlruns_dir=Path("mlruns"),
            output_dir=Path("deploy"),
            export_aux_heads=True,
        )
        assert config.export_aux_heads is True
