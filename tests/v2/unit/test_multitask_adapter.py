"""Tests for generic MultiTaskAdapter (T8 — #233)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from minivess.adapters.base import SegmentationOutput
from minivess.adapters.multitask_adapter import AuxHeadConfig, MultiTaskAdapter

pytestmark = pytest.mark.model_loading


class _StubNet(torch.nn.Module):  # type: ignore[misc]
    """Inner network module (avoids circular reference with self.net = self)."""

    def __init__(self, in_ch: int = 1, out_ch: int = 2) -> None:
        super().__init__()
        self.decoder_conv = torch.nn.Conv3d(in_ch, 16, 3, padding=1)
        self.output_conv = torch.nn.Conv3d(16, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.decoder_conv(x)
        return self.output_conv(features)


class _StubModel(torch.nn.Module):  # type: ignore[misc]
    """Minimal stub model mimicking DynUNet for testing."""

    def __init__(self, in_ch: int = 1, out_ch: int = 2) -> None:
        super().__init__()
        self.net = _StubNet(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> SegmentationOutput:
        logits = self.net(x)
        return SegmentationOutput(
            prediction=torch.softmax(logits, dim=1),
            logits=logits,
            metadata={},
        )

    # Minimal ModelAdapter interface
    def get_config(self):  # noqa: ANN201
        from minivess.adapters.base import AdapterConfigInfo

        return AdapterConfigInfo(family="stub", name="stub")

    def load_checkpoint(self, path):  # noqa: ANN001, ANN201
        pass

    def save_checkpoint(self, path):  # noqa: ANN001, ANN201
        pass

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def export_onnx(self, path, example_input):  # noqa: ANN001, ANN201
        pass

    @property
    def config(self):  # noqa: ANN201
        from minivess.config.models import ModelConfig, ModelFamily

        return ModelConfig(family=ModelFamily.MONAI_DYNUNET, name="stub")


def _make_adapter(
    aux_configs: list[AuxHeadConfig] | None = None,
) -> MultiTaskAdapter:
    """Create a MultiTaskAdapter wrapping a stub model."""
    base = _StubModel()
    return MultiTaskAdapter(base_model=base, aux_head_configs=aux_configs or [])


class TestMultiTaskAdapter:
    """Tests for MultiTaskAdapter."""

    def test_multitask_adapter_generic_two_heads(self) -> None:
        """Works with 2 config-driven aux heads."""
        configs = [
            AuxHeadConfig(name="head_a", head_type="regression", out_channels=1),
            AuxHeadConfig(name="head_b", head_type="regression", out_channels=1),
        ]
        adapter = _make_adapter(configs)
        x = torch.randn(1, 1, 4, 8, 8)
        out = adapter(x)
        assert "head_a" in out.metadata
        assert "head_b" in out.metadata

    def test_multitask_adapter_generic_single_head(self) -> None:
        """Works with 1 aux head."""
        configs = [
            AuxHeadConfig(name="single", head_type="regression", out_channels=1),
        ]
        adapter = _make_adapter(configs)
        x = torch.randn(1, 1, 4, 8, 8)
        out = adapter(x)
        assert "single" in out.metadata

    def test_multitask_adapter_generic_four_heads(self) -> None:
        """Works with 4 aux heads (stress test)."""
        configs = [
            AuxHeadConfig(name=f"head_{i}", head_type="regression", out_channels=1)
            for i in range(4)
        ]
        adapter = _make_adapter(configs)
        x = torch.randn(1, 1, 4, 8, 8)
        out = adapter(x)
        for i in range(4):
            assert f"head_{i}" in out.metadata

    def test_multitask_adapter_regression_head_shape(self) -> None:
        """[B, 1, D, H, W] for regression."""
        configs = [
            AuxHeadConfig(name="reg", head_type="regression", out_channels=1),
        ]
        adapter = _make_adapter(configs)
        x = torch.randn(2, 1, 4, 8, 8)
        out = adapter(x)
        assert out.metadata["reg"].shape == (2, 1, 4, 8, 8)

    def test_multitask_adapter_classification_head_shape(self) -> None:
        """[B, N, D, H, W] for classification."""
        configs = [
            AuxHeadConfig(name="cls", head_type="classification", out_channels=3),
        ]
        adapter = _make_adapter(configs)
        x = torch.randn(2, 1, 4, 8, 8)
        out = adapter(x)
        assert out.metadata["cls"].shape == (2, 3, 4, 8, 8)

    def test_multitask_adapter_gradient_flows_to_all_heads(self) -> None:
        """All heads get gradients."""
        configs = [
            AuxHeadConfig(name="a", head_type="regression", out_channels=1),
            AuxHeadConfig(name="b", head_type="regression", out_channels=1),
        ]
        adapter = _make_adapter(configs)
        x = torch.randn(1, 1, 4, 8, 8)
        out = adapter(x)
        loss = out.logits.sum() + out.metadata["a"].sum() + out.metadata["b"].sum()
        loss.backward()
        for name, p in adapter.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_multitask_adapter_checkpoint_roundtrip(self) -> None:
        """Save/load preserves all heads."""
        configs = [
            AuxHeadConfig(name="sdf", head_type="regression", out_channels=1),
        ]
        adapter = _make_adapter(configs)
        x = torch.randn(1, 1, 4, 8, 8)
        out1 = adapter(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ckpt.pt"
            adapter.save_checkpoint(ckpt_path)

            adapter2 = _make_adapter(configs)
            adapter2.load_checkpoint(ckpt_path)
            out2 = adapter2(x)

        torch.testing.assert_close(out1.logits, out2.logits)
        torch.testing.assert_close(out1.metadata["sdf"], out2.metadata["sdf"])

    def test_multitask_adapter_onnx_export_mask_only(self) -> None:
        """Default ONNX export delegates to base model (mask only)."""
        configs = [
            AuxHeadConfig(name="sdf", head_type="regression", out_channels=1),
        ]
        adapter = _make_adapter(configs)
        # Just verify export_onnx method exists and is callable
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.onnx"
            x = torch.randn(1, 1, 4, 8, 8)
            adapter.export_onnx(path, x)

    def test_multitask_adapter_config_driven_no_code_change(self) -> None:
        """New head from config only — no code changes needed."""
        # Create adapter with a completely novel head name
        configs = [
            AuxHeadConfig(
                name="novel_task_xyz",
                head_type="segmentation",
                out_channels=5,
            ),
        ]
        adapter = _make_adapter(configs)
        x = torch.randn(1, 1, 4, 8, 8)
        out = adapter(x)
        assert "novel_task_xyz" in out.metadata
        assert out.metadata["novel_task_xyz"].shape[1] == 5

    def test_get_output_channels_in_utils(self) -> None:
        """Factored utility function works."""
        from minivess.adapters.utils import get_output_channels

        conv = torch.nn.Conv3d(16, 32, 3)
        assert get_output_channels(conv) == 32

    def test_multitask_adapter_wraps_dynunet(self) -> None:
        """Works with DynUNet-like base model."""
        configs = [
            AuxHeadConfig(name="sdf", head_type="regression", out_channels=1),
        ]
        adapter = _make_adapter(configs)
        x = torch.randn(1, 1, 4, 8, 8)
        out = adapter(x)
        assert isinstance(out, SegmentationOutput)
        assert out.logits.shape == (1, 2, 4, 8, 8)
