"""T06 — RED phase: MambaVesselNetAdapter + model_builder registration.

Tests are CPU-only and use MockMamba (nn.Linear stub) so they run
without mamba-ssm installed. GPU tests are in tests/gpu_instance/.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MockMamba(nn.Module):
    """CPU stub for mamba_ssm.Mamba — reused from test_mambavesselnet_blocks."""

    def __init__(
        self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class TestMambaVesselNetAdapterInterface:
    """T06: MambaVesselNetAdapter must implement ModelAdapter ABC."""

    def _make_adapter(self) -> object:
        from minivess.adapters.mambavesselnet import MambaVesselNetAdapter
        from minivess.config.models import ModelConfig, ModelFamily

        config = ModelConfig(
            family=ModelFamily.MAMBAVESSELNET,
            name="mambavesselnet_test",
            in_channels=1,
            out_channels=2,
        )
        return MambaVesselNetAdapter(config, mamba_cls=MockMamba)

    def test_is_model_adapter(self) -> None:
        from minivess.adapters.base import ModelAdapter

        adapter = self._make_adapter()
        assert isinstance(adapter, ModelAdapter)

    def test_has_config_attribute(self) -> None:
        adapter = self._make_adapter()
        assert hasattr(adapter, "config")

    def test_has_net_attribute(self) -> None:
        from minivess.adapters.mambavesselnet import MambaVesselNetBackbone

        adapter = self._make_adapter()
        assert hasattr(adapter, "net")
        assert isinstance(adapter.net, MambaVesselNetBackbone)  # type: ignore[attr-defined]

    def test_trainable_parameters_positive(self) -> None:
        adapter = self._make_adapter()
        assert adapter.trainable_parameters() > 0  # type: ignore[attr-defined]

    def test_get_config_returns_adapter_config_info(self) -> None:
        from minivess.adapters.base import AdapterConfigInfo

        adapter = self._make_adapter()
        cfg = adapter.get_config()  # type: ignore[attr-defined]
        assert isinstance(cfg, AdapterConfigInfo)

    def test_get_config_family_is_mambavesselnet(self) -> None:
        adapter = self._make_adapter()
        cfg = adapter.get_config()  # type: ignore[attr-defined]
        assert cfg.family == "mambavesselnet"

    def test_get_config_channels(self) -> None:
        adapter = self._make_adapter()
        cfg = adapter.get_config()  # type: ignore[attr-defined]
        assert cfg.in_channels == 1
        assert cfg.out_channels == 2

    def test_get_eval_roi_size(self) -> None:
        adapter = self._make_adapter()
        roi = adapter.get_eval_roi_size()  # type: ignore[attr-defined]
        assert roi == (64, 64, 64)


class TestMambaVesselNetAdapterForward:
    """T06: forward() must return SegmentationOutput with correct shapes."""

    def test_forward_output_shape(self) -> None:
        from minivess.adapters.base import SegmentationOutput
        from minivess.adapters.mambavesselnet import MambaVesselNetAdapter
        from minivess.config.models import ModelConfig, ModelFamily

        config = ModelConfig(
            family=ModelFamily.MAMBAVESSELNET,
            name="mambavesselnet_test",
            in_channels=1,
            out_channels=2,
        )
        adapter = MambaVesselNetAdapter(config, mamba_cls=MockMamba)
        x = torch.randn(1, 1, 32, 32, 32)
        out = adapter(x)
        assert isinstance(out, SegmentationOutput)
        assert out.logits.shape == (1, 2, 32, 32, 32)
        assert out.prediction.shape == (1, 2, 32, 32, 32)

    def test_forward_prediction_sums_to_one(self) -> None:
        """Softmax predictions must sum to 1 over channel dim."""
        from minivess.adapters.mambavesselnet import MambaVesselNetAdapter
        from minivess.config.models import ModelConfig, ModelFamily

        config = ModelConfig(
            family=ModelFamily.MAMBAVESSELNET,
            name="mambavesselnet_test",
            in_channels=1,
            out_channels=2,
        )
        adapter = MambaVesselNetAdapter(config, mamba_cls=MockMamba)
        x = torch.randn(1, 1, 32, 32, 32)
        out = adapter(x)
        prob_sum = out.prediction.sum(dim=1)  # sum over class dim → should be ~1
        assert torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-5)


class TestModelBuilderRegistration:
    """T06: model_builder must dispatch ModelFamily.MAMBAVESSELNET via _require_mamba gate."""

    def test_mambavesselnet_in_registry(self) -> None:
        from minivess.adapters.model_builder import _MODEL_REGISTRY
        from minivess.config.models import ModelFamily

        assert ModelFamily.MAMBAVESSELNET in _MODEL_REGISTRY

    def test_build_adapter_raises_without_mamba(self, monkeypatch: object) -> None:
        """When mamba-ssm is missing, build_adapter raises RuntimeError."""
        import sys

        import minivess.adapters.model_builder as mb
        from minivess.config.models import ModelConfig, ModelFamily

        monkeypatch.setitem(sys.modules, "mamba_ssm", None)  # type: ignore[arg-type]

        config = ModelConfig(
            family=ModelFamily.MAMBAVESSELNET,
            name="mambavesselnet_test",
            in_channels=1,
            out_channels=2,
        )
        import pytest

        with pytest.raises(RuntimeError, match="mamba-ssm not installed"):
            mb.build_adapter(config)

    def test_method_capabilities_lists_mambavesselnet(self) -> None:
        """mambavesselnet must appear in method_capabilities.yaml."""
        from pathlib import Path

        import yaml

        caps_path = Path("configs/method_capabilities.yaml")
        caps = yaml.safe_load(caps_path.read_text(encoding="utf-8"))
        all_models: list[str] = caps.get("implemented_models", []) + caps.get(
            "not_implemented", []
        )
        assert "mambavesselnet" in all_models


class TestOnnxExportRaisesNotImplemented:
    """T06 D10: ONNX export must raise NotImplementedError."""

    def test_export_onnx_raises(self, tmp_path: object) -> None:
        from pathlib import Path

        import pytest

        from minivess.adapters.mambavesselnet import MambaVesselNetAdapter
        from minivess.config.models import ModelConfig, ModelFamily

        config = ModelConfig(
            family=ModelFamily.MAMBAVESSELNET,
            name="mambavesselnet_test",
            in_channels=1,
            out_channels=2,
        )
        adapter = MambaVesselNetAdapter(config, mamba_cls=MockMamba)
        x = torch.randn(1, 1, 32, 32, 32)
        with pytest.raises(NotImplementedError):
            adapter.export_onnx(Path(tmp_path) / "model.onnx", x)  # type: ignore[arg-type]
