"""Tests for typed return types replacing dict[str, Any] (Code Review R4.1).

Validates that get_config() returns AdapterConfigInfo dataclass,
ONNX predict() returns OnnxPrediction, and get_metadata() returns
OnnxModelMetadata — all with proper attribute access.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# T1: AdapterConfigInfo dataclass
# ---------------------------------------------------------------------------


class TestAdapterConfigInfo:
    """Test the AdapterConfigInfo typed return for get_config()."""

    def test_dataclass_has_required_fields(self) -> None:
        """AdapterConfigInfo should expose family, name, and extras."""
        from minivess.adapters.base import AdapterConfigInfo

        info = AdapterConfigInfo(family="dynunet", name="test")
        assert info.family == "dynunet"
        assert info.name == "test"
        assert info.extras == {}

    def test_optional_fields_default_to_none(self) -> None:
        """in_channels, out_channels, trainable_params default to None."""
        from minivess.adapters.base import AdapterConfigInfo

        info = AdapterConfigInfo(family="test", name="m")
        assert info.in_channels is None
        assert info.out_channels is None
        assert info.trainable_params is None

    def test_extras_captures_adapter_specific_fields(self) -> None:
        """Adapter-specific fields should go in extras dict."""
        from minivess.adapters.base import AdapterConfigInfo

        info = AdapterConfigInfo(
            family="dynunet",
            name="test",
            in_channels=1,
            out_channels=2,
            trainable_params=1000,
            extras={"init_filters": 32, "blocks_down": (1, 2, 2, 4)},
        )
        assert info.extras["init_filters"] == 32
        assert info.extras["blocks_down"] == (1, 2, 2, 4)

    def test_to_dict_round_trip(self) -> None:
        """to_dict() should produce a flat serializable dict."""
        from minivess.adapters.base import AdapterConfigInfo

        info = AdapterConfigInfo(
            family="dynunet",
            name="test",
            in_channels=1,
            out_channels=2,
            trainable_params=500,
            extras={"init_filters": 32},
        )
        d = info.to_dict()
        assert d["family"] == "dynunet"
        assert d["name"] == "test"
        assert d["in_channels"] == 1
        assert d["init_filters"] == 32
        assert isinstance(d, dict)


# ---------------------------------------------------------------------------
# T2: Adapters return AdapterConfigInfo
# ---------------------------------------------------------------------------


class TestAdaptersReturnConfigInfo:
    """Test that concrete adapters return AdapterConfigInfo from get_config()."""

    def test_dynunet_returns_config_info(self) -> None:
        """DynUNetAdapter.get_config() should return AdapterConfigInfo."""
        from minivess.adapters.base import AdapterConfigInfo
        from minivess.adapters.dynunet import DynUNetAdapter
        from minivess.config.models import ModelConfig, ModelFamily

        model = DynUNetAdapter(
            ModelConfig(
                family=ModelFamily.MONAI_DYNUNET,
                name="test",
                in_channels=1,
                out_channels=2,
            )
        )
        cfg = model.get_config()
        assert isinstance(cfg, AdapterConfigInfo)
        assert cfg.extras["filters"] == [32, 64, 128, 256]

    def test_comma_returns_config_info(self) -> None:
        """CommaAdapter.get_config() should return AdapterConfigInfo."""
        from minivess.adapters.base import AdapterConfigInfo
        from minivess.adapters.comma import CommaAdapter
        from minivess.config.models import ModelConfig, ModelFamily

        model = CommaAdapter(
            ModelConfig(
                family=ModelFamily.COMMA_MAMBA,
                name="test",
                in_channels=1,
                out_channels=2,
            )
        )
        cfg = model.get_config()
        assert isinstance(cfg, AdapterConfigInfo)
        assert cfg.extras["d_state"] == 16

    def test_lora_returns_config_info_with_lora_fields(self) -> None:
        """LoraModelAdapter.get_config() should include LoRA extras."""
        from minivess.adapters.base import AdapterConfigInfo
        from minivess.adapters.dynunet import DynUNetAdapter
        from minivess.adapters.lora import LoraModelAdapter
        from minivess.config.models import ModelConfig, ModelFamily

        base = DynUNetAdapter(
            ModelConfig(
                family=ModelFamily.MONAI_DYNUNET,
                name="test",
                in_channels=1,
                out_channels=2,
            )
        )
        lora = LoraModelAdapter(base, lora_rank=8)
        cfg = lora.get_config()
        assert isinstance(cfg, AdapterConfigInfo)
        assert cfg.extras["lora_rank"] == 8
        assert "lora_applied" in cfg.extras


# ---------------------------------------------------------------------------
# T3: OnnxPrediction dataclass
# ---------------------------------------------------------------------------


class TestOnnxPrediction:
    """Test the OnnxPrediction typed return."""

    def test_dataclass_has_required_fields(self) -> None:
        """OnnxPrediction should have segmentation, probabilities, shape."""
        import numpy as np

        from minivess.serving.onnx_inference import OnnxPrediction

        seg = np.zeros((1, 8, 8, 8), dtype=np.int64)
        probs = np.ones((1, 2, 8, 8, 8), dtype=np.float32) * 0.5
        pred = OnnxPrediction(
            segmentation=seg,
            probabilities=probs,
            shape=[1, 8, 8, 8],
        )
        assert pred.shape == [1, 8, 8, 8]
        assert pred.segmentation is seg
        assert pred.probabilities is probs


# ---------------------------------------------------------------------------
# T4: OnnxModelMetadata dataclass
# ---------------------------------------------------------------------------


class TestOnnxModelMetadata:
    """Test the OnnxModelMetadata typed return."""

    def test_dataclass_has_inputs_outputs(self) -> None:
        """OnnxModelMetadata should have inputs and outputs lists."""
        from minivess.serving.onnx_inference import OnnxModelMetadata, OnnxTensorSpec

        meta = OnnxModelMetadata(
            inputs=[
                OnnxTensorSpec(name="images", shape=[1, 1, 64, 64, 64], type="float")
            ],
            outputs=[
                OnnxTensorSpec(name="logits", shape=[1, 2, 64, 64, 64], type="float")
            ],
        )
        assert len(meta.inputs) == 1
        assert meta.inputs[0].name == "images"
        assert len(meta.outputs) == 1

    def test_tensor_spec_fields(self) -> None:
        """OnnxTensorSpec should expose name, shape, type."""
        from minivess.serving.onnx_inference import OnnxTensorSpec

        spec = OnnxTensorSpec(name="x", shape=[1, 3], type="float")
        assert spec.name == "x"
        assert spec.shape == [1, 3]
        assert spec.type == "float"
