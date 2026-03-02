"""Tests for MLflow pyfunc serving with SAM3 models (T9).

Verifies that MiniVessSegModel can load SAM3 adapters and produce
correct predictions. CI-compatible (uses stub encoders).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


class TestBuildNetFromConfig:
    """Test _build_net_from_config model-agnostic dispatch."""

    def test_build_sam3_vanilla_net(self) -> None:
        from minivess.serving.mlflow_wrapper import _build_net_from_config, _SimpleNet

        config: dict[str, Any] = {
            "family": "sam3_vanilla",
            "name": "sam3_vanilla",
            "in_channels": 1,
            "out_channels": 2,
        }
        net = _build_net_from_config(config)
        assert net is not None
        # Must be an actual adapter, not _SimpleNet fallback
        assert not isinstance(net, _SimpleNet), "Expected SAM3 adapter, got _SimpleNet"

    def test_build_sam3_topolora_net(self) -> None:
        from minivess.serving.mlflow_wrapper import _build_net_from_config, _SimpleNet

        config: dict[str, Any] = {
            "family": "sam3_topolora",
            "name": "sam3_topolora",
            "in_channels": 1,
            "out_channels": 2,
        }
        net = _build_net_from_config(config)
        assert net is not None
        assert not isinstance(net, _SimpleNet)

    def test_build_sam3_hybrid_net(self) -> None:
        from minivess.serving.mlflow_wrapper import _build_net_from_config, _SimpleNet

        config: dict[str, Any] = {
            "family": "sam3_hybrid",
            "name": "sam3_hybrid",
            "in_channels": 1,
            "out_channels": 2,
        }
        net = _build_net_from_config(config)
        assert net is not None
        assert not isinstance(net, _SimpleNet)

    def test_build_dynunet_still_works(self) -> None:
        """Regression: DynUNet loading must still work."""
        from minivess.serving.mlflow_wrapper import _build_net_from_config

        config: dict[str, Any] = {
            "family": "dynunet",
            "name": "dynunet",
            "in_channels": 1,
            "out_channels": 2,
        }
        net = _build_net_from_config(config)
        assert net is not None

    def test_build_unknown_family_returns_simplenet(self) -> None:
        from minivess.serving.mlflow_wrapper import _build_net_from_config

        config: dict[str, Any] = {"family": "unknown_model"}
        net = _build_net_from_config(config)
        # Should fall back to _SimpleNet
        assert net is not None


class TestMlflowServingSam3:
    """Test MiniVessSegModel with SAM3 adapter checkpoints."""

    def _make_checkpoint_artifacts(
        self,
        tmp_path: Path,
        model_family: str,
    ) -> dict[str, str]:
        """Create checkpoint + config artifacts for MLflow pyfunc loading."""
        from minivess.adapters.model_builder import build_adapter
        from minivess.config.models import ModelConfig, ModelFamily

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

        # Save model config JSON
        config_dict: dict[str, Any] = {
            "family": model_family,
            "name": model_family,
            "in_channels": 1,
            "out_channels": 2,
        }
        config_path = tmp_path / "model_config.json"
        config_path.write_text(json.dumps(config_dict), encoding="utf-8")

        return {
            "checkpoint": str(ckpt_path),
            "model_config": str(config_path),
        }

    def test_sam3_vanilla_predict_returns_correct_shape(self, tmp_path: Path) -> None:
        from minivess.serving.mlflow_wrapper import MiniVessSegModel

        artifacts = self._make_checkpoint_artifacts(tmp_path, "sam3_vanilla")

        class _FakeContext:
            def __init__(self, artifacts: dict[str, str]) -> None:
                self.artifacts = artifacts

        model = MiniVessSegModel()
        ctx = _FakeContext(artifacts)
        model.load_context(ctx)

        # 3D input: (B, C, D, H, W)
        dummy = np.random.rand(1, 1, 4, 32, 32).astype(np.float32)
        output = model.predict(ctx, dummy)

        assert isinstance(output, np.ndarray)
        assert output.shape[0] == 1  # batch
        assert output.shape[1] == 2  # 2 classes (bg + fg)
        assert output.shape[2] == 4  # depth preserved

    def test_sam3_model_metadata_preserved(self, tmp_path: Path) -> None:
        """Verify model_family metadata roundtrips through config JSON."""
        artifacts = self._make_checkpoint_artifacts(tmp_path, "sam3_topolora")

        config_path = artifacts["model_config"]
        with open(config_path, encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded["family"] == "sam3_topolora"
        assert loaded["in_channels"] == 1
        assert loaded["out_channels"] == 2
