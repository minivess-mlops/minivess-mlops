"""Tests for SAM3 adapter save/load/export methods (T4).

Verifies that all 3 SAM3 adapters have working checkpoint save/load
and export_onnx methods that don't crash on self.net.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import torch

from minivess.config.models import ModelConfig, ModelFamily

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def vanilla_adapter() -> Any:
    from minivess.adapters.sam3_vanilla import Sam3VanillaAdapter

    config = ModelConfig(
        family=ModelFamily.SAM3_VANILLA,
        name="test_vanilla",
        in_channels=1,
        out_channels=2,
    )
    return Sam3VanillaAdapter(config, use_stub=True)


@pytest.fixture
def topolora_adapter() -> Any:
    from minivess.adapters.sam3_topolora import Sam3TopoLoraAdapter

    config = ModelConfig(
        family=ModelFamily.SAM3_TOPOLORA,
        name="test_topolora",
        in_channels=1,
        out_channels=2,
        lora_rank=2,
    )
    return Sam3TopoLoraAdapter(config, use_stub=True)


@pytest.fixture
def hybrid_adapter() -> Any:
    from minivess.adapters.sam3_hybrid import Sam3HybridAdapter

    config = ModelConfig(
        family=ModelFamily.SAM3_HYBRID,
        name="test_hybrid",
        in_channels=1,
        out_channels=2,
        architecture_params={"filters": [16, 32, 64]},
    )
    return Sam3HybridAdapter(config, use_stub=True)


# ---------------------------------------------------------------------------
# Test save_checkpoint
# ---------------------------------------------------------------------------
class TestSaveCheckpoint:
    """save_checkpoint must not crash (no self.net dependency)."""

    def test_vanilla_save(self, vanilla_adapter: Any, tmp_path: Path) -> None:
        path = tmp_path / "vanilla.pt"
        vanilla_adapter.save_checkpoint(path)
        assert path.exists()

    def test_topolora_save(self, topolora_adapter: Any, tmp_path: Path) -> None:
        path = tmp_path / "topolora.pt"
        topolora_adapter.save_checkpoint(path)
        assert path.exists()

    def test_hybrid_save(self, hybrid_adapter: Any, tmp_path: Path) -> None:
        path = tmp_path / "hybrid.pt"
        hybrid_adapter.save_checkpoint(path)
        assert path.exists()


# ---------------------------------------------------------------------------
# Test load_checkpoint roundtrip
# ---------------------------------------------------------------------------
class TestLoadCheckpoint:
    """load_checkpoint after save_checkpoint produces identical forward output."""

    def test_vanilla_roundtrip(self, vanilla_adapter: Any, tmp_path: Path) -> None:
        path = tmp_path / "vanilla.pt"
        x = torch.randn(1, 1, 8, 32, 32)
        vanilla_adapter.eval()
        with torch.no_grad():
            out_before = vanilla_adapter(x).logits.clone()
        vanilla_adapter.save_checkpoint(path)

        # Create fresh adapter and load checkpoint
        from minivess.adapters.sam3_vanilla import Sam3VanillaAdapter

        adapter2 = Sam3VanillaAdapter(vanilla_adapter.config, use_stub=True)
        adapter2.load_checkpoint(path)
        adapter2.eval()
        with torch.no_grad():
            out_after = adapter2(x).logits
        assert torch.allclose(out_before, out_after, atol=1e-5)

    def test_topolora_roundtrip(self, topolora_adapter: Any, tmp_path: Path) -> None:
        path = tmp_path / "topolora.pt"
        x = torch.randn(1, 1, 8, 32, 32)
        topolora_adapter.eval()
        with torch.no_grad():
            out_before = topolora_adapter(x).logits.clone()
        topolora_adapter.save_checkpoint(path)

        from minivess.adapters.sam3_topolora import Sam3TopoLoraAdapter

        adapter2 = Sam3TopoLoraAdapter(topolora_adapter.config, use_stub=True)
        adapter2.load_checkpoint(path)
        adapter2.eval()
        with torch.no_grad():
            out_after = adapter2(x).logits
        assert torch.allclose(out_before, out_after, atol=1e-5)

    def test_hybrid_roundtrip(self, hybrid_adapter: Any, tmp_path: Path) -> None:
        path = tmp_path / "hybrid.pt"
        x = torch.randn(1, 1, 8, 32, 32)
        hybrid_adapter.eval()
        with torch.no_grad():
            out_before = hybrid_adapter(x).logits.clone()
        hybrid_adapter.save_checkpoint(path)

        from minivess.adapters.sam3_hybrid import Sam3HybridAdapter

        adapter2 = Sam3HybridAdapter(hybrid_adapter.config, use_stub=True)
        adapter2.load_checkpoint(path)
        adapter2.eval()
        with torch.no_grad():
            out_after = adapter2(x).logits
        assert torch.allclose(out_before, out_after, atol=1e-5)


# ---------------------------------------------------------------------------
# Test trainable_parameters
# ---------------------------------------------------------------------------
class TestTrainableParameters:
    """trainable_parameters must return a positive int for all variants."""

    def test_vanilla_params(self, vanilla_adapter: Any) -> None:
        count = vanilla_adapter.trainable_parameters()
        assert isinstance(count, int)
        assert count > 0

    def test_topolora_params(self, topolora_adapter: Any) -> None:
        count = topolora_adapter.trainable_parameters()
        assert isinstance(count, int)
        assert count > 0

    def test_hybrid_params(self, hybrid_adapter: Any) -> None:
        count = hybrid_adapter.trainable_parameters()
        assert isinstance(count, int)
        assert count > 0


# ---------------------------------------------------------------------------
# Test export_onnx
# ---------------------------------------------------------------------------
class TestExportOnnx:
    """export_onnx must produce a valid file (not crash on self.net)."""

    def test_vanilla_export(self, vanilla_adapter: Any, tmp_path: Path) -> None:
        path = tmp_path / "vanilla.onnx"
        example = torch.randn(1, 1, 8, 32, 32)
        vanilla_adapter.export_onnx(path, example)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_topolora_export(self, topolora_adapter: Any, tmp_path: Path) -> None:
        path = tmp_path / "topolora.onnx"
        example = torch.randn(1, 1, 8, 32, 32)
        topolora_adapter.export_onnx(path, example)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_hybrid_export(self, hybrid_adapter: Any, tmp_path: Path) -> None:
        path = tmp_path / "hybrid.onnx"
        example = torch.randn(1, 1, 8, 32, 32)
        hybrid_adapter.export_onnx(path, example)
        assert path.exists()
        assert path.stat().st_size > 0
