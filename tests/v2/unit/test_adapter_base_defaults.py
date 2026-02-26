"""Tests for ModelAdapter base class default implementations (R5.1 + R5.2).

Validates that ModelAdapter provides concrete default load/save/trainable/export
methods and that adapters using self.net inherit them without duplication.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from minivess.adapters.base import AdapterConfigInfo, ModelAdapter, SegmentationOutput

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Minimal concrete adapter for testing base defaults
# ---------------------------------------------------------------------------


class _SimpleAdapter(ModelAdapter):
    """Minimal adapter with self.net for testing base defaults."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(4, 2)

    def forward(self, images: Tensor, **kwargs: Any) -> SegmentationOutput:
        logits = self.net(images)
        return SegmentationOutput(
            prediction=torch.softmax(logits, dim=-1),
            logits=logits,
        )

    def get_config(self) -> AdapterConfigInfo:
        return AdapterConfigInfo(family="test", name="simple")


# ---------------------------------------------------------------------------
# T1: Default checkpoint methods from base class
# ---------------------------------------------------------------------------


class TestBaseCheckpointDefaults:
    """Test that ModelAdapter provides working default checkpoint methods."""

    def test_save_creates_file(self, tmp_path: Path) -> None:
        """Default save_checkpoint should create a .pth file."""
        adapter = _SimpleAdapter()
        ckpt = tmp_path / "model.pth"
        adapter.save_checkpoint(ckpt)
        assert ckpt.exists()

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Default save_checkpoint should create parent directories."""
        adapter = _SimpleAdapter()
        ckpt = tmp_path / "nested" / "dir" / "model.pth"
        adapter.save_checkpoint(ckpt)
        assert ckpt.exists()

    def test_load_restores_weights(self, tmp_path: Path) -> None:
        """Default load_checkpoint should restore weights exactly."""
        adapter1 = _SimpleAdapter()
        ckpt = tmp_path / "model.pth"
        adapter1.save_checkpoint(ckpt)

        adapter2 = _SimpleAdapter()
        adapter2.load_checkpoint(ckpt)

        # Weights should match
        for p1, p2 in zip(
            adapter1.net.parameters(), adapter2.net.parameters(), strict=True
        ):
            assert torch.allclose(p1, p2)

    def test_trainable_parameters_positive(self) -> None:
        """Default trainable_parameters should return > 0 for a trained net."""
        adapter = _SimpleAdapter()
        assert adapter.trainable_parameters() > 0

    def test_trainable_parameters_counts_correctly(self) -> None:
        """Default trainable_parameters should count only requires_grad=True."""
        adapter = _SimpleAdapter()
        # Freeze all params
        for p in adapter.net.parameters():
            p.requires_grad = False
        assert adapter.trainable_parameters() == 0


# ---------------------------------------------------------------------------
# T2: Default ONNX export from base class
# ---------------------------------------------------------------------------


class TestBaseOnnxExportDefault:
    """Test that ModelAdapter provides a working default export_onnx."""

    def test_export_creates_file(self, tmp_path: Path) -> None:
        """Default export_onnx should create an ONNX file."""
        adapter = _SimpleAdapter()
        onnx_path = tmp_path / "model.onnx"
        example = torch.randn(1, 4)
        adapter.export_onnx(onnx_path, example)
        assert onnx_path.exists()

    def test_export_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Default export_onnx should create parent directories."""
        adapter = _SimpleAdapter()
        onnx_path = tmp_path / "nested" / "model.onnx"
        example = torch.randn(1, 4)
        adapter.export_onnx(onnx_path, example)
        assert onnx_path.exists()


# ---------------------------------------------------------------------------
# T3: Adapters using self.net DO NOT override checkpoint methods
# ---------------------------------------------------------------------------


class TestAdaptersInheritDefaults:
    """Verify standard adapters removed their checkpoint overrides."""

    def test_segresnet_uses_base_save(self) -> None:
        """SegResNetAdapter should NOT define its own save_checkpoint."""
        from minivess.adapters.segresnet import SegResNetAdapter

        assert "save_checkpoint" not in SegResNetAdapter.__dict__

    def test_segresnet_uses_base_load(self) -> None:
        """SegResNetAdapter should NOT define its own load_checkpoint."""
        from minivess.adapters.segresnet import SegResNetAdapter

        assert "load_checkpoint" not in SegResNetAdapter.__dict__

    def test_segresnet_uses_base_trainable(self) -> None:
        """SegResNetAdapter should NOT define its own trainable_parameters."""
        from minivess.adapters.segresnet import SegResNetAdapter

        assert "trainable_parameters" not in SegResNetAdapter.__dict__

    def test_swinunetr_uses_base_methods(self) -> None:
        """SwinUNETRAdapter should NOT define checkpoint/trainable methods."""
        from minivess.adapters.swinunetr import SwinUNETRAdapter

        for method in ("save_checkpoint", "load_checkpoint", "trainable_parameters"):
            assert method not in SwinUNETRAdapter.__dict__, (
                f"{method} should be inherited"
            )

    def test_dynunet_uses_base_methods(self) -> None:
        """DynUNetAdapter should NOT define checkpoint/trainable methods."""
        from minivess.adapters.dynunet import DynUNetAdapter

        for method in ("save_checkpoint", "load_checkpoint", "trainable_parameters"):
            assert method not in DynUNetAdapter.__dict__, (
                f"{method} should be inherited"
            )

    def test_vista3d_uses_base_methods(self) -> None:
        """Vista3dAdapter should NOT define checkpoint/trainable methods."""
        from minivess.adapters.vista3d import Vista3dAdapter

        for method in ("save_checkpoint", "load_checkpoint", "trainable_parameters"):
            assert method not in Vista3dAdapter.__dict__, (
                f"{method} should be inherited"
            )

    def test_vesselfm_uses_base_methods(self) -> None:
        """VesselFMAdapter should NOT define checkpoint/trainable methods."""
        from minivess.adapters.vesselfm import VesselFMAdapter

        for method in ("save_checkpoint", "load_checkpoint", "trainable_parameters"):
            assert method not in VesselFMAdapter.__dict__, (
                f"{method} should be inherited"
            )


# ---------------------------------------------------------------------------
# T4: Adapters with special logic still override
# ---------------------------------------------------------------------------


class TestSpecialAdaptersOverride:
    """Adapters with non-standard checkpoint logic should still override."""

    def test_lora_overrides_checkpoint(self) -> None:
        """LoraModelAdapter has LoRA-specific checkpoint logic."""
        from minivess.adapters.lora import LoraModelAdapter

        assert "save_checkpoint" in LoraModelAdapter.__dict__
        assert "load_checkpoint" in LoraModelAdapter.__dict__

    def test_comma_overrides_checkpoint(self) -> None:
        """CommaAdapter saves self (not self.net) — must keep override."""
        from minivess.adapters.comma import CommaAdapter

        assert "save_checkpoint" in CommaAdapter.__dict__
        assert "load_checkpoint" in CommaAdapter.__dict__
