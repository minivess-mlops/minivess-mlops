"""Integration tests for TFFMWrapper + DynUNet (T12 — #239)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from minivess.adapters.base import AdapterConfigInfo, SegmentationOutput
from minivess.adapters.tffm_wrapper import TFFMWrapper


class _StubNet(torch.nn.Module):  # type: ignore[misc]
    """Inner network for stub DynUNet-like model."""

    def __init__(self) -> None:
        super().__init__()
        # Mimic DynUNet structure: bottleneck → decoder → output
        self.bottleneck = torch.nn.Conv3d(1, 32, 3, padding=1)
        self.decoder_conv = torch.nn.Conv3d(32, 16, 3, padding=1)
        self.output_conv = torch.nn.Conv3d(16, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.bottleneck(x)
        feat = self.decoder_conv(feat)
        return self.output_conv(feat)


class _StubDynUNet(torch.nn.Module):  # type: ignore[misc]
    """Minimal DynUNet-like model for TFFM integration tests."""

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

    def get_config(self) -> AdapterConfigInfo:
        return AdapterConfigInfo(family="dynunet", name="stub_dynunet")

    def load_checkpoint(self, path: Path) -> None:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state_dict, strict=False)

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def export_onnx(self, path: Path, example_input: torch.Tensor) -> None:
        pass

    @property
    def config(self) -> AdapterConfigInfo:
        return AdapterConfigInfo(family="dynunet", name="stub_dynunet")


def _make_tffm_wrapper() -> TFFMWrapper:
    """Create TFFMWrapper wrapping a stub DynUNet."""
    base = _StubDynUNet()
    return TFFMWrapper(
        base_model=base,
        grid_size=4,
        hidden_dim=16,
        n_heads=2,
        k_neighbors=4,
    )


class TestTFFMIntegration:
    """Integration tests for TFFMWrapper wrapping DynUNet."""

    def test_tffm_wraps_dynunet_forward(self) -> None:
        """Forward pass completes without error."""
        wrapper = _make_tffm_wrapper()
        x = torch.randn(1, 1, 4, 8, 8)
        output = wrapper(x)
        assert isinstance(output, SegmentationOutput)

    def test_tffm_dynunet_output_shape(self) -> None:
        """Output shape matches standard DynUNet."""
        wrapper = _make_tffm_wrapper()
        x = torch.randn(1, 1, 4, 8, 8)
        output = wrapper(x)
        assert output.logits.shape == (1, 2, 4, 8, 8)

    def test_tffm_gradient_to_base_model(self) -> None:
        """DynUNet params receive gradients through TFFM."""
        wrapper = _make_tffm_wrapper()
        x = torch.randn(1, 1, 4, 8, 8)
        output = wrapper(x)
        loss = output.logits.sum()
        loss.backward()
        # Check base model params have gradients
        for name, p in wrapper._base_model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for base model param {name}"

    def test_tffm_gradient_to_tffm_block(self) -> None:
        """TFFM block params receive gradients."""
        wrapper = _make_tffm_wrapper()
        x = torch.randn(1, 1, 4, 8, 8)
        output = wrapper(x)
        loss = output.logits.sum()
        loss.backward()
        # Check TFFM block params have gradients
        tffm_has_grad = False
        for _name, p in wrapper.tffm_block.named_parameters():
            if p.requires_grad and p.grad is not None:
                tffm_has_grad = True
                break
        assert tffm_has_grad, "TFFM block should receive gradients"

    def test_tffm_metadata_applied(self) -> None:
        """Output metadata indicates TFFM was applied."""
        wrapper = _make_tffm_wrapper()
        x = torch.randn(1, 1, 4, 8, 8)
        output = wrapper(x)
        assert output.metadata.get("tffm_applied") is True

    def test_tffm_checkpoint_roundtrip(self) -> None:
        """Save/load preserves both DynUNet and TFFM components."""
        wrapper = _make_tffm_wrapper()
        x = torch.randn(1, 1, 4, 8, 8)
        out1 = wrapper(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "tffm_ckpt.pt"
            wrapper.save_checkpoint(ckpt)

            wrapper2 = _make_tffm_wrapper()
            wrapper2.load_checkpoint(ckpt)
            out2 = wrapper2(x)

        torch.testing.assert_close(out1.logits, out2.logits)

    def test_tffm_training_step(self) -> None:
        """One training step with loss backward completes."""
        wrapper = _make_tffm_wrapper()
        optimizer = torch.optim.Adam(wrapper.parameters(), lr=1e-3)

        x = torch.randn(1, 1, 4, 8, 8)
        labels = torch.randint(0, 2, (1, 4, 8, 8))

        output = wrapper(x)
        loss = torch.nn.functional.cross_entropy(output.logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss)

    def test_tffm_config_info(self) -> None:
        """get_config includes TFFM params."""
        wrapper = _make_tffm_wrapper()
        config = wrapper.get_config()
        assert isinstance(config, AdapterConfigInfo)
        assert config.extras.get("tffm_grid_size") == 4
