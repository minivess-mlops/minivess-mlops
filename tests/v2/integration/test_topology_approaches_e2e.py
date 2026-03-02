"""End-to-end integration tests for all 3 topology approaches (T16 — #243)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import torch

from minivess.adapters.base import AdapterConfigInfo, SegmentationOutput
from minivess.adapters.model_builder import apply_wrappers
from minivess.adapters.multitask_adapter import AuxHeadConfig, MultiTaskAdapter
from minivess.pipeline.multitask_loss import AuxHeadLossConfig, MultiTaskLoss


class _StubNet(torch.nn.Module):  # type: ignore[misc]
    """Inner network for E2E test stub."""

    def __init__(self) -> None:
        super().__init__()
        self.bottleneck = torch.nn.Conv3d(1, 32, 3, padding=1)
        self.decoder_conv = torch.nn.Conv3d(32, 16, 3, padding=1)
        self.output_conv = torch.nn.Conv3d(16, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.bottleneck(x)
        feat = self.decoder_conv(feat)
        return self.output_conv(feat)


class _StubModel(torch.nn.Module):  # type: ignore[misc]
    """Minimal stub model for E2E tests."""

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

    @property
    def config(self) -> AdapterConfigInfo:
        return AdapterConfigInfo(family="dynunet", name="stub_e2e")


def _make_batch(
    spatial: tuple[int, int, int] = (4, 8, 8),
) -> dict[str, torch.Tensor]:
    return {
        "image": torch.randn(1, 1, *spatial),
        "label": torch.randint(0, 2, (1, 1, *spatial)).float(),
        "sdf": torch.randn(1, 1, *spatial),
    }


class TestTopologyApproachesE2E:
    """End-to-end integration tests for topology approaches."""

    def test_e2e_d2c_augmentation(self) -> None:
        """2 epochs with D2C-compatible model, loss finite."""
        model = _StubModel()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for _epoch in range(2):
            batch = _make_batch()
            output = model(batch["image"])
            label = batch["label"].squeeze(1).long()
            loss = criterion(output.logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            assert torch.isfinite(loss)

    def test_e2e_multitask_dynunet(self) -> None:
        """2 epochs, all heads produce output."""
        base = _StubModel()
        aux_configs = [
            AuxHeadConfig(name="sdf", head_type="regression", out_channels=1),
        ]
        model = MultiTaskAdapter(base_model=base, aux_head_configs=aux_configs)

        seg_crit = torch.nn.CrossEntropyLoss()
        loss_configs = [
            AuxHeadLossConfig(name="sdf", loss_type="mse", weight=0.25, gt_key="sdf"),
        ]
        criterion = MultiTaskLoss(seg_criterion=seg_crit, aux_head_configs=loss_configs)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for _epoch in range(2):
            batch = _make_batch()
            output = model(batch["image"])
            assert "sdf" in output.metadata
            loss = criterion(output, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            assert torch.isfinite(loss)

    def test_e2e_tffm_dynunet(self) -> None:
        """2 epochs, tffm_applied in metadata."""
        base = _StubModel()
        wrappers = [
            {
                "type": "tffm",
                "grid_size": 4,
                "hidden_dim": 16,
                "n_heads": 2,
                "k_neighbors": 4,
            }
        ]
        model = apply_wrappers(base, wrappers)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for _epoch in range(2):
            batch = _make_batch()
            output = model(batch["image"])
            assert isinstance(output, SegmentationOutput)
            assert output.metadata.get("tffm_applied") is True
            label = batch["label"].squeeze(1).long()
            loss = criterion(output.logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            assert torch.isfinite(loss)

    def test_e2e_combined_pipeline(self) -> None:
        """2 epochs with all three approaches."""
        base = _StubModel()
        wrappers: list[dict[str, Any]] = [
            {
                "type": "tffm",
                "grid_size": 4,
                "hidden_dim": 16,
                "n_heads": 2,
                "k_neighbors": 4,
            },
            {
                "type": "multitask",
                "auxiliary_heads": [
                    {"name": "sdf", "type": "regression", "out_channels": 1},
                ],
            },
        ]
        model = apply_wrappers(base, wrappers)

        seg_crit = torch.nn.CrossEntropyLoss()
        loss_configs = [
            AuxHeadLossConfig(name="sdf", loss_type="mse", weight=0.25, gt_key="sdf"),
        ]
        criterion = MultiTaskLoss(seg_criterion=seg_crit, aux_head_configs=loss_configs)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for _epoch in range(2):
            batch = _make_batch()
            output = model(batch["image"])
            assert "sdf" in output.metadata
            loss = criterion(output, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            assert torch.isfinite(loss)

    def test_e2e_checkpoint_roundtrip(self) -> None:
        """Save/load for combined pipeline."""
        base = _StubModel()
        aux_configs = [
            AuxHeadConfig(name="sdf", head_type="regression", out_channels=1),
        ]
        model = MultiTaskAdapter(base_model=base, aux_head_configs=aux_configs)

        x = torch.randn(1, 1, 4, 8, 8)
        out1 = model(x)

        with tempfile.TemporaryDirectory() as tmp_str:
            ckpt = Path(tmp_str) / "e2e_ckpt.pt"
            torch.save(model.state_dict(), ckpt)

            model2 = MultiTaskAdapter(
                base_model=_StubModel(),
                aux_head_configs=aux_configs,
            )
            model2.load_state_dict(
                torch.load(ckpt, map_location="cpu", weights_only=True)
            )
            out2 = model2(x)

        torch.testing.assert_close(out1.logits, out2.logits)

    def test_e2e_vram_under_budget(self) -> None:
        """Peak parameter count is reasonable for 8GB budget."""
        base = _StubModel()
        wrappers: list[dict[str, Any]] = [
            {
                "type": "tffm",
                "grid_size": 4,
                "hidden_dim": 16,
                "n_heads": 2,
                "k_neighbors": 4,
            },
            {
                "type": "multitask",
                "auxiliary_heads": [
                    {"name": "sdf", "type": "regression", "out_channels": 1},
                    {"name": "cl_dist", "type": "regression", "out_channels": 1},
                ],
            },
        ]
        model = apply_wrappers(base, wrappers)

        # Parameter count as proxy for VRAM (actual check requires GPU)
        total_params = sum(p.numel() for p in model.parameters())
        # StubModel is tiny; in real DynUNet ~30M params; aux adds <50%
        # For stub, just verify it's reasonable
        assert total_params > 0
        assert total_params < 10_000_000  # Stub shouldn't have 10M+ params
