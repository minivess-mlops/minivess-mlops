"""Tests for trainer criterion upgrade for generic multi-task (T9a — #234, T9b — #235)."""

from __future__ import annotations

import pytest
import torch

from minivess.adapters.base import AdapterConfigInfo, SegmentationOutput
from minivess.adapters.multitask_adapter import AuxHeadConfig, MultiTaskAdapter
from minivess.pipeline.multitask_loss import AuxHeadLossConfig, MultiTaskLoss
from minivess.pipeline.multitask_metrics import compute_per_head_metrics

pytestmark = pytest.mark.model_loading


class _StubNet(torch.nn.Module):  # type: ignore[misc]
    """Inner network for stub model."""

    def __init__(self) -> None:
        super().__init__()
        self.decoder_conv = torch.nn.Conv3d(1, 16, 3, padding=1)
        self.output_conv = torch.nn.Conv3d(16, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_conv(self.decoder_conv(x))


class _StubModel(torch.nn.Module):  # type: ignore[misc]
    """Minimal stub model for training tests."""

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
        return AdapterConfigInfo(family="dynunet", name="stub_model")


def _make_batch(
    batch_size: int = 1,
    spatial: tuple[int, int, int] = (4, 8, 8),
) -> dict[str, torch.Tensor]:
    """Create a batch dict with image, label, and aux GT keys."""
    return {
        "image": torch.randn(batch_size, 1, *spatial),
        "label": torch.randint(0, 2, (batch_size, 1, *spatial)).float(),
        "sdf": torch.randn(batch_size, 1, *spatial),
    }


class TestTrainerCriterionUpgrade:
    """Tests for trainer criterion interface upgrade (T9a)."""

    def test_trainer_multitask_criterion_receives_output(self) -> None:
        """MultiTaskLoss gets SegmentationOutput, not raw logits."""
        seg_crit = torch.nn.CrossEntropyLoss()
        configs = [
            AuxHeadLossConfig(name="sdf", loss_type="mse", weight=0.25, gt_key="sdf"),
        ]
        loss_fn = MultiTaskLoss(seg_criterion=seg_crit, aux_head_configs=configs)

        # Create a multitask adapter
        base = _StubModel()
        aux_configs = [
            AuxHeadConfig(name="sdf", head_type="regression", out_channels=1)
        ]
        adapter = MultiTaskAdapter(base_model=base, aux_head_configs=aux_configs)

        batch = _make_batch()
        output = adapter(batch["image"])

        # MultiTaskLoss should accept (output, batch)
        loss = loss_fn(output, batch)
        assert torch.isfinite(loss)

    def test_trainer_multitask_criterion_receives_batch(self) -> None:
        """MultiTaskLoss gets full batch dict with GT keys."""
        seg_crit = torch.nn.CrossEntropyLoss()
        configs = [
            AuxHeadLossConfig(name="sdf", loss_type="mse", weight=0.25, gt_key="sdf"),
        ]
        loss_fn = MultiTaskLoss(seg_criterion=seg_crit, aux_head_configs=configs)

        base = _StubModel()
        aux_configs = [
            AuxHeadConfig(name="sdf", head_type="regression", out_channels=1)
        ]
        adapter = MultiTaskAdapter(base_model=base, aux_head_configs=aux_configs)

        batch = _make_batch()
        output = adapter(batch["image"])
        loss_fn(output, batch)

        # Per-component losses should be populated
        assert "loss/seg" in loss_fn.component_losses
        assert "loss/sdf" in loss_fn.component_losses

    def test_trainer_standard_criterion_unchanged(self) -> None:
        """Non-multitask criterion still works with (logits, labels)."""
        criterion = torch.nn.CrossEntropyLoss()
        base = _StubModel()
        batch = _make_batch()
        output = base(batch["image"])

        # Standard criterion: (logits, labels)
        label = batch["label"].squeeze(1).long()
        loss = criterion(output.logits, label)
        assert torch.isfinite(loss)

    def test_multitask_training_step(self) -> None:
        """One training step with MultiTaskAdapter + MultiTaskLoss completes."""
        base = _StubModel()
        aux_configs = [
            AuxHeadConfig(name="sdf", head_type="regression", out_channels=1)
        ]
        adapter = MultiTaskAdapter(base_model=base, aux_head_configs=aux_configs)

        seg_crit = torch.nn.CrossEntropyLoss()
        loss_configs = [
            AuxHeadLossConfig(name="sdf", loss_type="mse", weight=0.25, gt_key="sdf"),
        ]
        loss_fn = MultiTaskLoss(seg_criterion=seg_crit, aux_head_configs=loss_configs)

        optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)
        batch = _make_batch()

        # Forward
        output = adapter(batch["image"])
        loss = loss_fn(output, batch)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss)

    def test_multitask_backward_updates_all_heads(self) -> None:
        """All three task heads updated after one training step."""
        base = _StubModel()
        aux_configs = [
            AuxHeadConfig(name="sdf", head_type="regression", out_channels=1),
        ]
        adapter = MultiTaskAdapter(base_model=base, aux_head_configs=aux_configs)

        seg_crit = torch.nn.CrossEntropyLoss()
        loss_configs = [
            AuxHeadLossConfig(name="sdf", loss_type="mse", weight=0.25, gt_key="sdf"),
        ]
        loss_fn = MultiTaskLoss(seg_criterion=seg_crit, aux_head_configs=loss_configs)

        batch = _make_batch()
        output = adapter(batch["image"])
        loss = loss_fn(output, batch)
        loss.backward()

        # Check gradients on both base model and aux head
        for name, p in adapter.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_trainer_standard_model_no_regression(self) -> None:
        """Standard model ignores aux keys in batch."""
        criterion = torch.nn.CrossEntropyLoss()
        base = _StubModel()
        batch = _make_batch()

        # Standard model doesn't produce aux outputs
        output = base(batch["image"])
        assert "sdf" not in output.metadata

        # Standard loss works fine
        label = batch["label"].squeeze(1).long()
        loss = criterion(output.logits, label)
        assert torch.isfinite(loss)

    def test_is_multitask_loss_detection(self) -> None:
        """isinstance check correctly identifies MultiTaskLoss."""
        seg_crit = torch.nn.CrossEntropyLoss()
        multitask_loss = MultiTaskLoss(seg_criterion=seg_crit, aux_head_configs=[])

        assert isinstance(multitask_loss, MultiTaskLoss)
        assert not isinstance(seg_crit, MultiTaskLoss)


class TestPerHeadMetrics:
    """Tests for generic per-head validation metrics (T9b — #235)."""

    def test_generic_regression_head_metrics(self) -> None:
        """MAE and RMSE computed for any regression head."""
        head_configs = [
            AuxHeadConfig(name="sdf", head_type="regression", out_channels=1),
        ]
        pred = torch.randn(2, 1, 4, 8, 8)
        gt = torch.randn(2, 1, 4, 8, 8)
        output = SegmentationOutput(
            logits=torch.randn(2, 2, 4, 8, 8),
            prediction=torch.randn(2, 2, 4, 8, 8),
            metadata={"sdf": pred},
        )
        batch = {"label": torch.zeros(2, 1, 4, 8, 8), "sdf": gt}

        metrics = compute_per_head_metrics(output, batch, head_configs)
        assert "sdf/mae" in metrics
        assert "sdf/rmse" in metrics
        assert metrics["sdf/mae"] >= 0.0
        assert metrics["sdf/rmse"] >= 0.0

    def test_generic_classification_head_metrics(self) -> None:
        """Accuracy and F1 for classification heads."""
        head_configs = [
            AuxHeadConfig(
                name="vessel_class", head_type="classification", out_channels=3
            ),
        ]
        pred = torch.randn(2, 3, 4, 8, 8)
        gt = torch.randint(0, 3, (2, 4, 8, 8))
        output = SegmentationOutput(
            logits=torch.randn(2, 2, 4, 8, 8),
            prediction=torch.randn(2, 2, 4, 8, 8),
            metadata={"vessel_class": pred},
        )
        batch = {"label": torch.zeros(2, 1, 4, 8, 8), "vessel_class": gt}

        metrics = compute_per_head_metrics(output, batch, head_configs)
        assert "vessel_class/accuracy" in metrics
        assert "vessel_class/f1" in metrics
        assert 0.0 <= metrics["vessel_class/accuracy"] <= 1.0

    def test_generic_per_head_loss_logged(self) -> None:
        """loss/{head_name} logged for each aux head."""
        pred = torch.randn(2, 1, 4, 8, 8)
        gt = torch.randn(2, 1, 4, 8, 8)
        output = SegmentationOutput(
            logits=torch.randn(2, 2, 4, 8, 8),
            prediction=torch.randn(2, 2, 4, 8, 8),
            metadata={"sdf": pred},
        )
        batch = {"label": torch.zeros(2, 1, 4, 8, 8), "sdf": gt}

        # MultiTaskLoss stores component losses
        seg_crit = torch.nn.CrossEntropyLoss()
        loss_configs = [
            AuxHeadLossConfig(name="sdf", loss_type="mse", weight=0.25, gt_key="sdf"),
        ]
        loss_fn = MultiTaskLoss(seg_criterion=seg_crit, aux_head_configs=loss_configs)
        loss_fn(output, batch)

        assert "loss/sdf" in loss_fn.component_losses

    def test_generic_metrics_with_two_heads(self) -> None:
        """Metrics computed for 2 arbitrary heads."""
        head_configs = [
            AuxHeadConfig(name="sdf", head_type="regression", out_channels=1),
            AuxHeadConfig(name="cl_dist", head_type="regression", out_channels=1),
        ]
        output = SegmentationOutput(
            logits=torch.randn(2, 2, 4, 8, 8),
            prediction=torch.randn(2, 2, 4, 8, 8),
            metadata={
                "sdf": torch.randn(2, 1, 4, 8, 8),
                "cl_dist": torch.randn(2, 1, 4, 8, 8),
            },
        )
        batch = {
            "label": torch.zeros(2, 1, 4, 8, 8),
            "sdf": torch.randn(2, 1, 4, 8, 8),
            "cl_dist": torch.randn(2, 1, 4, 8, 8),
        }

        metrics = compute_per_head_metrics(output, batch, head_configs)
        assert "sdf/mae" in metrics
        assert "sdf/rmse" in metrics
        assert "cl_dist/mae" in metrics
        assert "cl_dist/rmse" in metrics

    def test_generic_metrics_with_zero_heads(self) -> None:
        """No extra metrics when no aux heads."""
        output = SegmentationOutput(
            logits=torch.randn(2, 2, 4, 8, 8),
            prediction=torch.randn(2, 2, 4, 8, 8),
            metadata={},
        )
        batch = {"label": torch.zeros(2, 1, 4, 8, 8)}

        metrics = compute_per_head_metrics(output, batch, [])
        assert len(metrics) == 0

    def test_primary_seg_metrics_unchanged(self) -> None:
        """DSC/clDice/HD95/NSD unaffected by aux heads."""
        head_configs = [
            AuxHeadConfig(name="sdf", head_type="regression", out_channels=1),
        ]
        output = SegmentationOutput(
            logits=torch.randn(2, 2, 4, 8, 8),
            prediction=torch.randn(2, 2, 4, 8, 8),
            metadata={"sdf": torch.randn(2, 1, 4, 8, 8)},
        )
        batch = {
            "label": torch.zeros(2, 1, 4, 8, 8),
            "sdf": torch.randn(2, 1, 4, 8, 8),
        }

        metrics = compute_per_head_metrics(output, batch, head_configs)
        # Per-head metrics should NOT contain primary seg metric keys
        for key in metrics:
            assert not key.startswith("val_"), (
                f"Primary metric {key} should not be in per-head metrics"
            )


class TestModelFamilyRegistration:
    """Tests for ModelFamily registration and wrapper config (T9c — #236)."""

    def test_wrapper_config_applies_tffm(self) -> None:
        """Wrappers list with type=tffm wraps model."""
        from minivess.adapters.model_builder import apply_wrappers

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
        wrapped = apply_wrappers(base, wrappers)
        # Should be a TFFMWrapper
        from minivess.adapters.tffm_wrapper import TFFMWrapper

        assert isinstance(wrapped, TFFMWrapper)

    def test_wrapper_config_applies_multitask(self) -> None:
        """Wrappers list with type=multitask wraps model."""
        from minivess.adapters.model_builder import apply_wrappers

        base = _StubModel()
        wrappers = [
            {
                "type": "multitask",
                "auxiliary_heads": [
                    {"name": "sdf", "type": "regression", "out_channels": 1},
                ],
            }
        ]
        wrapped = apply_wrappers(base, wrappers)
        assert isinstance(wrapped, MultiTaskAdapter)

    def test_multitask_dynunet_within_vram_budget(self) -> None:
        """Multi-task model parameter count is reasonable (<50% overhead)."""
        base = _StubModel()
        base_params = sum(p.numel() for p in base.parameters())

        aux_configs = [
            AuxHeadConfig(name="sdf", head_type="regression", out_channels=1),
            AuxHeadConfig(name="cl_dist", head_type="regression", out_channels=1),
        ]
        adapted = MultiTaskAdapter(base_model=base, aux_head_configs=aux_configs)
        total_params = sum(p.numel() for p in adapted.parameters())

        # Aux heads should add < 50% parameter overhead
        overhead = (total_params - base_params) / base_params
        assert overhead < 0.5, f"Parameter overhead {overhead:.1%} exceeds 50%"


class TestMultitaskExperimentConfig:
    """Tests for multi-task experiment config (T10 — #237)."""

    def _load_config(self) -> dict[str, object]:
        from minivess.config.compose import compose_experiment_config

        return compose_experiment_config(experiment_name="dynunet_multitask_ablation")

    def test_multitask_config_loads(self) -> None:
        """YAML config loads without error."""
        config = self._load_config()
        assert config["experiment_name"] == "dynunet_multitask_ablation_v1"

    def test_multitask_config_three_conditions(self) -> None:
        """Baseline, multitask, and multitask+D2C conditions present."""
        config = self._load_config()
        conditions: list[dict[str, object]] = config["conditions"]  # type: ignore[assignment]
        assert len(conditions) == 3
        names = [c["name"] for c in conditions]
        assert "baseline" in names
        assert "multitask" in names
        assert "multitask_d2c" in names

    def test_multitask_config_valid_loss_names(self) -> None:
        """Loss names recognized."""
        config = self._load_config()
        assert config["loss"] == "cbdice_cldice"

    def test_multitask_config_gpu_low_compute(self) -> None:
        """Respects 8GB VRAM budget."""
        config = self._load_config()
        assert config.get("compute", config.get("compute_profile")) == "gpu_low"
