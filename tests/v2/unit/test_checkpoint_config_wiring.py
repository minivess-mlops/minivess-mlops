"""Tests for Phase 3: YAML checkpoint config wiring through experiment scripts.

These tests verify that:
1. dynunet_losses.yaml contains a valid checkpoint section
2. CheckpointConfig can be built from a YAML-parsed dict
3. train_monitored.py uses the primary metric checkpoint name (not best_model.pth)
4. run_experiment.py delegates checkpoint config from YAML to training
"""

from __future__ import annotations

from pathlib import Path

import yaml


def test_experiment_yaml_has_checkpoint_section() -> None:
    """dynunet_losses.yaml includes checkpoint config."""
    yaml_path = (
        Path(__file__).resolve().parents[3]
        / "configs"
        / "experiments"
        / "dynunet_losses.yaml"
    )
    with open(yaml_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert "checkpoint" in config
    assert "tracked_metrics" in config["checkpoint"]
    assert (
        len(config["checkpoint"]["tracked_metrics"]) >= 2
    )  # at least val_loss + one more


def test_experiment_yaml_checkpoint_primary_metric() -> None:
    """Primary metric in YAML is val_loss."""
    yaml_path = (
        Path(__file__).resolve().parents[3]
        / "configs"
        / "experiments"
        / "dynunet_losses.yaml"
    )
    with open(yaml_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert config["checkpoint"]["primary_metric"] == "val_loss"


def test_experiment_yaml_checkpoint_min_epochs() -> None:
    """min_epochs is set to prevent premature stopping."""
    yaml_path = (
        Path(__file__).resolve().parents[3]
        / "configs"
        / "experiments"
        / "dynunet_losses.yaml"
    )
    with open(yaml_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert config["checkpoint"]["min_epochs"] >= 10


def test_experiment_yaml_checkpoint_has_early_stopping_strategy() -> None:
    """Checkpoint section includes early_stopping_strategy."""
    yaml_path = (
        Path(__file__).resolve().parents[3]
        / "configs"
        / "experiments"
        / "dynunet_losses.yaml"
    )
    with open(yaml_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert "early_stopping_strategy" in config["checkpoint"]
    assert config["checkpoint"]["early_stopping_strategy"] in {"all", "any", "primary"}


def test_checkpoint_config_from_yaml_dict() -> None:
    """CheckpointConfig can be built from a YAML-parsed dict."""
    from minivess.config.models import CheckpointConfig, TrackedMetricConfig

    raw = {
        "tracked_metrics": [
            {"name": "val_loss", "direction": "minimize", "patience": 15},
            {"name": "val_dice", "direction": "maximize", "patience": 20},
        ],
        "early_stopping_strategy": "all",
        "primary_metric": "val_loss",
        "min_delta": 1e-4,
        "min_epochs": 10,
        "save_last": True,
        "save_history": True,
    }
    tracked = [TrackedMetricConfig(**m) for m in raw["tracked_metrics"]]
    cfg = CheckpointConfig(
        tracked_metrics=tracked,
        early_stopping_strategy=raw["early_stopping_strategy"],
        primary_metric=raw["primary_metric"],
        min_delta=raw["min_delta"],
        min_epochs=raw["min_epochs"],
        save_last=raw["save_last"],
        save_history=raw["save_history"],
    )
    assert len(cfg.tracked_metrics) == 2
    assert cfg.primary_metric == "val_loss"
    assert cfg.min_epochs == 10


def test_checkpoint_config_tracked_metrics_have_correct_directions() -> None:
    """TrackedMetricConfig correctly stores direction values."""
    from minivess.config.models import TrackedMetricConfig

    minimize_metric = TrackedMetricConfig(
        name="val_loss", direction="minimize", patience=15
    )
    maximize_metric = TrackedMetricConfig(
        name="val_dice", direction="maximize", patience=20
    )
    assert minimize_metric.direction == "minimize"
    assert maximize_metric.direction == "maximize"
    assert minimize_metric.patience == 15
    assert maximize_metric.patience == 20


def test_train_monitored_uses_primary_metric_checkpoint() -> None:
    """run_fold_safe looks for best_<primary_metric>.pth, not best_model.pth."""
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "train_monitored.py"
    source = script_path.read_text(encoding="utf-8")
    # Should NOT contain "best_model.pth" (the old stale reference)
    assert "best_model.pth" not in source, (
        "train_monitored.py still references old best_model.pth"
    )
    # Should contain the new pattern (best_ prefix for dynamic metric name)
    assert "best_" in source  # e.g., best_val_loss.pth or best_{primary}


def test_train_py_uses_primary_metric_checkpoint() -> None:
    """train.py run_fold looks for best_<primary_metric>.pth, not best_model.pth."""
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "train.py"
    source = script_path.read_text(encoding="utf-8")
    # Should NOT contain "best_model.pth"
    assert "best_model.pth" not in source, (
        "train.py still references old best_model.pth"
    )


def test_train_monitored_imports_checkpoint_config() -> None:
    """train_monitored.py imports CheckpointConfig and TrackedMetricConfig."""
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "train_monitored.py"
    source = script_path.read_text(encoding="utf-8")
    assert "CheckpointConfig" in source
    assert "TrackedMetricConfig" in source


def test_train_monitored_has_checkpoint_config_parsing() -> None:
    """train_monitored.py parses checkpoint_config from args."""
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "train_monitored.py"
    source = script_path.read_text(encoding="utf-8")
    # Should reference checkpoint_config somewhere in build_configs or similar
    assert "checkpoint_config" in source


def test_run_experiment_passes_checkpoint_config() -> None:
    """run_experiment.py delegates checkpoint config from YAML to train."""
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "run_experiment.py"
    source = script_path.read_text(encoding="utf-8")
    # Should reference checkpoint config somewhere
    assert "checkpoint" in source.lower()
    # Should set checkpoint_config on parsed namespace
    assert "checkpoint_config" in source


def test_run_experiment_delegates_to_train_monitored() -> None:
    """run_experiment.py delegates training to train_monitored.py (not train.py)."""
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "run_experiment.py"
    source = script_path.read_text(encoding="utf-8")
    # Should reference train_monitored script
    assert "train_monitored" in source


def test_load_checkpoint_handles_new_format(tmp_path: Path) -> None:
    """ModelAdapter.load_checkpoint handles the new dict-format checkpoint."""
    import torch

    from minivess.adapters.base import ModelAdapter, SegmentationOutput
    from minivess.config.models import ModelConfig, ModelFamily

    # Create a concrete minimal adapter for testing
    class _MinimalAdapter(ModelAdapter):
        def __init__(self) -> None:
            super().__init__()
            import torch.nn as nn

            self.net = nn.Linear(4, 2)
            # Fake config for _build_config
            self.config = ModelConfig(
                family=ModelFamily.CUSTOM,
                name="test",
                in_channels=1,
                out_channels=2,
            )

        def forward(self, images: torch.Tensor, **kwargs: object) -> SegmentationOutput:
            out = self.net(images)
            return self._build_output(out, "test")

        def get_config(self) -> object:
            return self._build_config()

    model = _MinimalAdapter()
    original_weight = model.net.weight.data.clone()

    # Save a new-format checkpoint (as produced by save_metric_checkpoint)
    ckpt_path = tmp_path / "best_val_loss.pth"
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
        "checkpoint_metadata": {"epoch": 5, "metric_value": 0.1},
    }
    torch.save(payload, ckpt_path)

    # Mutate the model weights
    with torch.no_grad():
        model.net.weight.fill_(99.0)

    # load_checkpoint should restore the original weights
    model.load_checkpoint(ckpt_path)
    assert torch.allclose(model.net.weight.data, original_weight), (
        "load_checkpoint did not restore model weights from new-format checkpoint"
    )


def test_load_checkpoint_handles_legacy_format(tmp_path: Path) -> None:
    """ModelAdapter.load_checkpoint handles the old bare-state-dict format."""
    import torch

    from minivess.adapters.base import ModelAdapter, SegmentationOutput
    from minivess.config.models import ModelConfig, ModelFamily

    class _MinimalAdapter(ModelAdapter):
        def __init__(self) -> None:
            super().__init__()
            import torch.nn as nn

            self.net = nn.Linear(4, 2)
            self.config = ModelConfig(
                family=ModelFamily.CUSTOM,
                name="test",
                in_channels=1,
                out_channels=2,
            )

        def forward(self, images: torch.Tensor, **kwargs: object) -> SegmentationOutput:
            out = self.net(images)
            return self._build_output(out, "test")

        def get_config(self) -> object:
            return self._build_config()

    model = _MinimalAdapter()
    original_weight = model.net.weight.data.clone()

    # Save a legacy-format checkpoint (bare state dict)
    ckpt_path = tmp_path / "best_model.pth"
    torch.save(model.net.state_dict(), ckpt_path)

    # Mutate the model weights
    with torch.no_grad():
        model.net.weight.fill_(99.0)

    # load_checkpoint should still work with legacy format
    model.load_checkpoint(ckpt_path)
    assert torch.allclose(model.net.weight.data, original_weight), (
        "load_checkpoint did not restore model weights from legacy-format checkpoint"
    )
