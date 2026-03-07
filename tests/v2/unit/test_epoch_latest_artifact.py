"""Tests for T-17: epoch_latest.yaml writer in SegmentationTrainer.fit() loop.

Verifies that the trainer writes epoch_latest.yaml and epoch_latest.pth after
each epoch, using yaml.dump() (NOT f-strings or regex).

Uses yaml.safe_load() for parsing — NO regex (CLAUDE.md Rule #16).
"""

from __future__ import annotations

import ast
from pathlib import Path

import torch
import yaml

_TRAINER_SRC = Path("src/minivess/pipeline/trainer.py")


# ---------------------------------------------------------------------------
# AST-level: no regex used to read YAML, yaml.dump used to write
# ---------------------------------------------------------------------------


class TestEpochLatestNoRegex:
    def test_trainer_uses_yaml_dump_not_fstring(self) -> None:
        """trainer.py must use yaml.dump() to write epoch_latest.yaml."""
        source = _TRAINER_SRC.read_text(encoding="utf-8")
        assert "yaml.dump" in source, (
            "trainer.py must use yaml.dump() to write epoch_latest.yaml. "
            "Do NOT use f-strings or string concatenation for YAML serialization."
        )

    def test_trainer_imports_yaml(self) -> None:
        """trainer.py must import yaml."""
        source = _TRAINER_SRC.read_text(encoding="utf-8")
        tree = ast.parse(source)
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "yaml":
                        found = True
                        break
            if isinstance(node, ast.ImportFrom) and node.module == "yaml":
                found = True
                break
        assert found, "trainer.py must import yaml. Add: import yaml"

    def test_trainer_references_epoch_latest_yaml(self) -> None:
        """trainer.py must reference 'epoch_latest.yaml'."""
        source = _TRAINER_SRC.read_text(encoding="utf-8")
        assert "epoch_latest.yaml" in source, (
            "trainer.py must write epoch_latest.yaml after each epoch. "
            "Add epoch_latest_path = checkpoint_dir / 'epoch_latest.yaml' "
            "and write with yaml.dump()."
        )

    def test_trainer_references_epoch_latest_pth(self) -> None:
        """trainer.py must reference 'epoch_latest.pth'."""
        source = _TRAINER_SRC.read_text(encoding="utf-8")
        assert "epoch_latest.pth" in source, (
            "trainer.py must write epoch_latest.pth after each epoch. "
            "Add torch.save(model.state_dict(), checkpoint_dir / 'epoch_latest.pth')."
        )


# ---------------------------------------------------------------------------
# Helpers for building a minimal trainer for testing
# ---------------------------------------------------------------------------


def _build_mock_trainer(tmp_path):
    """Build a minimal SegmentationTrainer with a tiny model for testing."""
    import torch.nn as nn

    from minivess.config.models import CheckpointConfig, TrainingConfig
    from minivess.pipeline.trainer import SegmentationTrainer

    # Tiny model that returns a namespace with .logits (as expected by trainer)
    class _Output:
        def __init__(self, logits):
            self.logits = logits

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv3d(1, 1, kernel_size=1)

        def forward(self, x):
            return _Output(logits=self.conv(x))

    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    cfg = TrainingConfig(
        max_epochs=1,
        learning_rate=1e-3,
        batch_size=1,
        checkpoint=CheckpointConfig(save_last=False, save_history=False),
    )

    trainer = SegmentationTrainer(
        model=model,
        optimizer=optimizer,
        config=cfg,
    )
    return trainer


def _build_fake_loader(n_batches: int = 1):
    """Build a DataLoader yielding tiny dict batches {image, label}."""
    # Trainer expects batches as dicts with "image" and "label" keys
    batches = [
        {
            "image": torch.zeros(1, 1, 4, 4, 4),
            "label": torch.zeros(1, 1, 4, 4, 4),
        }
        for _ in range(n_batches)
    ]
    return batches  # plain list — trainer iterates it directly


# ---------------------------------------------------------------------------
# Functional: epoch_latest.yaml written after fit()
# ---------------------------------------------------------------------------


class TestEpochLatestYaml:
    def test_epoch_latest_yaml_written(self, tmp_path) -> None:
        """trainer.fit() must write epoch_latest.yaml in checkpoint_dir."""
        trainer = _build_mock_trainer(tmp_path)
        loader = _build_fake_loader()
        checkpoint_dir = tmp_path / "checkpoints"

        trainer.fit(
            train_loader=loader,
            val_loader=loader,
            fold_id=0,
            checkpoint_dir=checkpoint_dir,
        )

        yaml_path = checkpoint_dir / "epoch_latest.yaml"
        assert yaml_path.exists(), (
            f"epoch_latest.yaml not found at {yaml_path}. "
            "SegmentationTrainer.fit() must write epoch_latest.yaml after each epoch."
        )

    def test_epoch_latest_yaml_parseable(self, tmp_path) -> None:
        """epoch_latest.yaml must be parseable by yaml.safe_load() with required keys."""
        trainer = _build_mock_trainer(tmp_path)
        loader = _build_fake_loader()
        checkpoint_dir = tmp_path / "checkpoints"

        trainer.fit(
            train_loader=loader,
            val_loader=loader,
            fold_id=0,
            checkpoint_dir=checkpoint_dir,
        )

        yaml_path = checkpoint_dir / "epoch_latest.yaml"
        if not yaml_path.exists():
            return  # Covered by previous test

        content = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        assert isinstance(content, dict), (
            f"epoch_latest.yaml content must be a dict. Got {type(content)}"
        )
        for key in ("epoch", "fold", "best_val_loss", "timestamp"):
            assert key in content, (
                f"epoch_latest.yaml missing key '{key}'. "
                f"Got keys: {list(content.keys())}"
            )

    def test_epoch_latest_yaml_epoch_is_int(self, tmp_path) -> None:
        """epoch in epoch_latest.yaml must be an int, not a string."""
        trainer = _build_mock_trainer(tmp_path)
        loader = _build_fake_loader()
        checkpoint_dir = tmp_path / "checkpoints"

        trainer.fit(
            train_loader=loader,
            val_loader=loader,
            fold_id=0,
            checkpoint_dir=checkpoint_dir,
        )

        yaml_path = checkpoint_dir / "epoch_latest.yaml"
        if not yaml_path.exists():
            return

        content = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        assert isinstance(content.get("epoch"), int), (
            f"epoch_latest.yaml 'epoch' must be int, got {type(content.get('epoch'))}"
        )

    def test_epoch_latest_yaml_utc_timestamp(self, tmp_path) -> None:
        """timestamp in epoch_latest.yaml must be UTC (ends with +00:00 or Z)."""
        trainer = _build_mock_trainer(tmp_path)
        loader = _build_fake_loader()
        checkpoint_dir = tmp_path / "checkpoints"

        trainer.fit(
            train_loader=loader,
            val_loader=loader,
            fold_id=0,
            checkpoint_dir=checkpoint_dir,
        )

        yaml_path = checkpoint_dir / "epoch_latest.yaml"
        if not yaml_path.exists():
            return

        content = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        ts = str(content.get("timestamp", ""))
        assert "+00:00" in ts or ts.endswith("Z"), (
            f"epoch_latest.yaml timestamp '{ts}' is not UTC. "
            "Use datetime.now(timezone.utc).isoformat()."
        )


# ---------------------------------------------------------------------------
# Functional: epoch_latest.pth written after fit()
# ---------------------------------------------------------------------------


class TestEpochLatestPth:
    def test_epoch_latest_pth_written(self, tmp_path) -> None:
        """trainer.fit() must write epoch_latest.pth in checkpoint_dir."""
        trainer = _build_mock_trainer(tmp_path)
        loader = _build_fake_loader()
        checkpoint_dir = tmp_path / "checkpoints"

        trainer.fit(
            train_loader=loader,
            val_loader=loader,
            fold_id=0,
            checkpoint_dir=checkpoint_dir,
        )

        pth_path = checkpoint_dir / "epoch_latest.pth"
        assert pth_path.exists(), (
            f"epoch_latest.pth not found at {pth_path}. "
            "SegmentationTrainer.fit() must save epoch_latest.pth after each epoch."
        )

    def test_epoch_latest_pth_loadable(self, tmp_path) -> None:
        """epoch_latest.pth must be a loadable state_dict (dict)."""
        trainer = _build_mock_trainer(tmp_path)
        loader = _build_fake_loader()
        checkpoint_dir = tmp_path / "checkpoints"

        trainer.fit(
            train_loader=loader,
            val_loader=loader,
            fold_id=0,
            checkpoint_dir=checkpoint_dir,
        )

        pth_path = checkpoint_dir / "epoch_latest.pth"
        if not pth_path.exists():
            return

        state = torch.load(str(pth_path), map_location="cpu", weights_only=True)
        assert isinstance(state, dict), (
            f"epoch_latest.pth content must be a dict (state_dict). Got {type(state)}"
        )
