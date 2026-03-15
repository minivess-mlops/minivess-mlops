"""Tests for SAM3 hybrid experiment configs.

T1.1 + T1.4: Verify cloud smoke and full-dataset configs exist and are valid.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_CONFIGS_DIR = Path("configs/experiment")


class TestSam3HybridCloudConfig:
    """T1.1: Cloud smoke config with validation enabled."""

    def test_exists(self) -> None:
        assert (_CONFIGS_DIR / "smoke_sam3_hybrid_cloud.yaml").exists()

    def test_val_interval_is_1(self) -> None:
        config = yaml.safe_load(
            (_CONFIGS_DIR / "smoke_sam3_hybrid_cloud.yaml").read_text(encoding="utf-8")
        )
        assert config["val_interval"] == 1

    def test_max_epochs_at_least_3(self) -> None:
        config = yaml.safe_load(
            (_CONFIGS_DIR / "smoke_sam3_hybrid_cloud.yaml").read_text(encoding="utf-8")
        )
        assert config["max_epochs"] >= 3


class TestSam3HybridFullConfig:
    """T1.4: Full-dataset config for convergence testing."""

    def test_exists(self) -> None:
        assert (_CONFIGS_DIR / "sam3_hybrid_full.yaml").exists()

    def test_uses_full_splits(self) -> None:
        config = yaml.safe_load(
            (_CONFIGS_DIR / "sam3_hybrid_full.yaml").read_text(encoding="utf-8")
        )
        assert "3fold" in config.get("splits_file", "")

    def test_has_reasonable_epochs(self) -> None:
        config = yaml.safe_load(
            (_CONFIGS_DIR / "sam3_hybrid_full.yaml").read_text(encoding="utf-8")
        )
        assert config["max_epochs"] >= 30

    def test_uses_topology_loss(self) -> None:
        config = yaml.safe_load(
            (_CONFIGS_DIR / "sam3_hybrid_full.yaml").read_text(encoding="utf-8")
        )
        losses = config.get("losses", [])
        assert "cbdice_cldice" in losses
