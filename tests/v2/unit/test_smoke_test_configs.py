"""Tests for GPU smoke test experiment configs (#634, T1.2).

Validates that smoke_*.yaml configs are correct for RunPod GPU testing.
Uses yaml.safe_load() — NO regex (CLAUDE.md Rule #16).

Run: uv run pytest tests/v2/unit/test_smoke_test_configs.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parent.parent.parent.parent
EXPERIMENT_DIR = ROOT / "configs" / "experiment"
SPLITS_DIR = ROOT / "configs" / "splits"

SMOKE_CONFIGS = [
    "smoke_sam3_vanilla.yaml",
    "smoke_sam3_hybrid.yaml",
    "smoke_vesselfm.yaml",
]


def _load_yaml(name: str) -> dict:
    return yaml.safe_load((EXPERIMENT_DIR / name).read_text(encoding="utf-8"))


class TestSmokeTestConfigsExist:
    """All smoke test configs must exist and be valid YAML."""

    @pytest.mark.parametrize("config_name", SMOKE_CONFIGS)
    def test_config_exists(self, config_name: str) -> None:
        path = EXPERIMENT_DIR / config_name
        assert path.exists(), f"Missing smoke config: {path}"

    @pytest.mark.parametrize("config_name", SMOKE_CONFIGS)
    def test_config_is_valid_yaml(self, config_name: str) -> None:
        config = _load_yaml(config_name)
        assert isinstance(config, dict)


class TestSmokeTestConfigValues:
    """Smoke configs must have correct training params."""

    @pytest.mark.parametrize("config_name", SMOKE_CONFIGS)
    def test_max_epochs_is_2(self, config_name: str) -> None:
        config = _load_yaml(config_name)
        assert config.get("max_epochs") == 2, (
            f"{config_name}: max_epochs must be 2 for smoke test"
        )

    @pytest.mark.parametrize("config_name", SMOKE_CONFIGS)
    def test_num_folds_is_1(self, config_name: str) -> None:
        config = _load_yaml(config_name)
        assert config.get("num_folds") == 1, (
            f"{config_name}: num_folds must be 1 for smoke test"
        )

    @pytest.mark.parametrize("config_name", SMOKE_CONFIGS)
    def test_max_train_volumes_is_2(self, config_name: str) -> None:
        config = _load_yaml(config_name)
        assert config.get("max_train_volumes") == 2

    @pytest.mark.parametrize("config_name", SMOKE_CONFIGS)
    def test_max_val_volumes_is_2(self, config_name: str) -> None:
        config = _load_yaml(config_name)
        assert config.get("max_val_volumes") == 2

    @pytest.mark.parametrize("config_name", SMOKE_CONFIGS)
    def test_splits_file_is_smoke_test(self, config_name: str) -> None:
        config = _load_yaml(config_name)
        splits = config.get("splits_file", "")
        assert "smoke_test_1fold_4vol" in splits, (
            f"{config_name}: splits_file must reference smoke_test_1fold_4vol.json"
        )

    @pytest.mark.parametrize("config_name", SMOKE_CONFIGS)
    def test_dvc_commit_field_exists(self, config_name: str) -> None:
        config = _load_yaml(config_name)
        data = config.get("data", {})
        assert "dvc_commit" in data, (
            f"{config_name}: must have data.dvc_commit for provenance tracking"
        )


class TestSmokeTestSplitFile:
    """Validate smoke_test_1fold_4vol.json structure."""

    def test_split_file_exists(self) -> None:
        path = SPLITS_DIR / "smoke_test_1fold_4vol.json"
        assert path.exists()

    def test_split_file_is_valid_json(self) -> None:
        path = SPLITS_DIR / "smoke_test_1fold_4vol.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(data, list)

    def test_split_has_exactly_1_fold(self) -> None:
        path = SPLITS_DIR / "smoke_test_1fold_4vol.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data) == 1, f"Expected 1 fold, got {len(data)}"

    def test_split_has_2_train_2_val(self) -> None:
        path = SPLITS_DIR / "smoke_test_1fold_4vol.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        fold = data[0]
        assert len(fold["train"]) == 2, f"Expected 2 train, got {len(fold['train'])}"
        assert len(fold["val"]) == 2, f"Expected 2 val, got {len(fold['val'])}"

    def test_split_has_4_total_volumes(self) -> None:
        path = SPLITS_DIR / "smoke_test_1fold_4vol.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        fold = data[0]
        total = len(fold["train"]) + len(fold["val"])
        assert total == 4, f"Expected 4 volumes total, got {total}"
