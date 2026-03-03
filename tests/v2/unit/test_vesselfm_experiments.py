"""Tests for VesselFM experiment configs (#291).

All experiments use Hydra delta-only format in configs/experiment/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


class TestVesselFMHydraExperiments:
    """Hydra delta-only experiment configs in configs/experiment/."""

    def test_vesselfm_zeroshot_exists(self) -> None:
        assert (CONFIGS_DIR / "experiment" / "vesselfm_zeroshot.yaml").exists()

    def test_vesselfm_finetune_exists(self) -> None:
        assert (CONFIGS_DIR / "experiment" / "vesselfm_finetune.yaml").exists()

    def test_zeroshot_config_loads(self) -> None:
        cfg = _load_yaml(CONFIGS_DIR / "experiment" / "vesselfm_zeroshot.yaml")
        assert cfg.get("experiment_name") == "vesselfm_zeroshot_eval"

    def test_zeroshot_max_epochs_zero(self) -> None:
        cfg = _load_yaml(CONFIGS_DIR / "experiment" / "vesselfm_zeroshot.yaml")
        assert cfg.get("max_epochs") == 0

    def test_finetune_has_loss(self) -> None:
        cfg = _load_yaml(CONFIGS_DIR / "experiment" / "vesselfm_finetune.yaml")
        assert "losses" in cfg

    def test_finetune_has_leakage_tag(self) -> None:
        cfg = _load_yaml(CONFIGS_DIR / "experiment" / "vesselfm_finetune.yaml")
        tags = cfg.get("tags", {})
        assert "data_leakage" in tags

    def test_zeroshot_has_leakage_tag(self) -> None:
        cfg = _load_yaml(CONFIGS_DIR / "experiment" / "vesselfm_zeroshot.yaml")
        tags = cfg.get("tags", {})
        assert "data_leakage" in tags

    def test_zeroshot_delta_only_compact(self) -> None:
        """Zero-shot experiment should be compact (delta-only)."""
        text = (CONFIGS_DIR / "experiment" / "vesselfm_zeroshot.yaml").read_text(
            encoding="utf-8"
        )
        lines = [
            line
            for line in text.strip().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        assert len(lines) < 25, f"Expected compact delta-only, got {len(lines)} lines"

    def test_vesselfm_zeroshot_hydra_compose(self) -> None:
        """Hydra compose should produce valid merged config."""
        from minivess.config.compose import compose_experiment_config

        result = compose_experiment_config(experiment_name="vesselfm_zeroshot")
        assert result.get("experiment_name") == "vesselfm_zeroshot_eval"
        assert result.get("model") == "vesselfm"
        assert result.get("max_epochs") == 0
