"""Tests that configs/evaluation/default.yaml loads correctly with InferenceStrategyConfig."""

from __future__ import annotations

from pathlib import Path

import yaml

from minivess.config.evaluation_config import EvaluationConfig

_DEFAULT_YAML = Path(__file__).parents[3] / "configs" / "evaluation" / "default.yaml"


def _load_config() -> EvaluationConfig:
    raw = yaml.safe_load(_DEFAULT_YAML.read_text(encoding="utf-8"))
    return EvaluationConfig(**raw)


class TestEvaluationYamlLoading:
    def test_default_yaml_loads_without_error(self) -> None:
        cfg = _load_config()
        assert isinstance(cfg, EvaluationConfig)

    def test_default_yaml_has_standard_patch_strategy(self) -> None:
        cfg = _load_config()
        names = [s.name for s in cfg.inference_strategies]
        assert "standard_patch" in names
        assert len(cfg.inference_strategies) >= 1

    def test_default_yaml_standard_patch_is_primary(self) -> None:
        cfg = _load_config()
        standard = next(
            s for s in cfg.inference_strategies if s.name == "standard_patch"
        )
        assert standard.is_primary is True

    def test_default_yaml_roi_size_is_list_of_ints(self) -> None:
        cfg = _load_config()
        standard = next(
            s for s in cfg.inference_strategies if s.name == "standard_patch"
        )
        assert standard.roi_size == [128, 128, 16]

    def test_default_yaml_has_fast_strategy(self) -> None:
        cfg = _load_config()
        names = [s.name for s in cfg.inference_strategies]
        assert "fast" in names
        fast = next(s for s in cfg.inference_strategies if s.name == "fast")
        assert fast.roi_size == "per_model"

    def test_default_yaml_fast_is_not_primary(self) -> None:
        cfg = _load_config()
        fast = next(s for s in cfg.inference_strategies if s.name == "fast")
        assert fast.is_primary is False
