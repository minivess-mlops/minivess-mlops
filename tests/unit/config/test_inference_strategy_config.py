"""Tests for InferenceStrategyConfig and EvaluationConfig.inference_strategies."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from minivess.config.evaluation_config import EvaluationConfig, InferenceStrategyConfig


def _make_strategy(**kwargs: object) -> dict[str, object]:
    base = {
        "name": "standard_patch",
        "roi_size": [128, 128, 16],
        "is_primary": True,
    }
    base.update(kwargs)
    return base


class TestInferenceStrategyConfig:
    def test_strategy_config_constructed_from_dict(self) -> None:
        cfg = InferenceStrategyConfig(**_make_strategy())  # type: ignore[arg-type]
        assert cfg.name == "standard_patch"
        assert cfg.roi_size == [128, 128, 16]
        assert cfg.is_primary is True

    def test_roi_size_per_model_string_accepted(self) -> None:
        cfg = InferenceStrategyConfig(**_make_strategy(roi_size="per_model"))  # type: ignore[arg-type]
        assert cfg.roi_size == "per_model"

    def test_roi_size_list_of_ints_accepted(self) -> None:
        cfg = InferenceStrategyConfig(**_make_strategy(roi_size=[128, 128, 16]))  # type: ignore[arg-type]
        assert cfg.roi_size == [128, 128, 16]

    def test_roi_size_wildcard_depth_accepted(self) -> None:
        cfg = InferenceStrategyConfig(**_make_strategy(roi_size=[512, 512, -1]))  # type: ignore[arg-type]
        assert cfg.roi_size == [512, 512, -1]

    def test_overlap_must_be_less_than_one(self) -> None:
        with pytest.raises(ValidationError, match="overlap"):
            InferenceStrategyConfig(**_make_strategy(overlap=1.0))  # type: ignore[arg-type]

    def test_sw_batch_size_must_be_positive(self) -> None:
        with pytest.raises(ValidationError, match="sw_batch_size"):
            InferenceStrategyConfig(**_make_strategy(sw_batch_size=0))  # type: ignore[arg-type]


class TestEvaluationConfigStrategies:
    def test_evaluation_config_requires_exactly_one_primary_raises_on_two(self) -> None:
        with pytest.raises(ValidationError):
            EvaluationConfig(
                inference_strategies=[
                    InferenceStrategyConfig(
                        name="a", roi_size=[128, 128, 16], is_primary=True
                    ),
                    InferenceStrategyConfig(
                        name="b", roi_size=[64, 64, 8], is_primary=True
                    ),
                ]
            )

    def test_evaluation_config_no_primary_raises(self) -> None:
        with pytest.raises(ValidationError):
            EvaluationConfig(
                inference_strategies=[
                    InferenceStrategyConfig(
                        name="a", roi_size=[128, 128, 16], is_primary=False
                    ),
                ]
            )

    def test_evaluation_config_unique_names_enforced(self) -> None:
        with pytest.raises(ValidationError):
            EvaluationConfig(
                inference_strategies=[
                    InferenceStrategyConfig(
                        name="same", roi_size=[128, 128, 16], is_primary=True
                    ),
                    InferenceStrategyConfig(
                        name="same", roi_size=[64, 64, 8], is_primary=False
                    ),
                ]
            )

    def test_evaluation_config_empty_strategies_allowed(self) -> None:
        cfg = EvaluationConfig(inference_strategies=[])
        assert cfg.inference_strategies == []

    def test_evaluation_config_default_has_one_primary(self) -> None:
        cfg = EvaluationConfig()
        primaries = [s for s in cfg.inference_strategies if s.is_primary]
        assert len(primaries) == 1
