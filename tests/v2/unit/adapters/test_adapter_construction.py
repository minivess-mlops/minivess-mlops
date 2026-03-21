"""Tests for adapter registry — all factorial models must be registered.

Catches issues like missing model registrations before cloud runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from minivess.adapters.model_builder import _MODEL_REGISTRY
from minivess.config.models import ModelFamily


@pytest.mark.model_construction
class TestAdapterRegistry:
    """All model families in the factorial config must be registered."""

    def test_all_factorial_models_registered(self) -> None:
        """Every model in debug_factorial.yaml must have a registry entry."""
        config_path = Path("configs/experiment/debug_factorial.yaml")
        if not config_path.exists():
            pytest.skip("debug_factorial.yaml not found")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        for model_name in config["factors"]["model_family"]:
            family = ModelFamily(model_name)
            assert family in _MODEL_REGISTRY, (
                f"Model '{model_name}' (ModelFamily.{family.name}) "
                f"not in _MODEL_REGISTRY. Available: "
                f"{sorted(f.value for f in _MODEL_REGISTRY)}"
            )

    def test_zero_shot_models_registered(self) -> None:
        """Zero-shot baselines must also be registered."""
        config_path = Path("configs/experiment/debug_factorial.yaml")
        if not config_path.exists():
            pytest.skip("debug_factorial.yaml not found")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        for baseline in config.get("zero_shot_baselines", []):
            model_name = baseline["model"]
            family = ModelFamily(model_name)
            assert family in _MODEL_REGISTRY, (
                f"Zero-shot model '{model_name}' not in _MODEL_REGISTRY"
            )

    def test_all_model_family_enum_values_registered(self) -> None:
        """Every ModelFamily enum value (except CUSTOM) should be registered."""
        for family in ModelFamily:
            if family == ModelFamily.CUSTOM:
                continue
            assert family in _MODEL_REGISTRY, (
                f"ModelFamily.{family.name} ({family.value}) not registered"
            )
