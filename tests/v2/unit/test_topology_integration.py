"""Tests for topology metric integration and config wiring.

Covers Issue #118: wire topology metrics into metric registry,
create experiment config, add compound NSD+clDice metric.
TDD RED phase: tests written before implementation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import yaml

# ---------------------------------------------------------------------------
# Config loading tests
# ---------------------------------------------------------------------------


class TestTopologyExperimentConfig:
    """Tests for dynunet_topology.yaml experiment config."""

    def test_topology_experiment_config_loadable(self) -> None:
        config_path = (
            Path(__file__).parents[3]
            / "configs"
            / "experiments"
            / "dynunet_topology.yaml"
        )
        assert config_path.exists(), f"Config not found: {config_path}"
        with config_path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f)
        assert "experiment_name" in config
        assert "losses" in config

    def test_topology_config_has_topology_losses(self) -> None:
        config_path = (
            Path(__file__).parents[3]
            / "configs"
            / "experiments"
            / "dynunet_topology.yaml"
        )
        with config_path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f)
        losses = config["losses"]
        # Should include topology-aware losses
        assert (
            "skeleton_recall" in losses or "cape" in losses or "cbdice_cldice" in losses
        )


class TestTopologyMetricRegistry:
    """Tests for topology metrics registered in metric_registry.yaml."""

    @staticmethod
    def _load_registry() -> list[dict[str, Any]]:
        registry_path = Path(__file__).parents[3] / "configs" / "metric_registry.yaml"
        with registry_path.open(encoding="utf-8") as f:
            result: list[dict[str, Any]] = yaml.safe_load(f)["metrics"]
            return result

    def _metric_names(self) -> set[str]:
        return {m["name"] for m in self._load_registry()}

    def test_nsd_registered_in_metric_registry(self) -> None:
        assert "nsd" in self._metric_names()

    def test_hd95_registered_in_metric_registry(self) -> None:
        assert "hd95" in self._metric_names()

    def test_ccdice_registered_in_metric_registry(self) -> None:
        assert "ccdice" in self._metric_names()

    def test_betti_error_registered_in_metric_registry(self) -> None:
        assert "betti_error_beta0" in self._metric_names()

    def test_compound_nsd_cldice_registered(self) -> None:
        assert "val_compound_nsd_cldice" in self._metric_names()


# ---------------------------------------------------------------------------
# Compound NSD+clDice metric tests
# ---------------------------------------------------------------------------


class TestCompoundNsdCldice:
    """Tests for compute_compound_nsd_cldice in validation_metrics.py."""

    def test_compound_nsd_cldice_bounded_zero_one(self) -> None:
        from minivess.pipeline.validation_metrics import compute_compound_nsd_cldice

        result = compute_compound_nsd_cldice(nsd=0.8, cldice=0.7)
        assert 0.0 <= result <= 1.0

    def test_compound_nsd_cldice_perfect_scores_returns_one(self) -> None:
        from minivess.pipeline.validation_metrics import compute_compound_nsd_cldice

        result = compute_compound_nsd_cldice(nsd=1.0, cldice=1.0)
        assert result == pytest.approx(1.0, abs=1e-4)

    def test_compound_nsd_cldice_nan_safe(self) -> None:
        from minivess.pipeline.validation_metrics import compute_compound_nsd_cldice

        result = compute_compound_nsd_cldice(nsd=float("nan"), cldice=0.8)
        assert not math.isnan(result)
        result2 = compute_compound_nsd_cldice(nsd=0.8, cldice=float("nan"))
        assert not math.isnan(result2)

    def test_compound_nsd_cldice_zero_inputs(self) -> None:
        from minivess.pipeline.validation_metrics import compute_compound_nsd_cldice

        result = compute_compound_nsd_cldice(nsd=0.0, cldice=0.0)
        assert result == pytest.approx(0.0, abs=1e-4)
