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


# ---------------------------------------------------------------------------
# Graph-topology experiment sweep config tests (Issue #125)
# ---------------------------------------------------------------------------

CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"


class TestGraphTopologyExperimentConfig:
    """Tests for dynunet_graph_topology.yaml config."""

    def test_graph_topology_experiment_config_loadable(self) -> None:
        config_path = CONFIGS_DIR / "experiments" / "dynunet_graph_topology.yaml"
        assert config_path.exists(), f"Config not found: {config_path}"
        with config_path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f)
        assert isinstance(config, dict)
        assert "experiment_name" in config
        assert "model" in config
        assert "losses" in config

    def test_graph_topology_config_has_five_losses(self) -> None:
        config_path = CONFIGS_DIR / "experiments" / "dynunet_graph_topology.yaml"
        with config_path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f)
        losses = config["losses"]
        assert len(losses) == 5
        expected = {
            "dice_ce",
            "cbdice_cldice",
            "graph_topology",
            "skeleton_recall",
            "betti_matching",
        }
        assert set(losses) == expected

    def test_graph_topology_config_has_graph_metrics(self) -> None:
        config_path = CONFIGS_DIR / "experiments" / "dynunet_graph_topology.yaml"
        with config_path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f)
        topo_metrics = config["topology_metrics"]["metrics"]
        assert "apls" in topo_metrics
        assert "skeleton_recall_metric" in topo_metrics
        assert "bdr" in topo_metrics
        assert "ccdice" in topo_metrics
        assert "junction_f1" in topo_metrics

    def test_graph_topology_config_has_champion_selection(self) -> None:
        config_path = CONFIGS_DIR / "experiments" / "dynunet_graph_topology.yaml"
        with config_path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f)
        champion = config["champion_selection"]
        assert champion["strategy"] == "rank_then_aggregate"
        categories = champion["categories"]
        category_names = [c["name"] for c in categories]
        assert "champion_topology" in category_names
        assert "champion_overlap" in category_names
        assert "champion_balanced" in category_names

    def test_graph_topology_config_primary_metric(self) -> None:
        config_path = CONFIGS_DIR / "experiments" / "dynunet_graph_topology.yaml"
        with config_path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f)
        primary = config["checkpoint"]["primary_metric"]
        assert primary == "val_compound_nsd_cldice"

    def test_all_config_losses_are_buildable(self) -> None:
        """Every loss in the config must be registered in the factory."""
        from minivess.pipeline.loss_functions import build_loss_function

        config_path = CONFIGS_DIR / "experiments" / "dynunet_graph_topology.yaml"
        with config_path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f)
        for loss_name in config["losses"]:
            loss_fn = build_loss_function(loss_name)
            assert loss_fn is not None, f"Loss '{loss_name}' not registered"


class TestGraphMetricRegistryCompleteness:
    """Tests that all graph metrics are registered in metric_registry.yaml."""

    @staticmethod
    def _load_registry() -> list[dict[str, Any]]:
        registry_path = CONFIGS_DIR / "metric_registry.yaml"
        with registry_path.open(encoding="utf-8") as f:
            result: list[dict[str, Any]] = yaml.safe_load(f)["metrics"]
            return result

    def _metric_names(self) -> set[str]:
        return {m["name"] for m in self._load_registry()}

    def test_apls_registered_in_metric_registry(self) -> None:
        assert "apls" in self._metric_names()

    def test_skeleton_recall_metric_registered(self) -> None:
        assert "skeleton_recall_metric" in self._metric_names()

    def test_bdr_registered_in_metric_registry(self) -> None:
        assert "bdr" in self._metric_names()

    def test_all_topology_config_metrics_registered(self) -> None:
        """Every metric in the topology config must exist in the registry."""
        config_path = CONFIGS_DIR / "experiments" / "dynunet_graph_topology.yaml"
        with config_path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f)
        topo_metrics = config["topology_metrics"]["metrics"]
        registry_names = self._metric_names()
        for metric_name in topo_metrics:
            assert metric_name in registry_names, (
                f"Metric '{metric_name}' not in metric_registry.yaml"
            )

    def test_graph_metrics_have_required_fields(self) -> None:
        """Graph metrics should have all standard fields."""
        metrics = self._load_registry()
        graph_metric_names = {"apls", "skeleton_recall_metric", "bdr"}
        for metric in metrics:
            if metric["name"] in graph_metric_names:
                assert "display_name" in metric
                assert "mlflow_name" in metric
                assert "direction" in metric
                assert metric["direction"] == "maximize"
                assert "bounds" in metric
                assert metric["bounds"] == [0.0, 1.0]
