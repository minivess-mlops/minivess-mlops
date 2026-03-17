"""Tests for shared Prometheus Gauges module (issue #747).

Verifies that prometheus_metrics.py exports correct Gauge objects
and provides update functions for cost and estimated cost dicts.
"""

from __future__ import annotations

import prometheus_client


class TestModuleExportsGauges:
    """Importing the module provides at least 6 Gauge objects."""

    def test_module_exports_gauges(self) -> None:
        from minivess.observability.prometheus_metrics import GAUGES

        assert len(GAUGES) >= 6

    def test_gauges_have_correct_names(self) -> None:
        from minivess.observability.prometheus_metrics import GAUGES

        for name in GAUGES:
            assert name.startswith("minivess_training_"), (
                f"Gauge name {name} does not start with minivess_training_"
            )

    def test_gauges_are_gauge_instances(self) -> None:
        from minivess.observability.prometheus_metrics import GAUGES

        for name, gauge in GAUGES.items():
            assert isinstance(gauge, prometheus_client.Gauge), (
                f"{name} is not a Gauge instance"
            )


class TestUpdateCostGauges:
    """update_cost_gauges sets Gauge values from compute_cost_analysis output."""

    def test_update_cost_gauges_from_dict(self) -> None:
        from minivess.observability.prometheus_metrics import (
            GAUGES,
            update_cost_gauges,
        )

        cost_dict = {
            "cost_total_usd": 1.23,
            "cost_effective_gpu_rate": 0.45,
            "cost_gpu_utilization_fraction": 0.89,
            "cost_setup_fraction": 0.11,
            "cost_total_wall_seconds": 3600.0,
            "cost_setup_usd": 0.13,
            "cost_training_usd": 1.10,
            "cost_epochs_to_amortize_setup": 5,
            "cost_break_even_epochs": 3,
        }
        update_cost_gauges(cost_dict)

        # Verify the key gauges were set
        gauge_total = GAUGES["minivess_training_cost_total_usd"]
        assert gauge_total._value.get() == 1.23

    def test_update_cost_gauges_missing_keys(self) -> None:
        from minivess.observability.prometheus_metrics import (
            GAUGES,
            update_cost_gauges,
        )

        # Set a known value first
        GAUGES["minivess_training_cost_total_usd"].set(99.0)

        # Pass incomplete dict — should not raise, gauge retains previous value
        update_cost_gauges({"cost_setup_fraction": 0.5})

        # The gauge for cost_total_usd should retain its previous value
        assert GAUGES["minivess_training_cost_total_usd"]._value.get() == 99.0


class TestUpdateEstimatedCostGauges:
    """update_estimated_cost_gauges sets Gauges from estimate dict."""

    def test_update_estimated_cost_gauges(self) -> None:
        from minivess.observability.prometheus_metrics import (
            GAUGES,
            update_estimated_cost_gauges,
        )

        estimate_dict = {
            "estimated_total_cost": 2.50,
            "estimated_total_hours": 1.5,
            "cost_per_epoch": 0.05,
            "epoch_seconds": 120.0,
        }
        update_estimated_cost_gauges(estimate_dict)

        gauge_est = GAUGES["minivess_training_estimated_total_cost_usd"]
        assert gauge_est._value.get() == 2.50


class TestGenerateLatest:
    """prometheus_client.generate_latest() contains our registered metrics."""

    def test_generate_latest_contains_metrics(self) -> None:
        from minivess.observability.prometheus_metrics import GAUGES, update_cost_gauges

        update_cost_gauges({"cost_total_usd": 1.0})
        output = prometheus_client.generate_latest().decode("utf-8")

        for name in GAUGES:
            assert name in output, f"Metric {name} not found in generate_latest output"
