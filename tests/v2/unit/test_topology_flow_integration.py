"""Tests for Flow 3/5 topology metrics integration (T20 — #247)."""

from __future__ import annotations


def _make_condition_results() -> dict[str, list[dict[str, float]]]:
    return {
        "baseline": [
            {"dice": 0.80, "cldice": 0.75},
            {"dice": 0.82, "cldice": 0.77},
        ],
        "multitask": [
            {"dice": 0.83, "cldice": 0.79},
            {"dice": 0.84, "cldice": 0.80},
        ],
    }


class TestAnalysisFlowTopology:
    """Tests for Flow 3 topology integration."""

    def test_analysis_flow_extracts_per_head_metrics(self) -> None:
        """SDF/CL metrics extracted from MLflow run data."""
        from minivess.orchestration.topology_helpers import extract_per_head_metrics

        run_data = {
            "metrics": {
                "val_loss": 0.5,
                "sdf/mae": 0.12,
                "sdf/rmse": 0.15,
                "cl_dist/mae": 0.08,
                "loss/seg": 0.4,
            }
        }
        per_head = extract_per_head_metrics(run_data)
        assert "sdf/mae" in per_head
        assert "sdf/rmse" in per_head
        assert "cl_dist/mae" in per_head
        # Should NOT include val_ or loss/ prefixed metrics
        assert "val_loss" not in per_head
        assert "loss/seg" not in per_head

    def test_analysis_flow_topology_comparison(self) -> None:
        """Comparison table includes topology metrics."""
        from minivess.orchestration.topology_helpers import build_topology_comparison

        results = _make_condition_results()
        comparison = build_topology_comparison(results, metric_names=["dice", "cldice"])
        assert "baseline" in comparison
        assert "multitask" in comparison
        assert "dice" in comparison["baseline"]
        assert "mean" in comparison["baseline"]["dice"]


class TestDashboardFlowTopology:
    """Tests for Flow 5 topology integration."""

    def test_dashboard_flow_multitask_curves(self) -> None:
        """Training curve data for 3 loss components."""
        from minivess.orchestration.topology_helpers import (
            extract_multitask_training_curves,
        )

        history = [
            {"loss/seg": 0.5, "loss/sdf": 0.3, "loss/cl": 0.2},
            {"loss/seg": 0.4, "loss/sdf": 0.25, "loss/cl": 0.18},
            {"loss/seg": 0.35, "loss/sdf": 0.2, "loss/cl": 0.15},
        ]
        curves = extract_multitask_training_curves(history)
        assert "loss/seg" in curves
        assert "loss/sdf" in curves
        assert "loss/cl" in curves
        assert len(curves["loss/seg"]) == 3

    def test_dashboard_flow_graceful_no_multitask(self) -> None:
        """No crash when no multitask runs exist."""
        from minivess.orchestration.topology_helpers import (
            build_topology_comparison_data,
        )

        result = build_topology_comparison_data({})
        assert result == {}
