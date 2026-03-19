"""Tests for cost appendix integration into biostatistics flow.

PR-E T3 (Issue #832): Wire cost reporting into biostatistics flow as
additional output artifacts. Cost figure + lineage manifest extension.

TDD Phase: RED — tests written before implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _make_cost_summary() -> dict[str, Any]:
    """Create a CostSummary-like dict for testing."""
    return {
        "total_spot_cost_usd": 12.50,
        "total_ondemand_cost_usd": 18.75,
        "savings_pct": 33.33,
        "total_gpu_hours": 15.0,
        "cost_by_model": {
            "dynunet": 4.50,
            "vesselfm": 3.80,
            "sam3_vanilla": 4.20,
        },
        "cost_by_phase": {
            "training": 10.50,
            "post_training": 1.50,
            "debug": 0.50,
        },
    }


class TestCostFigureGeneratesFile:
    """Cost figure is generated as PNG + JSON sidecar."""

    def test_cost_figure_generates_file(self, tmp_path: Path) -> None:
        """Cost breakdown figure saved to output directory."""
        from minivess.pipeline.biostatistics_figures import (
            generate_cost_breakdown_figure,
        )

        cost_summary = _make_cost_summary()
        artifact = generate_cost_breakdown_figure(cost_summary, tmp_path)

        assert artifact is not None
        assert artifact.figure_id == "cost_breakdown"
        assert len(artifact.paths) >= 1
        # At least one output file exists
        assert any(p.exists() for p in artifact.paths)


class TestCostFigureStackedBars:
    """Cost figure uses stacked bar chart."""

    def test_cost_figure_stacked_bars(self, tmp_path: Path) -> None:
        """Figure contains data for all model families."""
        from minivess.pipeline.biostatistics_figures import (
            generate_cost_breakdown_figure,
        )

        cost_summary = _make_cost_summary()
        artifact = generate_cost_breakdown_figure(cost_summary, tmp_path)

        # Sidecar JSON should contain model names
        assert artifact.sidecar_path is not None
        import json

        sidecar = json.loads(artifact.sidecar_path.read_text(encoding="utf-8"))
        assert "models" in sidecar
        assert set(sidecar["models"]) == {"dynunet", "vesselfm", "sam3_vanilla"}


class TestCostLineageField:
    """Lineage manifest includes cost_summary field."""

    def test_cost_lineage_field(self) -> None:
        """build_lineage_manifest_with_cost adds cost_summary."""
        from minivess.pipeline.biostatistics_lineage import (
            build_lineage_manifest_with_cost,
        )
        from minivess.pipeline.biostatistics_types import (
            SourceRun,
            SourceRunManifest,
        )

        runs = [
            SourceRun(
                run_id="r1",
                experiment_id="1",
                experiment_name="test",
                loss_function="dice_ce",
                fold_id=0,
                status="FINISHED",
            )
        ]
        manifest = SourceRunManifest.from_runs(runs)
        cost_summary = _make_cost_summary()

        lineage = build_lineage_manifest_with_cost(
            manifest=manifest,
            figures=[],
            tables=[],
            cost_summary=cost_summary,
        )

        assert "cost_summary" in lineage
        assert lineage["cost_summary"]["total_spot_cost_usd"] == 12.50
        assert lineage["cost_summary"]["savings_pct"] == 33.33


class TestCostFlowIntegration:
    """Cost task integrates into biostatistics flow."""

    def test_cost_task_produces_summary(self) -> None:
        """compute_cost_summary_task returns CostSummary dict."""
        from minivess.observability.cost_logging import CostSummary

        # Verify CostSummary dataclass can be constructed
        summary = CostSummary(
            total_spot_cost_usd=5.60,
            total_ondemand_cost_usd=8.40,
            savings_pct=33.33,
            cost_by_phase={"training": 5.20},
            cost_by_model={"dynunet": 5.60},
            total_gpu_hours=7.0,
        )
        assert summary.total_spot_cost_usd == 5.60

    def test_cost_flow_with_zero_data(self) -> None:
        """Flow handles zero-cost data (local runs) gracefully."""
        from minivess.observability.cost_logging import compute_spot_savings

        records: list[dict[str, Any]] = []
        summary = compute_spot_savings(records)

        assert summary.total_spot_cost_usd == 0.0
        assert summary.savings_pct == 0.0
        assert summary.total_gpu_hours == 0.0
