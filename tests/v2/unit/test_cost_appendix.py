"""Tests for cost appendix LaTeX table generation.

Validates _generate_cost_appendix_table() which produces a LaTeX table
with GPU cost breakdown, spot savings, and total cost for the manuscript appendix.

Addresses TRIPOD-LLM Item 12 (cost transparency).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _make_mock_cost_data() -> list[dict[str, Any]]:
    """Create mock MLflow cost data for testing.

    Each dict represents one training run's cost information.
    """
    return [
        {
            "model": "dynunet",
            "loss": "dice_ce",
            "fold": 0,
            "gpu_hours": 2.5,
            "cost_usd": 1.25,
            "spot_cost_usd": 0.75,
            "gpu_type": "L4",
        },
        {
            "model": "dynunet",
            "loss": "dice_ce",
            "fold": 1,
            "gpu_hours": 2.4,
            "cost_usd": 1.20,
            "spot_cost_usd": 0.72,
            "gpu_type": "L4",
        },
        {
            "model": "dynunet",
            "loss": "cbdice_cldice",
            "fold": 0,
            "gpu_hours": 2.8,
            "cost_usd": 1.40,
            "spot_cost_usd": 0.84,
            "gpu_type": "L4",
        },
        {
            "model": "segresnet",
            "loss": "dice_ce",
            "fold": 0,
            "gpu_hours": 3.1,
            "cost_usd": 1.55,
            "spot_cost_usd": 0.93,
            "gpu_type": "L4",
        },
        {
            "model": "segresnet",
            "loss": "cbdice_cldice",
            "fold": 0,
            "gpu_hours": 3.3,
            "cost_usd": 1.65,
            "spot_cost_usd": 0.99,
            "gpu_type": "L4",
        },
    ]


class TestCostTableLatexOutput:
    """T6: _generate_cost_appendix_table returns valid LaTeX."""

    def test_cost_table_latex_output(self, tmp_path: Path) -> None:
        from minivess.pipeline.biostatistics_tables import (
            _generate_cost_appendix_table,
        )

        cost_data = _make_mock_cost_data()
        artifact = _generate_cost_appendix_table(cost_data, tmp_path)

        assert artifact is not None
        assert artifact.path.exists()

        content = artifact.path.read_text(encoding="utf-8")
        assert "\\begin{table}" in content
        assert "\\end{table}" in content
        assert "\\begin{tabular}" in content
        assert "\\end{tabular}" in content
        assert artifact.format == "latex"


class TestCostTableTotalRow:
    """T6: Table contains a Total summary row."""

    def test_cost_table_total_row(self, tmp_path: Path) -> None:
        from minivess.pipeline.biostatistics_tables import (
            _generate_cost_appendix_table,
        )

        cost_data = _make_mock_cost_data()
        artifact = _generate_cost_appendix_table(cost_data, tmp_path)

        content = artifact.path.read_text(encoding="utf-8")
        assert "Total" in content


class TestCostTableSpotSavings:
    """T6: Spot savings are computed and displayed."""

    def test_cost_table_spot_savings(self, tmp_path: Path) -> None:
        from minivess.pipeline.biostatistics_tables import (
            _generate_cost_appendix_table,
        )

        cost_data = _make_mock_cost_data()
        artifact = _generate_cost_appendix_table(cost_data, tmp_path)

        content = artifact.path.read_text(encoding="utf-8")
        # Should contain "Savings" or "Spot" indicating spot savings column
        assert "Spot" in content or "Savings" in content or "savings" in content


class TestCostTableFromMockMlflowData:
    """T6: Works correctly with mock MLflow data dict."""

    def test_cost_table_from_mock_mlflow_data(self, tmp_path: Path) -> None:
        from minivess.pipeline.biostatistics_tables import (
            _generate_cost_appendix_table,
        )

        cost_data = _make_mock_cost_data()
        artifact = _generate_cost_appendix_table(cost_data, tmp_path)

        assert artifact is not None
        assert artifact.table_id == "cost_appendix"

        content = artifact.path.read_text(encoding="utf-8")
        # Verify booktabs format
        assert "\\toprule" in content
        assert "\\midrule" in content
        assert "\\bottomrule" in content

        # Verify individual model names appear
        assert "dynunet" in content
        assert "segresnet" in content
