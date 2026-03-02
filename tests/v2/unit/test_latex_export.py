"""Tests for LaTeX comparison table export.

Covers:
- Booktabs markers present (toprule, midrule, bottomrule)
- Best values bolded
- NaN handling
- Roundtrip consistency with markdown
- Empty table handling
- Multi-metric table

Closes #183.
"""

from __future__ import annotations

from minivess.pipeline.comparison import (
    ComparisonTable,
    LossResult,
    MetricSummary,
    format_comparison_latex,
    format_comparison_markdown,
)


def _make_table() -> ComparisonTable:
    """Build a sample ComparisonTable with two losses and two metrics."""
    return ComparisonTable(
        losses=[
            LossResult(
                loss_name="dice_ce",
                num_folds=3,
                metrics={
                    "dsc": MetricSummary(
                        mean=0.824,
                        std=0.015,
                        ci_lower=0.80,
                        ci_upper=0.85,
                        per_fold=[0.81, 0.83, 0.83],
                    ),
                    "centreline_dsc": MetricSummary(
                        mean=0.832,
                        std=0.020,
                        ci_lower=0.81,
                        ci_upper=0.86,
                        per_fold=[0.82, 0.83, 0.84],
                    ),
                },
            ),
            LossResult(
                loss_name="cbdice_cldice",
                num_folds=3,
                metrics={
                    "dsc": MetricSummary(
                        mean=0.780,
                        std=0.025,
                        ci_lower=0.75,
                        ci_upper=0.81,
                        per_fold=[0.76, 0.78, 0.80],
                    ),
                    "centreline_dsc": MetricSummary(
                        mean=0.906,
                        std=0.010,
                        ci_lower=0.89,
                        ci_upper=0.92,
                        per_fold=[0.90, 0.91, 0.91],
                    ),
                },
            ),
        ],
        metric_names=["centreline_dsc", "dsc"],
    )


class TestFormatComparisonLatex:
    """Validate LaTeX table formatting."""

    def test_booktabs_markers_present(self) -> None:
        table = _make_table()
        latex = format_comparison_latex(table)
        assert r"\toprule" in latex
        assert r"\midrule" in latex
        assert r"\bottomrule" in latex

    def test_best_values_bolded(self) -> None:
        table = _make_table()
        latex = format_comparison_latex(table)
        # Best DSC is dice_ce (0.824), best clDice is cbdice_cldice (0.906)
        assert r"\textbf{0.8240}" in latex
        assert r"\textbf{0.9060}" in latex

    def test_tabular_environment(self) -> None:
        table = _make_table()
        latex = format_comparison_latex(table)
        assert r"\begin{tabular}" in latex
        assert r"\end{tabular}" in latex

    def test_nan_handling(self) -> None:
        table = ComparisonTable(
            losses=[
                LossResult(
                    loss_name="broken",
                    num_folds=1,
                    metrics={
                        "dsc": MetricSummary(
                            mean=float("nan"),
                            std=float("nan"),
                            ci_lower=float("nan"),
                            ci_upper=float("nan"),
                            per_fold=[],
                        ),
                    },
                ),
            ],
            metric_names=["dsc"],
        )
        latex = format_comparison_latex(table)
        assert "N/A" in latex

    def test_empty_table(self) -> None:
        table = ComparisonTable(losses=[], metric_names=[])
        latex = format_comparison_latex(table)
        assert "No results" in latex

    def test_markdown_and_latex_same_data(self) -> None:
        """Ensure markdown and LaTeX contain the same loss names."""
        table = _make_table()
        md = format_comparison_markdown(table)
        latex = format_comparison_latex(table)
        assert "dice_ce" in md
        assert r"dice\_ce" in latex
        assert "cbdice_cldice" in md
        assert r"cbdice\_cldice" in latex

    def test_loss_name_escaped(self) -> None:
        """Underscores in loss names should be escaped for LaTeX."""
        table = _make_table()
        latex = format_comparison_latex(table)
        assert r"dice\_ce" in latex
        assert r"cbdice\_cldice" in latex

    def test_column_alignment(self) -> None:
        """Table should have left-aligned first column and centered metrics."""
        table = _make_table()
        latex = format_comparison_latex(table)
        # Check that there's an alignment spec with 'l' and 'c' columns
        assert "{lcc}" in latex or "{l c c}" in latex
