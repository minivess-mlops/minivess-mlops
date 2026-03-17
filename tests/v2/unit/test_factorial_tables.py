"""Tests for factorial ANOVA LaTeX table generation.

Validates _generate_anova_table() which produces publication-quality LaTeX tables
with booktabs formatting for the two-way factorial ANOVA results.
"""

from __future__ import annotations

from pathlib import Path

from minivess.pipeline.biostatistics_types import FactorialAnovaResult


def _make_mock_anova_result(
    *,
    model_p: float = 0.0001,
    loss_p: float = 0.04,
    interaction_p: float = 0.12,
) -> FactorialAnovaResult:
    """Create a mock FactorialAnovaResult for testing."""
    return FactorialAnovaResult(
        metric="cldice",
        n_models=4,
        n_losses=3,
        f_values={
            "Model": 12.5,
            "Loss": 3.2,
            "Model:Loss": 1.8,
            "Residual": 0.0,
        },
        p_values={
            "Model": model_p,
            "Loss": loss_p,
            "Model:Loss": interaction_p,
        },
        eta_squared_partial={
            "Model": 0.35,
            "Loss": 0.08,
            "Model:Loss": 0.04,
            "Residual": 0.51,
        },
        omega_squared={
            "Model": 0.30,
            "Loss": 0.06,
            "Model:Loss": 0.02,
            "Residual": 0.50,
        },
    )


class TestAnovaTableLatexOutput:
    """T3: _generate_anova_table returns valid LaTeX string."""

    def test_anova_table_latex_output(self, tmp_path: Path) -> None:
        from minivess.pipeline.biostatistics_tables import _generate_anova_table

        anova_result = _make_mock_anova_result()
        artifact = _generate_anova_table(anova_result, tmp_path)

        assert artifact is not None
        assert artifact.path.exists()

        content = artifact.path.read_text(encoding="utf-8")
        assert "\\begin{table}" in content
        assert "\\end{table}" in content
        assert "\\begin{tabular}" in content
        assert "\\end{tabular}" in content
        assert artifact.format == "latex"


class TestAnovaTableAllFactors:
    """T3: Table contains Model, Loss, Model:Loss, and Residual rows."""

    def test_anova_table_all_factors(self, tmp_path: Path) -> None:
        from minivess.pipeline.biostatistics_tables import _generate_anova_table

        anova_result = _make_mock_anova_result()
        artifact = _generate_anova_table(anova_result, tmp_path)

        content = artifact.path.read_text(encoding="utf-8")
        assert "Model" in content
        assert "Loss" in content
        # Interaction term — may be "Model:Loss" or "Model $\\times$ Loss"
        assert "Model" in content and "Loss" in content
        assert "Residual" in content


class TestAnovaTableSignificanceStars:
    """T3: *** for p<0.001, ** for p<0.01, * for p<0.05."""

    def test_anova_table_significance_stars(self, tmp_path: Path) -> None:
        from minivess.pipeline.biostatistics_tables import _generate_anova_table

        anova_result = _make_mock_anova_result(
            model_p=0.0001,  # *** (p < 0.001)
            loss_p=0.005,  # ** (p < 0.01)
            interaction_p=0.04,  # * (p < 0.05)
        )
        artifact = _generate_anova_table(anova_result, tmp_path)

        content = artifact.path.read_text(encoding="utf-8")
        # Should contain significance markers
        assert "***" in content  # Model p < 0.001
        assert "**" in content  # Loss p < 0.01


class TestAnovaTableBooktabsFormat:
    """T3: Contains \\toprule, \\midrule, \\bottomrule."""

    def test_anova_table_booktabs_format(self, tmp_path: Path) -> None:
        from minivess.pipeline.biostatistics_tables import _generate_anova_table

        anova_result = _make_mock_anova_result()
        artifact = _generate_anova_table(anova_result, tmp_path)

        content = artifact.path.read_text(encoding="utf-8")
        assert "\\toprule" in content
        assert "\\midrule" in content
        assert "\\bottomrule" in content
