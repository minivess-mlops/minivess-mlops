"""Tests for TRIPOD+AI compliance verification.

Validates that the biostatistics flow output includes all required
statistical methods, figures, and tables.
"""

from __future__ import annotations

from pathlib import Path


def _make_mock_lineage_manifest() -> dict:
    """Create mock lineage manifest with expected fields."""
    from minivess.pipeline.biostatistics_lineage import build_lineage_manifest
    from minivess.pipeline.biostatistics_types import (
        FigureArtifact,
        SourceRun,
        SourceRunManifest,
        TableArtifact,
    )

    runs = [
        SourceRun(
            run_id=f"run_{i}",
            experiment_id="exp_1",
            experiment_name="minivess_training",
            loss_function="cbdice_cldice",
            fold_id=i % 3,
            status="FINISHED",
        )
        for i in range(12)
    ]
    manifest = SourceRunManifest.from_runs(runs)

    figures = [
        FigureArtifact(
            figure_id="interaction_plot",
            title="Model x Loss Interaction",
            paths=[Path("/tmp/interaction.png")],
        ),
        FigureArtifact(
            figure_id="effect_size_heatmap",
            title="Effect Size Heatmap",
            paths=[Path("/tmp/heatmap.png")],
        ),
        FigureArtifact(
            figure_id="distribution_plot",
            title="Score Distributions",
            paths=[Path("/tmp/dist.png")],
        ),
        FigureArtifact(
            figure_id="instability_plot",
            title="Riley Instability",
            paths=[Path("/tmp/instab.png")],
        ),
    ]

    tables = [
        TableArtifact(
            table_id="anova_table",
            title="ANOVA Summary",
            path=Path("/tmp/anova.tex"),
            format="latex",
        ),
        TableArtifact(
            table_id="cost_table",
            title="Cost Appendix",
            path=Path("/tmp/cost.tex"),
            format="latex",
        ),
        TableArtifact(
            table_id="ranking_table",
            title="Ranking",
            path=Path("/tmp/ranking.tex"),
            format="latex",
        ),
    ]

    return build_lineage_manifest(manifest, figures, tables)


class TestTripodStatisticalMethodsCoverage:
    """T7: Statistical methods list includes all planned analyses."""

    def test_tripod_statistical_methods_coverage(self) -> None:
        lineage = _make_mock_lineage_manifest()
        methods = lineage["statistical_methods"]

        # ANOVA methods should be present
        required_keywords = [
            "Wilcoxon",
            "Holm-Bonferroni",
            "Friedman",
            "ICC",
            "Bayesian",
        ]
        methods_text = " ".join(methods)
        for keyword in required_keywords:
            assert keyword in methods_text, f"Missing statistical method: {keyword}"


class TestTripodFigureCatalogCoverage:
    """T7: Figure catalog includes all planned figures."""

    def test_tripod_figure_catalog_coverage(self) -> None:
        lineage = _make_mock_lineage_manifest()
        figure_ids = lineage["artifacts_produced"]["figure_ids"]

        assert "interaction_plot" in figure_ids
        assert "effect_size_heatmap" in figure_ids
        assert lineage["artifacts_produced"]["n_figures"] >= 3


class TestTripodTableCatalogCoverage:
    """T7: Table catalog includes all planned tables."""

    def test_tripod_table_catalog_coverage(self) -> None:
        lineage = _make_mock_lineage_manifest()
        table_ids = lineage["artifacts_produced"]["table_ids"]

        assert "anova_table" in table_ids
        assert "cost_table" in table_ids
        assert lineage["artifacts_produced"]["n_tables"] >= 2


class TestTripodLineageManifestFields:
    """T7: Lineage manifest has all required fields."""

    def test_tripod_lineage_manifest_fields(self) -> None:
        lineage = _make_mock_lineage_manifest()

        required_fields = [
            "schema_version",
            "generated_at",
            "fingerprint",
            "n_source_runs",
            "source_experiments",
            "artifacts_produced",
            "statistical_methods",
        ]
        for field_name in required_fields:
            assert field_name in lineage, f"Missing field: {field_name}"

        assert lineage["n_source_runs"] == 12
        assert len(lineage["source_experiments"]) >= 1
