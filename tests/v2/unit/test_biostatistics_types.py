"""Unit tests for biostatistics result and manifest types (Task 1.2)."""

from __future__ import annotations


class TestSourceRunManifest:
    """SourceRunManifest fingerprinting and construction."""

    def test_source_run_manifest_fingerprint_deterministic(self) -> None:
        """Same runs produce the same fingerprint."""
        from minivess.pipeline.biostatistics_types import SourceRun, SourceRunManifest

        runs = [
            SourceRun(
                run_id="a1",
                experiment_id="e1",
                experiment_name="exp",
                loss_function="dice",
                fold_id=0,
                status="FINISHED",
            ),
            SourceRun(
                run_id="b2",
                experiment_id="e1",
                experiment_name="exp",
                loss_function="dice",
                fold_id=1,
                status="FINISHED",
            ),
        ]
        m1 = SourceRunManifest.from_runs(runs)
        m2 = SourceRunManifest.from_runs(runs)
        assert m1.fingerprint == m2.fingerprint
        assert len(m1.fingerprint) == 64  # SHA-256 hex length

    def test_source_run_manifest_fingerprint_changes_with_run_ids(self) -> None:
        """Different run IDs produce different fingerprints."""
        from minivess.pipeline.biostatistics_types import SourceRun, SourceRunManifest

        runs_a = [
            SourceRun(
                run_id="a1",
                experiment_id="e1",
                experiment_name="exp",
                loss_function="dice",
                fold_id=0,
                status="FINISHED",
            ),
        ]
        runs_b = [
            SourceRun(
                run_id="z9",
                experiment_id="e1",
                experiment_name="exp",
                loss_function="dice",
                fold_id=0,
                status="FINISHED",
            ),
        ]
        m_a = SourceRunManifest.from_runs(runs_a)
        m_b = SourceRunManifest.from_runs(runs_b)
        assert m_a.fingerprint != m_b.fingerprint


class TestValidationResult:
    """ValidationResult status logic."""

    def test_validation_result_valid_when_no_errors(self) -> None:
        """valid=True when errors list is empty."""
        from minivess.pipeline.biostatistics_types import ValidationResult

        result = ValidationResult(
            valid=True,
            warnings=[],
            errors=[],
            n_conditions=4,
            n_folds_per_condition=3,
        )
        assert result.valid is True
        assert result.n_conditions == 4

    def test_validation_result_invalid_with_errors(self) -> None:
        """valid=False when errors are present."""
        from minivess.pipeline.biostatistics_types import ValidationResult

        result = ValidationResult(
            valid=False,
            warnings=[],
            errors=["missing folds"],
            n_conditions=2,
            n_folds_per_condition=1,
        )
        assert result.valid is False
        assert len(result.errors) == 1


class TestPairwiseResult:
    """PairwiseResult effect size interpretation."""

    def test_pairwise_result_effect_size_interpretation(self) -> None:
        """PairwiseResult stores all three effect sizes."""
        from minivess.pipeline.biostatistics_types import PairwiseResult

        result = PairwiseResult(
            condition_a="dice_ce",
            condition_b="cbdice_cldice",
            metric="cldice",
            p_value=0.001,
            p_adjusted=0.006,
            correction_method="holm",
            significant=True,
            cohens_d=0.85,
            cliffs_delta=0.72,
            vda=0.86,
        )
        assert result.significant is True
        assert result.cohens_d == 0.85
        assert result.cliffs_delta == 0.72
        assert result.vda == 0.86

    def test_pairwise_result_bayesian_fields_optional(self) -> None:
        """Bayesian fields default to None."""
        from minivess.pipeline.biostatistics_types import PairwiseResult

        result = PairwiseResult(
            condition_a="a",
            condition_b="b",
            metric="dsc",
            p_value=0.05,
            p_adjusted=0.05,
            correction_method="holm",
            significant=False,
            cohens_d=0.1,
            cliffs_delta=0.05,
            vda=0.52,
        )
        assert result.bayesian_left is None
        assert result.bayesian_rope is None
        assert result.bayesian_right is None


class TestBiostatisticsResult:
    """BiostatisticsResult artifact counts."""

    def test_biostatistics_result_artifact_counts(self) -> None:
        """BiostatisticsResult tracks figures and tables."""
        from pathlib import Path

        from minivess.pipeline.biostatistics_types import (
            BiostatisticsResult,
            FigureArtifact,
            SourceRunManifest,
            TableArtifact,
        )

        manifest = SourceRunManifest(
            runs=[], fingerprint="abc" * 21 + "a", discovered_at="now"
        )
        result = BiostatisticsResult(
            manifest=manifest,
            db_path=Path("/tmp/test.duckdb"),
            pairwise=[],
            variance=[],
            rankings=[],
            figures=[
                FigureArtifact(
                    figure_id="F1", title="Raincloud", paths=[Path("f1.png")]
                ),
                FigureArtifact(figure_id="F2", title="Heatmap", paths=[Path("f2.png")]),
            ],
            tables=[
                TableArtifact(
                    table_id="T1",
                    title="Comparison",
                    path=Path("t1.tex"),
                    format="latex",
                ),
            ],
        )
        assert len(result.figures) == 2
        assert len(result.tables) == 1
        assert result.db_path.name == "test.duckdb"
