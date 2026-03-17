"""Result and manifest types for the biostatistics flow.

Pure dataclasses — no Prefect, no external library dependencies.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class SourceRun:
    """A single MLflow training run discovered for analysis."""

    run_id: str
    experiment_id: str
    experiment_name: str
    loss_function: str
    fold_id: int
    status: str


@dataclass
class SourceRunManifest:
    """Collection of source runs with a deterministic fingerprint."""

    runs: list[SourceRun]
    fingerprint: str
    discovered_at: str

    @classmethod
    def from_runs(cls, runs: list[SourceRun]) -> SourceRunManifest:
        """Build manifest from a list of runs, computing SHA-256 fingerprint."""
        sorted_ids = sorted(r.run_id for r in runs)
        digest = hashlib.sha256("|".join(sorted_ids).encode("utf-8")).hexdigest()
        return cls(
            runs=runs,
            fingerprint=digest,
            discovered_at=datetime.now(UTC).isoformat(),
        )


@dataclass
class ValidationResult:
    """Result of source data completeness validation."""

    valid: bool
    warnings: list[str]
    errors: list[str]
    n_conditions: int
    n_folds_per_condition: int


@dataclass
class PairwiseResult:
    """Result of a single pairwise statistical comparison."""

    condition_a: str
    condition_b: str
    metric: str
    p_value: float
    p_adjusted: float
    correction_method: str
    significant: bool
    cohens_d: float
    cliffs_delta: float
    vda: float
    # Bayesian fields — populated by baycomp if available
    bayesian_left: float | None = None
    bayesian_rope: float | None = None
    bayesian_right: float | None = None


@dataclass
class VarianceDecompositionResult:
    """Result of Friedman test + ICC analysis for one metric."""

    metric: str
    friedman_statistic: float
    friedman_p: float
    nemenyi_matrix: dict[str, dict[str, float]] | None
    icc_value: float
    icc_ci_lower: float
    icc_ci_upper: float
    icc_type: str  # e.g., "ICC2"
    power_caveat: bool = True  # Always True for K=3


@dataclass
class RankingResult:
    """Multi-metric ranking for one metric."""

    metric: str
    condition_ranks: dict[str, float]
    mean_ranks: dict[str, float]
    cd_value: float | None


@dataclass
class FigureArtifact:
    """Reference to a generated figure."""

    figure_id: str
    title: str
    paths: list[Path]
    sidecar_path: Path | None = None


@dataclass
class TableArtifact:
    """Reference to a generated table."""

    table_id: str
    title: str
    path: Path
    format: str  # "latex", "markdown"


@dataclass
class FactorialAnovaResult:
    """Result of two-way factorial ANOVA for one metric."""

    metric: str
    n_models: int
    n_losses: int
    f_values: dict[str, float]
    p_values: dict[str, float]
    eta_squared_partial: dict[str, float]
    omega_squared: dict[str, float]
    engine_pingouin: dict[str, object] | None = None
    engine_statsmodels: dict[str, object] | None = None


@dataclass
class CalibrationMetricsResult:
    """Result of calibration metrics for one set of predictions."""

    brier_score: float
    oe_ratio: float
    ipa: float
    calibration_slope: float


@dataclass
class BiostatisticsResult:
    """Complete result of the biostatistics flow."""

    manifest: SourceRunManifest
    db_path: Path
    pairwise: list[PairwiseResult]
    variance: list[VarianceDecompositionResult]
    rankings: list[RankingResult]
    figures: list[FigureArtifact] = field(default_factory=list)
    tables: list[TableArtifact] = field(default_factory=list)
    mlflow_run_id: str | None = None
