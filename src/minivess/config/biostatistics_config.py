"""Configuration for the Biostatistics Prefect Flow.

Defines ``BiostatisticsConfig`` — Pydantic model controlling which experiments
to analyze, which metrics to compute, and statistical testing parameters.
"""

from __future__ import annotations

from pathlib import (
    Path,  # noqa: TC003 — Pydantic needs runtime access for field validation
)

from pydantic import BaseModel, Field, field_validator, model_validator

# Default 8 segmentation + 6 calibration metrics per CLAUDE.md and biostatistics plan
_DEFAULT_METRICS: list[str] = [
    "dsc",
    "hd95",
    "assd",
    "nsd",
    "cldice",
    "be_0",
    "be_1",
    "junction_f1",
    # Calibration metrics (Phase B3)
    "cal_ece",
    "cal_mce",
    "cal_brier",
    "cal_nll",
    "cal_ace",
    "cal_ba_ece",
]

# ROPE values per reviewer findings: DSC ±0.01, clDice ±0.01, HD95 ±0.5, NSD ±0.01
# Calibration ROPE: ECE ±0.02, Brier ±0.01, BA-ECE ±0.02
_DEFAULT_ROPE_VALUES: dict[str, float] = {
    "dsc": 0.01,
    "cldice": 0.01,
    "hd95": 0.5,
    "nsd": 0.01,
    # Calibration metrics ROPE (Phase B3)
    "cal_ece": 0.02,
    "cal_brier": 0.01,
    "cal_ba_ece": 0.02,
}


class BiostatisticsConfig(BaseModel):
    """Configuration for the biostatistics flow (Flow 5b).

    Controls experiment selection, metric computation, and statistical testing
    parameters for cross-loss comparison analysis.
    """

    # ------------------------------------------------------------------
    # Data source
    # ------------------------------------------------------------------
    experiment_names: list[str] = Field(
        default_factory=lambda: ["dynunet_loss_variation_v2"],
        description="MLflow experiment names to include in the analysis.",
    )
    mlruns_dir: Path = Field(
        default_factory=lambda: Path("mlruns"),
        description="Path to the MLflow mlruns directory.",
    )
    output_dir: Path = Field(
        default_factory=lambda: Path("outputs/biostatistics"),
        description="Directory for generated artifacts (figures, tables, DuckDB).",
    )
    analysis_duckdb_path: Path | None = Field(
        default=None,
        description=(
            "Path to analysis_results.duckdb (DuckDB 1). When provided, the "
            "Biostatistics Flow reads per-volume data from this DuckDB instead "
            "of querying MLflow API. When None, falls back to MLflow API."
        ),
    )
    factorial_yaml: Path | None = Field(
        default=None,
        description=(
            "Path to the composable factorial YAML (configs/factorial/*.yaml). "
            "If provided, factor names are auto-derived from this YAML. "
            "If None, factor_names must be provided explicitly."
        ),
    )
    factor_names: list[str] = Field(
        default_factory=lambda: ["model_family", "loss_name", "aux_calibration"],
        description=(
            "Factor names for ANOVA. Auto-derived from factorial_yaml if provided. "
            "Default is Layer A training factors only (backward compat)."
        ),
    )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    metrics: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_METRICS),
        description="Metrics to include in the statistical analysis.",
    )
    primary_metric: str = Field(
        default="cldice",
        description="Legacy single primary metric. Use co_primary_metrics instead.",
    )
    co_primary_metrics: list[str] = Field(
        default_factory=lambda: ["cldice", "masd"],
        description=(
            "Co-primary metrics (MetricsReloaded-driven). Both get Holm-Bonferroni "
            "correction. DSC is a FOIL metric (BH-FDR). The rank inversion between "
            "DSC and clDice IS a paper finding."
        ),
    )
    foil_metrics: list[str] = Field(
        default_factory=lambda: ["dsc"],
        description=(
            "FOIL metrics included to demonstrate misleading rankings for tubular "
            "structures. Get BH-FDR correction (secondary tier)."
        ),
    )
    calibration_co_primary_metrics: list[str] = Field(
        default_factory=lambda: ["cal_ece", "cal_ba_ece"],
        description=(
            "Co-primary calibration metrics for the factorial ANOVA. "
            "ECE (global) + BA-ECE (spatial/boundary). "
            "Both get Holm-Bonferroni correction alongside segmentation co-primaries."
        ),
    )

    # ------------------------------------------------------------------
    # Statistical testing
    # ------------------------------------------------------------------
    alpha: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Significance level for hypothesis tests.",
    )
    n_bootstrap: int = Field(
        default=10_000,
        ge=100,
        description="Number of bootstrap resamples.",
    )
    bca_min_n: int = Field(
        default=20,
        ge=5,
        description=(
            "Minimum sample size for BCa bootstrap. Below this, percentile "
            "bootstrap is used. Per DiCiccio & Efron (1996, Statistical Science)."
        ),
    )
    pece_fp_weight: float = Field(
        default=2.0,
        gt=0.0,
        description=(
            "False-positive overconfidence weight for pECE metric. "
            "Li et al. (2025) default is 2.0. Higher values penalize "
            "overconfident FPs more heavily."
        ),
    )
    rope_values: dict[str, float] = Field(
        default_factory=lambda: dict(_DEFAULT_ROPE_VALUES),
        description="Region of practical equivalence widths per metric.",
    )
    min_folds_per_condition: int = Field(
        default=3,
        ge=1,
        description="Minimum completed folds required per condition.",
    )
    min_conditions: int = Field(
        default=2,
        ge=2,
        description="Minimum number of conditions (loss functions) required.",
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility.",
    )
    splits: list[str] = Field(
        default_factory=lambda: ["trainval", "test"],
        description=(
            "Evaluation splits to analyze. 'trainval' = MiniVess cross-validated "
            "metrics (eval/ prefix). 'test' = external test set metrics (test/ prefix)."
        ),
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def _primary_metric_in_metrics(self) -> BiostatisticsConfig:
        """Ensure primary_metric is in the metrics list."""
        if self.primary_metric not in self.metrics:
            msg = (
                f"primary_metric '{self.primary_metric}' must be in metrics list "
                f"{self.metrics}"
            )
            raise ValueError(msg)
        return self

    @field_validator("rope_values")
    @classmethod
    def _rope_values_positive(cls, v: dict[str, float]) -> dict[str, float]:
        """All ROPE values must be positive."""
        for key, val in v.items():
            if val <= 0:
                msg = f"rope value for '{key}' must be positive, got {val}"
                raise ValueError(msg)
        return v
