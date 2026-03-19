"""Configuration for the Biostatistics Prefect Flow.

Defines ``BiostatisticsConfig`` — Pydantic model controlling which experiments
to analyze, which metrics to compute, and statistical testing parameters.
"""

from __future__ import annotations

from pathlib import (
    Path,  # noqa: TC003 — Pydantic needs runtime access for field validation
)

from pydantic import BaseModel, Field, field_validator, model_validator

# Default 8 metrics per CLAUDE.md and biostatistics plan
_DEFAULT_METRICS: list[str] = [
    "dsc",
    "hd95",
    "assd",
    "nsd",
    "cldice",
    "be_0",
    "be_1",
    "junction_f1",
]

# ROPE values per reviewer findings: DSC ±0.01, clDice ±0.01, HD95 ±0.5, NSD ±0.01
_DEFAULT_ROPE_VALUES: dict[str, float] = {
    "dsc": 0.01,
    "cldice": 0.01,
    "hd95": 0.5,
    "nsd": 0.01,
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

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    metrics: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_METRICS),
        description="Metrics to include in the statistical analysis.",
    )
    primary_metric: str = Field(
        default="cldice",
        description="Pre-registered primary metric for Holm-Bonferroni correction.",
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
        description="Number of BCa bootstrap resamples.",
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
