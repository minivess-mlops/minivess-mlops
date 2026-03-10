from __future__ import annotations

from enum import StrEnum
from pathlib import (
    Path,  # noqa: TC003 — Pydantic needs runtime access for field validation
)
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class EnsembleStrategyName(StrEnum):
    """Named ensemble construction strategies for model selection.

    These describe *which* checkpoints to group before applying an
    ``EnsembleStrategy`` (mean, vote, soup, ...) from ``models.py``.

    - ``PER_LOSS_SINGLE_BEST``  -- best fold per loss (4 models, one per loss)
    - ``ALL_LOSS_SINGLE_BEST``  -- single best model across all losses and folds
    - ``PER_LOSS_ALL_BEST``     -- all best-fold models per loss (12 models: 4 losses x 3 folds)
    - ``ALL_LOSS_ALL_BEST``     -- every best-fold model from every loss (12 models pooled)
    """

    PER_LOSS_SINGLE_BEST = "per_loss_single_best"
    ALL_LOSS_SINGLE_BEST = "all_loss_single_best"
    PER_LOSS_ALL_BEST = "per_loss_all_best"
    ALL_LOSS_ALL_BEST = "all_loss_all_best"


class MetricDirection(StrEnum):
    """Whether higher or lower metric values are better."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class InferenceStrategyConfig(BaseModel):
    """Configuration for a single sliding-window inference strategy.

    Multiple strategies allow comparing per-model-optimal vs. fixed-patch
    evaluation in a single analysis run. The ``is_primary`` strategy produces
    bare metric keys (for paper tables); all others are prefixed with the
    strategy name (e.g., ``fast/dsc``).
    """

    name: str
    description: str = ""
    roi_size: list[int] | Literal["per_model"]
    overlap: float = Field(default=0.5, ge=0.0, lt=1.0)
    sw_batch_size: int = Field(default=4, ge=1)
    aggregation_mode: Literal["gaussian", "constant"] = "gaussian"
    is_primary: bool = False


class EvaluationConfig(BaseModel):
    """Configuration for the evaluation and analysis flow (Prefect Flow 3).

    Defines the *single best metric* used for model selection, the ensemble
    strategies to build, and statistical testing parameters.
    """

    # ------------------------------------------------------------------
    # Primary metric for "single best" model selection
    # ------------------------------------------------------------------
    primary_metric: str = Field(
        default="val_compound_masd_cldice",
        description="Metric name used to rank checkpoints and select the best model.",
    )
    primary_metric_direction: MetricDirection = Field(
        default=MetricDirection.MAXIMIZE,
        description="Whether the primary metric should be maximized or minimized.",
    )

    # ------------------------------------------------------------------
    # MLflow experiment names
    # ------------------------------------------------------------------
    mlflow_evaluation_experiment: str = Field(
        default="minivess_evaluation",
        description="MLflow experiment that stores evaluation runs.",
    )
    mlflow_training_experiment: str = Field(
        default="minivess_training",
        description="MLflow experiment that stores training runs (read-only during eval).",
    )
    require_eval_metrics: bool = Field(
        default=True,
        description=(
            "When True (production), skip runs without eval_fold2_dsc metric. "
            "Set to False for debug runs that don't produce full eval metrics (#588)."
        ),
    )

    # ------------------------------------------------------------------
    # MetricsReloaded / statistical testing
    # ------------------------------------------------------------------
    include_expensive_metrics: bool = Field(
        default=True,
        description="Whether to compute skeleton-based metrics (clDice, MASD).",
    )
    bootstrap_n_resamples: int = Field(
        default=10_000,
        ge=100,
        description="Number of bootstrap resamples for paired comparison tests.",
    )
    confidence_level: float = Field(
        default=0.95,
        gt=0,
        lt=1,
        description="Confidence level for bootstrap confidence intervals.",
    )

    # ------------------------------------------------------------------
    # Ensemble strategies to build
    # ------------------------------------------------------------------
    ensemble_strategies: list[EnsembleStrategyName] = Field(
        default_factory=lambda: list(EnsembleStrategyName),
        description="Which checkpoint-grouping strategies to evaluate.",
    )

    # ------------------------------------------------------------------
    # Model registration
    # ------------------------------------------------------------------
    model_registry_name: str = Field(
        default="MiniVess-Segmentor",
        description="MLflow Model Registry name for promoted models.",
    )

    # ------------------------------------------------------------------
    # Optional external dataset config
    # ------------------------------------------------------------------
    datasets_config: Path | None = Field(
        default=None,
        description="Path to a YAML file listing evaluation datasets.",
    )

    # ------------------------------------------------------------------
    # Inference strategies (multi-strategy evaluation — #505)
    # ------------------------------------------------------------------
    inference_strategies: list[InferenceStrategyConfig] = Field(
        default_factory=lambda: [
            InferenceStrategyConfig(
                name="standard_patch",
                description="Fixed patch across ALL models — use for paper tables",
                roi_size=[128, 128, 16],
                overlap=0.5,
                sw_batch_size=4,
                is_primary=True,
            )
        ],
        description="Sliding-window strategies to run during evaluation.",
    )

    @model_validator(mode="after")
    def _validate_strategies(self) -> EvaluationConfig:
        strats = self.inference_strategies
        if strats:
            primaries = [s for s in strats if s.is_primary]
            if len(primaries) != 1:
                msg = (
                    f"Exactly 1 inference strategy must have is_primary=True, "
                    f"got {len(primaries)}"
                )
                raise ValueError(msg)
            names = [s.name for s in strats]
            if len(names) != len(set(names)):
                msg = "Inference strategy names must be unique"
                raise ValueError(msg)
        return self

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("primary_metric_direction", mode="before")
    @classmethod
    def validate_direction(cls, v: object) -> object:
        """Accept string values and validate they map to MetricDirection."""
        if isinstance(v, str):
            lowered = v.lower()
            valid = {member.value for member in MetricDirection}
            if lowered not in valid:
                msg = (
                    f"Invalid direction '{v}'. "
                    f"Must be one of: {', '.join(sorted(valid))}"
                )
                raise ValueError(msg)
            return lowered
        return v

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------
    def checkpoint_filename(self) -> str:
        """Return the ``.pth`` filename corresponding to the primary metric.

        Example: ``"val_compound_masd_cldice"`` -> ``"best_val_compound_masd_cldice.pth"``
        """
        return f"best_{self.primary_metric}.pth"

    def is_better(self, current: float, best: float) -> bool:
        """Check whether *current* metric value is strictly better than *best*.

        Respects :attr:`primary_metric_direction`:
        - ``MAXIMIZE``: ``current > best``
        - ``MINIMIZE``: ``current < best``
        """
        if self.primary_metric_direction == MetricDirection.MAXIMIZE:
            return current > best
        return current < best
