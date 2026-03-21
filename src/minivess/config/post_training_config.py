"""Post-training flow configuration for plugin-based post-hoc methods.

Defines the configuration schema for Flow 2.5 (Post-Training), including
checkpoint averaging, subsampled ensemble, SWAG, model merging, calibration,
CRC conformal, and ConSeCo FP control plugins. Each plugin has an
independent ``enabled`` toggle.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Valid factorial post-training methods
FactorialMethod = Literal["none", "checkpoint_averaging", "subsampled_ensemble", "swag"]
VALID_FACTORIAL_METHODS: frozenset[str] = frozenset(
    {"none", "checkpoint_averaging", "subsampled_ensemble", "swag"}
)


def factorial_checkpoint_name(run_id: str, method: str) -> str:
    """Generate deterministic checkpoint filename for a factorial variant.

    Parameters
    ----------
    run_id:
        MLflow run ID or checkpoint identifier.
    method:
        Post-training method name (none, checkpoint_averaging, subsampled_ensemble).

    Returns
    -------
    Checkpoint filename like ``"abc123_swa.pt"``.
    """
    return f"{run_id}_{method}.pt"


class MergeMethod(StrEnum):
    """Model merging interpolation methods."""

    LINEAR = "linear"
    SLERP = "slerp"
    LAYER_WISE = "layer_wise"


class CalibrationMethod(StrEnum):
    """Post-hoc calibration methods."""

    GLOBAL_TEMPERATURE = "global_temperature"
    ISOTONIC_REGRESSION = "isotonic_regression"
    SPATIAL_PLATT = "spatial_platt"


class ShrinkMethod(StrEnum):
    """ConSeCo mask shrinking mechanisms."""

    THRESHOLD = "threshold"
    EROSION = "erosion"


class CheckpointAveragingPluginConfig(BaseModel):
    """Configuration for the checkpoint averaging plugin."""

    enabled: bool = Field(default=True, description="Enable checkpoint averaging")
    per_loss: bool = Field(
        default=True,
        description="Produce one checkpoint-averaged model per loss type",
    )
    cross_loss: bool = Field(
        default=False,
        description="Produce one checkpoint-averaged model across all losses",
    )


class SubsampledEnsemblePluginConfig(BaseModel):
    """Configuration for the subsampled ensemble plugin.

    Subsampled ensemble produces M independent checkpoint-averaged models
    by subsampling checkpoints. This is purely post-hoc — no training-time
    modifications needed.
    NOT Multi-SWAG (which requires training-time second-moment collection).
    """

    enabled: bool = Field(
        default=False,
        description="Enable subsampled ensemble",
    )
    n_models: int = Field(
        default=3,
        ge=2,
        description="Number of independent checkpoint-averaged models to produce",
    )
    subsample_fraction: float = Field(
        default=0.7,
        gt=0.0,
        le=1.0,
        description="Fraction of checkpoints to include in each averaged model",
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducible subsampling",
    )


class SWAGPluginConfig(BaseModel):
    """Configuration for the SWAG plugin (Maddox et al. 2019).

    SWAG approximates the posterior over weights using a low-rank-plus-diagonal
    Gaussian. Requires resumed training to collect first and second moments.

    Reference: Maddox et al. (2019), "A Simple Baseline for Bayesian Inference
    in Deep Learning" (https://arxiv.org/abs/1902.02476)
    """

    enabled: bool = Field(default=False, description="Enable SWAG post-training")
    swa_lr: float = Field(
        default=0.01,
        gt=0.0,
        description="Learning rate for SWA phase (constant or cyclic max)",
    )
    swa_epochs: int = Field(
        default=10,
        ge=1,
        description="Number of additional training epochs for SWAG collection",
    )
    max_rank: int = Field(
        default=20,
        ge=1,
        description="Low-rank covariance approximation rank (K in SWAG paper)",
    )
    n_samples: int = Field(
        default=30,
        ge=1,
        description="Number of posterior samples at inference time",
    )
    update_bn: bool = Field(
        default=True,
        description="Recalibrate BatchNorm stats after weight sampling",
    )


class ModelMergingPluginConfig(BaseModel):
    """Configuration for the model merging plugin."""

    enabled: bool = Field(
        default=True,
        description="Enable model merging",
    )
    method: MergeMethod = Field(
        default=MergeMethod.SLERP,
        description="Interpolation method for merging",
    )
    t: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Interpolation weight (0=model1, 1=model2)",
    )
    merge_pairs: list[list[str]] = Field(
        default_factory=lambda: [["topology", "overlap"]],
        description="Pairs of champion categories to merge",
    )


class CalibrationPluginConfig(BaseModel):
    """Configuration for the post-hoc calibration plugin."""

    enabled: bool = Field(
        default=True,
        description="Enable post-hoc calibration",
    )
    methods: list[CalibrationMethod] = Field(
        default_factory=lambda: [
            CalibrationMethod.GLOBAL_TEMPERATURE,
            CalibrationMethod.ISOTONIC_REGRESSION,
            CalibrationMethod.SPATIAL_PLATT,
        ],
        description="Calibration methods to apply",
    )
    calibration_fraction: float = Field(
        default=0.3,
        gt=0.0,
        le=1.0,
        description="Fraction of validation data used for calibration",
    )
    n_bins: int = Field(
        default=15,
        ge=1,
        description="Number of bins for ECE computation",
    )


class CRCConformalPluginConfig(BaseModel):
    """Configuration for the CRC conformal plugin."""

    enabled: bool = Field(
        default=True,
        description="Enable Conformalized Risk Control",
    )
    alpha: float = Field(
        default=0.1,
        gt=0.0,
        lt=1.0,
        description="Significance level (e.g., 0.1 for 90% coverage)",
    )


class ConSeCoPluginConfig(BaseModel):
    """Configuration for the ConSeCo FP control plugin.

    Reference: arXiv:2511.15406
    """

    enabled: bool = Field(
        default=False,
        description="Enable ConSeCo false-positive control",
    )
    tolerance: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Maximum tolerated false positive rate",
    )
    shrink_method: ShrinkMethod = Field(
        default=ShrinkMethod.EROSION,
        description="Mask shrinking mechanism",
    )


class PostTrainingConfig(BaseModel):
    """Configuration for the post-training flow (Flow 2.5).

    Each plugin sub-config has an ``enabled`` toggle. Disabled plugins
    are skipped during flow execution. The flow sits between training
    and analysis: data → train → **post_training** → analyze → deploy.

    Classification: Best-effort (failure does NOT block analyze flow).
    """

    mlflow_experiment: str = Field(
        default="minivess_training",
        description=(
            "MLflow experiment name for post-training runs. "
            "MUST be same as training (synthesis Part 2.3) so Analysis Flow "
            "discovers all variants in one query."
        ),
    )

    factorial_methods: list[FactorialMethod] = Field(
        default_factory=list,
        description=(
            "Post-training methods to apply in factorial design. "
            "Valid values: none, checkpoint_averaging, subsampled_ensemble. "
            "Empty list = use plugin-by-plugin config (backward compatible)."
        ),
    )

    @field_validator("factorial_methods")
    @classmethod
    def _validate_factorial_methods(
        cls,
        v: list[str],
    ) -> list[str]:
        """Ensure all factorial methods are valid."""
        for method in v:
            if method not in VALID_FACTORIAL_METHODS:
                msg = (
                    f"Invalid factorial method '{method}'. "
                    f"Valid methods: {sorted(VALID_FACTORIAL_METHODS)}"
                )
                raise ValueError(msg)
        return v

    checkpoint_averaging: CheckpointAveragingPluginConfig = Field(
        default_factory=CheckpointAveragingPluginConfig,
        description="Checkpoint averaging plugin config",
    )
    subsampled_ensemble: SubsampledEnsemblePluginConfig = Field(
        default_factory=SubsampledEnsemblePluginConfig,
        description="Subsampled ensemble plugin config",
    )
    swag: SWAGPluginConfig = Field(
        default_factory=SWAGPluginConfig,
        description="SWAG posterior approximation plugin config (Maddox et al. 2019)",
    )
    model_merging: ModelMergingPluginConfig = Field(
        default_factory=ModelMergingPluginConfig,
        description="Model merging plugin config",
    )
    calibration: CalibrationPluginConfig = Field(
        default_factory=CalibrationPluginConfig,
        description="Post-hoc calibration plugin config",
    )
    crc_conformal: CRCConformalPluginConfig = Field(
        default_factory=CRCConformalPluginConfig,
        description="CRC conformal plugin config",
    )
    conseco_fp_control: ConSeCoPluginConfig = Field(
        default_factory=ConSeCoPluginConfig,
        description="ConSeCo FP control plugin config",
    )

    def enabled_plugin_names(self) -> list[str]:
        """Return names of all enabled plugins."""
        plugins = {
            "checkpoint_averaging": self.checkpoint_averaging.enabled,
            "subsampled_ensemble": self.subsampled_ensemble.enabled,
            "swag": self.swag.enabled,
            "model_merging": self.model_merging.enabled,
            "calibration": self.calibration.enabled,
            "crc_conformal": self.crc_conformal.enabled,
            "conseco_fp_control": self.conseco_fp_control.enabled,
        }
        return [name for name, enabled in plugins.items() if enabled]
