"""Post-training flow configuration for plugin-based post-hoc methods.

Defines the configuration schema for Flow 2.5 (Post-Training), including
SWA, Multi-SWA, model merging, calibration, CRC conformal, and ConSeCo
FP control plugins. Each plugin has an independent ``enabled`` toggle.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Valid factorial post-training methods
FactorialMethod = Literal["none", "swa", "multi_swa"]
VALID_FACTORIAL_METHODS: frozenset[str] = frozenset({"none", "swa", "multi_swa"})


def factorial_checkpoint_name(run_id: str, method: str) -> str:
    """Generate deterministic checkpoint filename for a factorial variant.

    Parameters
    ----------
    run_id:
        MLflow run ID or checkpoint identifier.
    method:
        Post-training method name (none, swa, multi_swa).

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


class SWAPluginConfig(BaseModel):
    """Configuration for the SWA plugin."""

    enabled: bool = Field(default=True, description="Enable SWA weight averaging")
    per_loss: bool = Field(
        default=True,
        description="Produce one SWA model per loss type",
    )
    cross_loss: bool = Field(
        default=False,
        description="Produce one SWA model across all losses",
    )


class MultiSWAPluginConfig(BaseModel):
    """Configuration for the Multi-SWA plugin.

    Multi-SWA produces M independent SWA models by subsampling checkpoints.
    This is purely post-hoc — no training-time modifications needed.
    NOT Multi-SWAG (which requires training-time second-moment collection).
    """

    enabled: bool = Field(
        default=False,
        description="Enable Multi-SWA ensemble",
    )
    n_models: int = Field(
        default=3,
        ge=2,
        description="Number of independent SWA models to produce",
    )
    subsample_fraction: float = Field(
        default=0.7,
        gt=0.0,
        le=1.0,
        description="Fraction of checkpoints to include in each SWA model",
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducible subsampling",
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
            "Valid values: none, swa, multi_swa. "
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

    swa: SWAPluginConfig = Field(
        default_factory=SWAPluginConfig,
        description="SWA weight averaging plugin config",
    )
    multi_swa: MultiSWAPluginConfig = Field(
        default_factory=MultiSWAPluginConfig,
        description="Multi-SWA ensemble plugin config",
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
            "swa": self.swa.enabled,
            "multi_swa": self.multi_swa.enabled,
            "model_merging": self.model_merging.enabled,
            "calibration": self.calibration.enabled,
            "crc_conformal": self.crc_conformal.enabled,
            "conseco_fp_control": self.conseco_fp_control.enabled,
        }
        return [name for name, enabled in plugins.items() if enabled]
