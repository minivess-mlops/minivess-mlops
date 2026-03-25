from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class ModelFamily(StrEnum):
    """Supported model families (paper lineup only)."""

    MONAI_DYNUNET = "dynunet"
    VESSEL_FM = "vesselfm"
    SAM3_VANILLA = "sam3_vanilla"
    SAM3_TOPOLORA = "sam3_topolora"
    SAM3_HYBRID = "sam3_hybrid"
    MAMBAVESSELNET = "mambavesselnet"
    CUSTOM = "custom"


class EnsembleStrategy(StrEnum):
    """Ensemble combination strategies."""

    MEAN = "mean"
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED = "weighted"
    GREEDY_SOUP = "greedy_soup"
    SWAG = "swag"
    TIES_DARE = "ties_dare"
    LEARNED_STACKING = "learned_stacking"


class DataConfig(BaseModel):
    """Configuration for data loading and preprocessing."""

    dataset_name: str = Field(description="Name of the dataset (e.g., 'minivess')")
    data_dir: Path = Field(default=Path("data/raw"), description="Root data directory")
    processed_dir: Path = Field(default=Path("data/processed"))
    patch_size: tuple[int, int, int] = Field(
        default=(128, 128, 32), description="3D patch dimensions (D, H, W)"
    )
    voxel_spacing: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0),
        description="Target voxel spacing ((0,0,0) = native, no resampling)",
    )
    intensity_range: tuple[float, float] = Field(
        default=(0.0, 1.0), description="Normalized intensity range"
    )
    image_key: str = Field(
        default="image", description="Dictionary key for image volumes"
    )
    label_key: str = Field(
        default="label", description="Dictionary key for label volumes"
    )
    num_workers: int = Field(default=4, ge=0)
    pin_memory: bool = True
    prefetch_factor: int = Field(default=2, ge=1)

    # Normalization method — 'zscore' (default) or 'percentile' (for VesselFM etc.)
    normalization: str = Field(
        default="zscore",
        description="Normalization method: 'zscore' or 'percentile'",
    )
    percentile_lower: float = Field(
        default=1.0,
        ge=0.0,
        le=100.0,
        description="Lower percentile for percentile normalization",
    )
    percentile_upper: float = Field(
        default=99.0,
        ge=0.0,
        le=100.0,
        description="Upper percentile for percentile normalization",
    )

    # Disconnect-to-Connect augmentation (pre-crop topology augmentation)
    d2c_enabled: bool = Field(
        default=False,
        description="Enable DisconnectToConnect augmentation for training",
    )
    d2c_probability: float = Field(
        default=0.3, ge=0.0, le=1.0, description="D2C application probability"
    )
    d2c_mode: str = Field(
        default="zero",
        description="D2C disconnection mode: 'zero' or 'noise'",
    )
    d2c_max_segment_length: int = Field(
        default=15, ge=1, description="Max voxels to trace from junction"
    )
    d2c_max_junctions: int = Field(
        default=3, ge=1, description="Max junctions to disconnect per call"
    )
    d2c_dilation_radius: int = Field(
        default=2, ge=0, description="Dilation radius for disconnection mask"
    )

    @field_validator("data_dir", "processed_dir")
    @classmethod
    def validate_path(cls, v: Path) -> Path:
        return Path(v)

    @field_validator("normalization")
    @classmethod
    def validate_normalization(cls, v: str) -> str:
        allowed = {"zscore", "percentile"}
        if v not in allowed:
            msg = f"normalization must be one of {allowed}, got '{v}'"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_percentile_order(self) -> DataConfig:
        if self.percentile_lower >= self.percentile_upper:
            msg = (
                f"percentile_lower ({self.percentile_lower}) must be less than "
                f"percentile_upper ({self.percentile_upper})"
            )
            raise ValueError(msg)
        return self


class ModelConfig(BaseModel):
    """Configuration for model architecture."""

    family: ModelFamily
    name: str = Field(description="Human-readable model name")
    in_channels: int = Field(default=1, ge=1)
    out_channels: int = Field(
        default=2, ge=1, description="Number of classes including background"
    )
    pretrained: bool = False
    checkpoint_path: Path | None = None

    # LoRA-specific (SAMv3)
    lora_rank: int = Field(default=16, ge=1)
    lora_alpha: float = Field(default=32.0)
    lora_dropout: float = Field(default=0.1, ge=0.0, le=1.0)

    # Architecture-specific overrides (e.g., init_filters, depths, num_heads).
    # Keys depend on the model family; unknown keys are silently ignored.
    architecture_params: dict[str, Any] = Field(default_factory=dict)


class TrackedMetricConfig(BaseModel):
    """Configuration for a single tracked metric."""

    name: str
    direction: str = "minimize"  # "minimize" or "maximize"
    patience: int = Field(default=10, ge=1)


class CheckpointConfig(BaseModel):
    """Configuration for multi-metric checkpointing."""

    tracked_metrics: list[TrackedMetricConfig] = Field(
        default_factory=lambda: [
            TrackedMetricConfig(name="val_loss", direction="minimize", patience=10)
        ]
    )
    early_stopping_strategy: str = Field(default="all")
    primary_metric: str = "val_loss"
    min_delta: float = 1e-4
    min_epochs: int = 0
    save_last: bool = True
    save_history: bool = True


class ProfilingConfig(BaseModel):
    """Configuration for PyTorch profiler integration.

    STANDALONE model — NOT a field on TrainingConfig (RC4).
    Constructed from config_dict.get("profiling", {}) in train_one_fold_task()
    and passed separately to build_profiler_context().

    Profiling is core infrastructure, enabled by default. The overhead budget
    is <=10% per individual profiled epoch, <=5% amortized across a full run.
    """

    enabled: bool = Field(default=True, description="Enable torch.profiler")
    epochs: int = Field(default=5, description="Number of epochs to actively profile")
    activities: list[str] = Field(
        default_factory=lambda: ["cpu", "cuda"],
        description="Profiler activities (cpu, cuda)",
    )
    profile_memory: bool = Field(
        default=True, description="Track CUDA memory allocations"
    )
    record_shapes: bool = Field(
        default=False, description="Record tensor shapes (~5-10% overhead)"
    )
    with_flops: bool = Field(
        default=True, description="Estimate FLOPs (low overhead, high value)"
    )
    with_stack: bool = Field(
        default=False, description="Record Python call stacks (~10-15% overhead)"
    )
    export_chrome_trace: bool = Field(
        default=True, description="Export Chrome trace JSON"
    )
    compress_traces: bool = Field(
        default=True, description="Gzip compress Chrome trace files"
    )
    trace_size_limit_mb: int = Field(
        default=50,
        ge=1,
        description="Skip MLflow upload if uncompressed trace exceeds this size (MB)",
    )

    @model_validator(mode="after")
    def validate_epochs_when_enabled(self) -> ProfilingConfig:
        """When enabled=True, epochs must be >= 1."""
        if self.enabled and self.epochs < 1:
            msg = (
                f"ProfilingConfig.epochs must be >= 1 when enabled=True, "
                f"got {self.epochs}"
            )
            raise ValueError(msg)
        return self


class TrainingConfig(BaseModel):
    """Configuration for training loop."""

    max_epochs: int = Field(default=100, ge=0)  # 0 = zero-shot evaluation only
    batch_size: int = Field(default=2, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=1e-5, ge=0)
    optimizer: str = Field(default="adamw")
    scheduler: str = Field(default="cosine")
    warmup_epochs: int = Field(default=5, ge=0)
    gradient_clip_val: float = Field(default=1.0, ge=0)
    mixed_precision: bool = True
    mixed_precision_val: bool = False  # AMP OFF for validation — MONAI #4243
    gradient_checkpointing: bool = False
    seed: int = Field(default=42)
    num_folds: int = Field(default=3, ge=1)
    early_stopping_patience: int = Field(default=10, ge=0)
    val_interval: int = Field(default=1, ge=1)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)

    @field_validator("optimizer")
    @classmethod
    def validate_optimizer(cls, v: str) -> str:
        allowed = {"adam", "adamw", "sgd", "lamb"}
        if v.lower() not in allowed:
            msg = f"Optimizer must be one of {allowed}, got '{v}'"
            raise ValueError(msg)
        return v.lower()


class ServingConfig(BaseModel):
    """Configuration for model serving."""

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=3333, ge=1, le=65535)
    max_batch_size: int = Field(default=4, ge=1)
    timeout_seconds: int = Field(default=30, ge=1)
    enable_onnx: bool = False
    onnx_opset: int = Field(default=17, ge=11)


class ConformalConfig(BaseModel):
    """Configuration for conformal prediction methods."""

    alpha: float = Field(
        default=0.1,
        gt=0,
        lt=1,
        description="Significance level (e.g., 0.1 for 90% coverage)",
    )
    methods: list[str] = Field(
        default_factory=lambda: ["voxel", "morphological", "distance"],
        description="CP methods to run",
    )
    max_dilation_radius: int = Field(
        default=20, ge=1, description="Max morphological dilation radius"
    )
    calibration_fraction: float = Field(
        default=0.3, gt=0, lt=1, description="Fraction of data for calibration"
    )
    risk_functions: list[str] = Field(
        default_factory=lambda: ["dice_loss", "fnr"],
        description="Risk functions for RCPS",
    )


class EnsembleConfig(BaseModel):
    """Configuration for model ensembling."""

    strategy: EnsembleStrategy = EnsembleStrategy.MEAN
    num_members: int = Field(default=5, ge=1)
    temperature: float = Field(default=1.0, gt=0)
    conformal_alpha: float = Field(
        default=0.1, gt=0, lt=1, description="Conformal prediction significance level"
    )
    conformal: ConformalConfig = Field(default_factory=ConformalConfig)
    weightwatcher_alpha_threshold: float = Field(
        default=5.0, gt=0, description="Reject models with alpha above this"
    )


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration combining all sub-configs."""

    experiment_name: str
    run_name: str | None = None
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    tags: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""
