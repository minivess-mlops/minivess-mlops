"""Dashboard section dataclasses for the Everything Dashboard.

Each section captures one perspective of the pipeline:
data, config, model, and pipeline status.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DataDashboardSection:
    """Data engineering summary for the dashboard.

    Attributes
    ----------
    n_volumes:
        Number of primary dataset volumes.
    n_external_datasets:
        Number of external test datasets available.
    quality_gate_passed:
        Whether the data quality gate passed.
    drift_summary:
        Summary of drift analysis (drift_type → severity).
    external_datasets:
        External dataset name → pair count.
    """

    n_volumes: int
    n_external_datasets: int
    quality_gate_passed: bool
    drift_summary: dict[str, Any] = field(default_factory=dict)
    external_datasets: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConfigDashboardSection:
    """Configuration summary for the dashboard.

    Attributes
    ----------
    environment:
        Active Dynaconf environment name.
    dynaconf_params:
        Key Dynaconf settings (debug, test_markers, etc.).
    experiment_config:
        Name of the experiment config file.
    model_profile_name:
        Name of the model profile (dynunet, segresnet, etc.).
    """

    environment: str
    dynaconf_params: dict[str, Any]
    experiment_config: str
    model_profile_name: str


@dataclass(frozen=True)
class ModelDashboardSection:
    """Model summary for the dashboard.

    Attributes
    ----------
    architecture_name:
        Model architecture name (DynUNet, SegResNet, etc.).
    param_count:
        Number of trainable parameters.
    onnx_exported:
        Whether ONNX export was successful.
    champion_category:
        Champion category (balanced, topology, overlap).
    loss_name:
        Loss function used for training.
    """

    architecture_name: str
    param_count: int
    onnx_exported: bool
    champion_category: str
    loss_name: str


@dataclass(frozen=True)
class PipelineDashboardSection:
    """Pipeline status summary for the dashboard.

    Attributes
    ----------
    flow_results:
        Dict of flow_name → status string.
    last_data_version:
        Latest DVC data version tag.
    last_training_run_id:
        MLflow run ID of the most recent training.
    trigger_source:
        What triggered the pipeline (manual, dvc_version_change, etc.).
    """

    flow_results: dict[str, str]
    last_data_version: str
    last_training_run_id: str
    trigger_source: str


@dataclass(frozen=True)
class EverythingDashboard:
    """Combined dashboard with all pipeline perspectives.

    Attributes
    ----------
    data:
        Data engineering section.
    config:
        Configuration section.
    model:
        Model section.
    pipeline:
        Pipeline status section.
    generated_at:
        ISO timestamp of dashboard generation.
    """

    data: DataDashboardSection
    config: ConfigDashboardSection
    model: ModelDashboardSection
    pipeline: PipelineDashboardSection
    generated_at: str
