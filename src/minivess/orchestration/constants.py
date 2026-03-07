"""Orchestration constants — canonical experiment and flow name strings.

All MLflow experiment names and Prefect @flow decorator names must be
imported from this module. Never hardcode these strings inline in flow
files — a typo causes silent cross-flow isolation failure.

Naming conventions:
  EXPERIMENT_*  — MLflow experiment name strings (snake_case, minivess_ prefix)
  FLOW_NAME_*   — Prefect @flow(name=...) strings (lowercase-hyphen)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# MLflow experiment names
# ---------------------------------------------------------------------------

EXPERIMENT_TRAINING: str = "minivess_training"
"""Primary model training experiment (DynUNet, SAM3, VesselFM variants)."""

EXPERIMENT_DATA: str = "minivess_data"
"""Data engineering experiment (volume profiling, split validation)."""

EXPERIMENT_EVALUATION: str = "minivess_evaluation"
"""Model evaluation experiment (ensemble metrics, bootstrap CIs)."""

EXPERIMENT_POST_TRAINING: str = "minivess_post_training"
"""Post-training experiment (SWA, calibration, conformal UQ)."""

EXPERIMENT_DEPLOYMENT: str = "minivess_deployment"
"""Deployment experiment (ONNX export, BentoML, promotion)."""

EXPERIMENT_DASHBOARD: str = "minivess_dashboard"
"""Dashboard and reporting experiment (figures, Parquet export)."""

EXPERIMENT_QA: str = "minivess_qa"
"""QA experiment (MLflow integrity checks, ghost run cleanup)."""

EXPERIMENT_HPO: str = "minivess_hpo"
"""Hyperparameter optimization experiment (Optuna + ASHA)."""

# ---------------------------------------------------------------------------
# Prefect @flow decorator names  (lowercase-hyphen, Prefect convention)
# ---------------------------------------------------------------------------

FLOW_NAME_TRAIN: str = "training-flow"
"""Prefect flow name for the model training flow (Flow 2)."""

FLOW_NAME_DATA: str = "data-flow"
"""Prefect flow name for the data engineering flow (Flow 1)."""

FLOW_NAME_POST_TRAINING: str = "post-training-flow"
"""Prefect flow name for the post-training flow."""

FLOW_NAME_ANALYSIS: str = "analysis-flow"
"""Prefect flow name for the model analysis flow (Flow 3)."""

FLOW_NAME_DEPLOY: str = "deploy-flow"
"""Prefect flow name for the deployment flow (Flow 4)."""

FLOW_NAME_DASHBOARD: str = "dashboard-flow"
"""Prefect flow name for the dashboard/reporting flow (Flow 5)."""

FLOW_NAME_QA: str = "qa-flow"
"""Prefect flow name for the QA flow (Flow 6)."""

FLOW_NAME_HPO: str = "hpo-flow"
"""Prefect flow name for the HPO flow."""

FLOW_NAME_ACQUISITION: str = "acquisition-flow"
"""Prefect flow name for the data acquisition flow (Flow 0)."""

FLOW_NAME_ANNOTATION: str = "annotation-flow"
"""Prefect flow name for the annotation flow."""
