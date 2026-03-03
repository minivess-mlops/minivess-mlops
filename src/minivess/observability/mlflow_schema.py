"""MLflow param naming schema and validation.

Defines conventions for MLflow parameter names, prefixes,
and required parameters for training runs.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Known param prefixes and their categories
KNOWN_PARAM_PREFIXES: dict[str, str] = {
    "arch_": "architecture",
    "sys_": "system",
    "data_": "data",
    "loss_": "loss",
    "eval_": "evaluation",
    "cfg_": "configuration",
    "upstream_": "cross_flow",
    "model_": "model",
}

# Required params for a valid training run
_REQUIRED_TRAINING_PARAMS: list[str] = [
    "loss_name",
    "model_family",
    "batch_size",
    "learning_rate",
    "max_epochs",
    "seed",
]

# Default schema definition
_DEFAULT_SCHEMA: dict[str, Any] = {
    "required_params": _REQUIRED_TRAINING_PARAMS,
    "param_prefixes": KNOWN_PARAM_PREFIXES,
    "forbidden_chars": ["/", "."],
}


def load_mlflow_schema() -> dict[str, Any]:
    """Load the MLflow param schema.

    Returns the default schema. In future, this could load from
    a YAML file (configs/mlflow_schema.yaml).

    Returns
    -------
    Schema dict with ``required_params``, ``param_prefixes``,
    ``forbidden_chars``.
    """
    return _DEFAULT_SCHEMA.copy()


def validate_param_name(name: str) -> bool:
    """Validate an MLflow parameter name against naming conventions.

    Parameters
    ----------
    name:
        Parameter name to validate.

    Returns
    -------
    True if the name follows conventions, False otherwise.

    Rules:
    - No forward slashes (``/``) — causes metric naming conflicts
    - No dots (``.``) — conflicts with nested param access
    - Must use underscores for word separation
    """
    schema = load_mlflow_schema()
    return all(char not in name for char in schema["forbidden_chars"])


def categorize_param(name: str) -> str:
    """Categorize a parameter by its prefix.

    Parameters
    ----------
    name:
        Parameter name to categorize.

    Returns
    -------
    Category string (e.g., ``"architecture"``, ``"system"``, ``"training"``).
    """
    for prefix, category in KNOWN_PARAM_PREFIXES.items():
        if name.startswith(prefix):
            return category
    return "training"


def check_required_params(logged_params: dict[str, str]) -> list[str]:
    """Check which required params are missing from a run.

    Parameters
    ----------
    logged_params:
        Dict of param_name → param_value from an MLflow run.

    Returns
    -------
    List of missing required parameter names.
    """
    schema = load_mlflow_schema()
    required = schema["required_params"]
    return [p for p in required if p not in logged_params]
