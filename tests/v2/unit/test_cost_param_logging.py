"""Tests for instance type and cost param logging.

PR-E T1 (Issue #830): Extend MLflow param logging to include cloud
instance metadata from SkyPilot environment or config YAML.

TDD Phase: RED — tests written before implementation.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch


def _make_skypilot_env() -> dict[str, str]:
    """SkyPilot-style environment variables for a GCP L4 instance."""
    return {
        "SKYPILOT_TASK_ID": "task-abc123",
        "SKYPILOT_CLUSTER_NAME": "minivess-train",
        "SKYPILOT_NUM_GPUS_PER_NODE": "1",
        "SKYPILOT_ACCELERATOR_TYPE": "L4",
        "SKYPILOT_CLOUD": "gcp",
        "SKYPILOT_REGION": "europe-west4",
        "SKYPILOT_ZONE": "europe-west4-b",
        "SKYPILOT_INSTANCE_TYPE": "g2-standard-4",
        "SKYPILOT_USE_SPOT": "True",
    }


class TestCostParamsLoggedToMlflow:
    """Cost params are structured for MLflow logging."""

    def test_cost_params_logged_to_mlflow(self) -> None:
        """collect_cost_params returns dict with cost/ prefix keys."""
        from minivess.observability.cost_logging import collect_cost_params

        env = _make_skypilot_env()
        with patch.dict("os.environ", env, clear=False):
            params = collect_cost_params()

        assert "cost/instance_type" in params
        assert "cost/gpu_type" in params
        assert "cost/spot_enabled" in params
        assert "cost/region" in params
        assert "cost/provider" in params

    def test_cost_params_values(self) -> None:
        """Cost param values match the environment."""
        from minivess.observability.cost_logging import collect_cost_params

        env = _make_skypilot_env()
        with patch.dict("os.environ", env, clear=False):
            params = collect_cost_params()

        assert params["cost/instance_type"] == "g2-standard-4"
        assert params["cost/gpu_type"] == "L4"
        assert params["cost/region"] == "europe-west4"
        assert params["cost/provider"] == "gcp"


class TestCostParamsFromSkypilotEnv:
    """Cost params are read from SkyPilot environment."""

    def test_cost_params_from_skypilot_env(self) -> None:
        """All SkyPilot env vars map to cost params."""
        from minivess.observability.cost_logging import collect_cost_params

        env = _make_skypilot_env()
        with patch.dict("os.environ", env, clear=False):
            params = collect_cost_params()

        assert params["cost/spot_enabled"] == "True"
        assert params["cost/task_id"] == "task-abc123"


class TestCostParamsFromConfigYaml:
    """Cost params fall back to config when env vars are absent."""

    def test_cost_params_from_config_yaml(self) -> None:
        """collect_cost_params uses config dict when env is empty."""
        from minivess.observability.cost_logging import collect_cost_params

        config: dict[str, Any] = {
            "instance_type": "p3.2xlarge",
            "gpu_type": "V100",
            "provider": "aws",
            "region": "eu-west-1",
            "spot_enabled": True,
        }

        # No SkyPilot env vars
        empty_env: dict[str, str] = {}
        with patch.dict("os.environ", empty_env, clear=True):
            params = collect_cost_params(config_overrides=config)

        assert params["cost/instance_type"] == "p3.2xlarge"
        assert params["cost/gpu_type"] == "V100"
        assert params["cost/provider"] == "aws"


class TestCostParamsNoHardcodedRates:
    """No hardcoded cloud rates in cost params."""

    def test_cost_params_no_hardcoded_rates(self) -> None:
        """collect_cost_params never produces hardcoded dollar amounts."""
        from minivess.observability.cost_logging import collect_cost_params

        env = _make_skypilot_env()
        with patch.dict("os.environ", env, clear=False):
            params = collect_cost_params()

        # Hourly rates should not be hardcoded — they're "unknown"
        # when not provided by env or config
        rate_keys = [k for k in params if "hourly_rate" in k]
        for key in rate_keys:
            assert params[key] == "unknown" or params[key] == "0.0"


class TestCostParamsAllFieldsPresent:
    """All required cost fields are present even with empty env."""

    def test_cost_params_all_fields_present(self) -> None:
        """Missing env vars produce 'unknown', not KeyError."""
        from minivess.observability.cost_logging import collect_cost_params

        with patch.dict("os.environ", {}, clear=True):
            params = collect_cost_params()

        required = {
            "cost/instance_type",
            "cost/gpu_type",
            "cost/spot_enabled",
            "cost/region",
            "cost/provider",
            "cost/task_id",
        }
        assert required.issubset(set(params.keys()))

        # All missing values should be "unknown" or "False"
        assert params["cost/instance_type"] == "unknown"
        assert params["cost/gpu_type"] == "unknown"
        assert params["cost/provider"] == "unknown"
