"""Tests for deploy_flow config default (from_env classmethod).

TDD RED phase for T-09 (closes #408): deploy_flow() must be callable
with no arguments by adding DeployConfig.from_env() and making
the config parameter optional.
"""

from __future__ import annotations

import os

import pytest


class TestDeployConfigFromEnv:
    def test_deploy_config_from_env_reads_tracking_uri(self) -> None:
        """DeployConfig.from_env() reads MLFLOW_TRACKING_URI from environment."""
        from minivess.config.deploy_config import DeployConfig

        os.environ["MLFLOW_TRACKING_URI"] = "file:///tmp/test_mlruns"
        try:
            config = DeployConfig.from_env()
            assert str(config.mlruns_dir) == "/tmp/test_mlruns"
        finally:
            os.environ.pop("MLFLOW_TRACKING_URI", None)

    def test_deploy_config_from_env_raises_without_tracking_uri(self) -> None:
        """from_env() raises RuntimeError when MLFLOW_TRACKING_URI is not set."""
        from minivess.config.deploy_config import DeployConfig

        os.environ.pop("MLFLOW_TRACKING_URI", None)
        with pytest.raises((RuntimeError, ValueError)):
            DeployConfig.from_env()


class TestDeployFlowOptionalConfig:
    def test_deploy_flow_has_optional_config_param(self) -> None:
        """deploy_flow() signature must have config: DeployConfig | None = None."""
        import inspect

        from minivess.orchestration.deploy_flow import deploy_flow

        fn = getattr(deploy_flow, "__wrapped__", deploy_flow)
        sig = inspect.signature(fn)
        assert "config" in sig.parameters, "deploy_flow must have a 'config' parameter"
        param = sig.parameters["config"]
        assert param.default is None, (
            f"config parameter default must be None, got {param.default!r}"
        )
