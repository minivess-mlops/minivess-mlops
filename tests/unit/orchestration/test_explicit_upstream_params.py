"""Tests for explicit upstream run ID parameters in all consumer flows (T-14, closes #412, #418).

TDD RED phase: all consumer flows must accept an optional upstream_X_run_id
parameter that bypasses auto-discovery when provided.
"""

from __future__ import annotations

import inspect
import typing


def _get_param_names(fn: typing.Callable) -> set[str]:
    """Return parameter names of a callable."""
    return set(inspect.signature(fn).parameters)


class TestTrainingFlowUpstreamParam:
    def test_training_flow_accepts_upstream_data_run_id(self) -> None:
        """training_flow() accepts optional upstream_data_run_id parameter."""
        from minivess.orchestration.flows.train_flow import training_flow

        params = _get_param_names(training_flow)
        assert "upstream_data_run_id" in params, (
            "training_flow() missing upstream_data_run_id parameter"
        )

    def test_upstream_data_run_id_default_is_none(self) -> None:
        """upstream_data_run_id default must be None (auto-discover mode)."""
        from minivess.orchestration.flows.train_flow import training_flow

        sig = inspect.signature(training_flow)
        param = sig.parameters.get("upstream_data_run_id")
        assert param is not None
        assert param.default is None


class TestAnalysisFlowUpstreamParam:
    def test_analysis_flow_accepts_upstream_training_run_id(self) -> None:
        """run_analysis_flow() accepts optional upstream_training_run_id parameter."""
        from minivess.orchestration.flows.analysis_flow import run_analysis_flow

        params = _get_param_names(run_analysis_flow)
        assert "upstream_training_run_id" in params, (
            "run_analysis_flow() missing upstream_training_run_id parameter"
        )

    def test_upstream_training_run_id_default_is_none(self) -> None:
        """upstream_training_run_id default must be None."""
        from minivess.orchestration.flows.analysis_flow import run_analysis_flow

        sig = inspect.signature(run_analysis_flow)
        param = sig.parameters.get("upstream_training_run_id")
        assert param is not None
        assert param.default is None


class TestDeployFlowUpstreamParam:
    def test_deploy_flow_accepts_upstream_analysis_run_id(self) -> None:
        """deploy_flow() accepts optional upstream_analysis_run_id parameter."""
        from minivess.orchestration.flows.deploy_flow import deploy_flow

        params = _get_param_names(deploy_flow)
        assert "upstream_analysis_run_id" in params, (
            "deploy_flow() missing upstream_analysis_run_id parameter"
        )

    def test_upstream_analysis_run_id_default_is_none(self) -> None:
        """upstream_analysis_run_id default must be None."""
        from minivess.orchestration.flows.deploy_flow import deploy_flow

        sig = inspect.signature(deploy_flow)
        param = sig.parameters.get("upstream_analysis_run_id")
        assert param is not None
        assert param.default is None
