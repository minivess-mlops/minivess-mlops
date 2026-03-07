"""Tests for MLflow helper utilities (T-02, closes #406).

TDD RED phase: mlflow_helpers.py must provide three helpers that
eliminate the 3-liner duplication across 6+ flow files.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestFindUpstreamSafely:
    def test_returns_none_on_exception(self) -> None:
        """find_upstream_safely() returns None when FlowContract raises."""
        from minivess.orchestration.mlflow_helpers import find_upstream_safely

        with patch(
            "minivess.orchestration.mlflow_helpers.FlowContract.find_upstream_run",
            side_effect=RuntimeError("mlflow unavailable"),
        ):
            result = find_upstream_safely(
                tracking_uri="mlruns",
                experiment_name="minivess_training",
                upstream_flow="training-flow",
            )
        assert result is None

    def test_returns_dict_on_success(self) -> None:
        """find_upstream_safely() returns the run dict on success."""
        from minivess.orchestration.mlflow_helpers import find_upstream_safely

        expected = {"run_id": "abc123", "status": "FINISHED"}
        with patch(
            "minivess.orchestration.mlflow_helpers.FlowContract.find_upstream_run",
            return_value=expected,
        ):
            result = find_upstream_safely(
                tracking_uri="mlruns",
                experiment_name="minivess_training",
                upstream_flow="training-flow",
            )
        assert result == expected

    def test_never_raises(self) -> None:
        """find_upstream_safely() must not propagate exceptions."""
        from minivess.orchestration.mlflow_helpers import find_upstream_safely

        with patch(
            "minivess.orchestration.mlflow_helpers.FlowContract.find_upstream_run",
            side_effect=ConnectionError("network error"),
        ):
            # Must not raise
            result = find_upstream_safely(
                tracking_uri="mlruns",
                experiment_name="any",
                upstream_flow="any-flow",
            )
        assert result is None


class TestLogCompletionSafe:
    def test_calls_flow_contract(self) -> None:
        """log_completion_safe() calls FlowContract.log_flow_completion()."""
        from minivess.orchestration.mlflow_helpers import log_completion_safe

        mock_contract = MagicMock()
        with patch(
            "minivess.orchestration.mlflow_helpers.FlowContract",
            return_value=mock_contract,
        ):
            log_completion_safe(
                flow_name="training-flow",
                tracking_uri="mlruns",
                run_id="abc123",
            )
        mock_contract.log_flow_completion.assert_called_once_with(
            flow_name="training-flow",
            run_id="abc123",
        )

    def test_handles_exception_gracefully(self) -> None:
        """log_completion_safe() catches FlowContract exceptions — it's best-effort."""
        from minivess.orchestration.mlflow_helpers import log_completion_safe

        mock_contract = MagicMock()
        mock_contract.log_flow_completion.side_effect = RuntimeError("mlflow down")
        with patch(
            "minivess.orchestration.mlflow_helpers.FlowContract",
            return_value=mock_contract,
        ):
            # Must not raise
            log_completion_safe(
                flow_name="training-flow",
                tracking_uri="mlruns",
                run_id="abc123",
            )

    def test_skips_when_run_id_is_none(self) -> None:
        """log_completion_safe() does nothing when run_id is None."""
        from minivess.orchestration.mlflow_helpers import log_completion_safe

        with patch("minivess.orchestration.mlflow_helpers.FlowContract") as mock_cls:
            log_completion_safe(
                flow_name="training-flow",
                tracking_uri="mlruns",
                run_id=None,
            )
        mock_cls.assert_not_called()
