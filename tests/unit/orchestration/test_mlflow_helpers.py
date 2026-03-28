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


class TestEmitLineageSafe:
    def test_never_raises_on_import_error(self) -> None:
        """emit_lineage_safe() must catch import errors."""
        from minivess.orchestration.mlflow_helpers import emit_lineage_safe

        with (
            patch(
                "minivess.orchestration.mlflow_helpers.emit_lineage_safe.__module__",
            ),
            # Patch the import to fail
            patch.dict("sys.modules", {"minivess.observability.lineage": None}),
        ):
            # Must not raise
            emit_lineage_safe(job_name="test-flow")

    def test_never_raises_on_exception(self) -> None:
        """emit_lineage_safe() catches all exceptions."""
        from minivess.orchestration.mlflow_helpers import emit_lineage_safe

        with patch(
            "minivess.observability.lineage.LineageEmitter",
            side_effect=RuntimeError("lineage down"),
        ):
            emit_lineage_safe(job_name="test-flow")


class TestStartMlflowRunSafe:
    def test_returns_none_on_exception(self) -> None:
        """start_mlflow_run_safe() returns None when MLflow errors."""
        from minivess.orchestration.mlflow_helpers import start_mlflow_run_safe

        with patch(
            "minivess.observability.tracking.resolve_tracking_uri",
            side_effect=RuntimeError("no MLflow"),
        ):
            result = start_mlflow_run_safe(experiment_name="test_exp")
        assert result is None

    def test_returns_run_id_on_success(self) -> None:
        """start_mlflow_run_safe() returns run ID on success."""
        from minivess.orchestration.mlflow_helpers import start_mlflow_run_safe

        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"

        with (
            patch(
                "minivess.observability.tracking.resolve_tracking_uri",
                return_value="file:///tmp/mlruns",
            ),
            patch("mlflow.set_tracking_uri"),
            patch("mlflow.set_experiment"),
            patch(
                "mlflow.start_run",
                return_value=MagicMock(
                    __enter__=MagicMock(return_value=mock_run),
                    __exit__=MagicMock(return_value=False),
                ),
            ),
            patch("mlflow.log_metrics"),
        ):
            result = start_mlflow_run_safe(
                experiment_name="test_exp",
                metrics={"accuracy": 0.95},
            )
        assert result == "test_run_123"
