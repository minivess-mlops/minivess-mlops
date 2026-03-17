"""Tests for epoch-0 cost estimate wiring into train_flow (issue #717).

Verifies that after trainer.fit() returns, the training flow logs
cost estimates to MLflow and updates Prometheus gauges.
"""

from __future__ import annotations

from unittest.mock import patch


class TestEpoch0CostLoggedToMLflow:
    """After training, epoch-0 cost estimate should be logged to MLflow."""

    def test_epoch0_cost_logged_to_mlflow(self) -> None:
        """After fit() with training_time > 0 and max_epochs > 1,
        verify log_epoch0_cost_estimate logs estimated metrics."""
        from minivess.orchestration.flows.train_flow import log_epoch0_cost_estimate

        mock_estimate = {
            "est/total_cost": 1.50,
            "est/total_hours": 2.0,
            "est/cost_per_epoch": 0.03,
            "est/epoch_seconds": 120.0,
        }

        with (
            patch(
                "minivess.orchestration.flows.train_flow.estimate_cost_from_first_epoch",
                return_value=mock_estimate,
            ),
            patch("minivess.orchestration.flows.train_flow.mlflow") as mock_mlflow,
            patch(
                "minivess.orchestration.flows.train_flow.update_estimated_cost_gauges"
            ),
        ):
            log_epoch0_cost_estimate(
                training_time_seconds=600.0,
                max_epochs=10,
                num_folds=3,
                hourly_rate_usd=0.19,
            )
            mock_mlflow.log_metrics.assert_called_once()
            logged_dict = mock_mlflow.log_metrics.call_args[0][0]
            assert "est/total_cost" in logged_dict
            assert "est/total_hours" in logged_dict
            assert "est/cost_per_epoch" in logged_dict

    def test_epoch0_cost_skipped_for_single_epoch(self) -> None:
        """When max_epochs == 1, cost estimate should not be logged."""
        from minivess.orchestration.flows.train_flow import log_epoch0_cost_estimate

        with (
            patch(
                "minivess.orchestration.flows.train_flow.estimate_cost_from_first_epoch",
            ) as mock_estimate,
            patch("minivess.orchestration.flows.train_flow.mlflow") as mock_mlflow,
        ):
            log_epoch0_cost_estimate(
                training_time_seconds=60.0,
                max_epochs=1,
                num_folds=1,
                hourly_rate_usd=0.19,
            )
            mock_estimate.assert_not_called()
            mock_mlflow.log_metrics.assert_not_called()

    def test_epoch0_cost_updates_prometheus_gauges(self) -> None:
        """After epoch-0 cost estimate, Prometheus gauges should be updated."""
        from minivess.orchestration.flows.train_flow import log_epoch0_cost_estimate

        mock_estimate = {
            "est/total_cost": 2.50,
            "est/total_hours": 1.5,
            "est/cost_per_epoch": 0.05,
            "est/epoch_seconds": 120.0,
        }

        with (
            patch(
                "minivess.orchestration.flows.train_flow.estimate_cost_from_first_epoch",
                return_value=mock_estimate,
            ),
            patch("minivess.orchestration.flows.train_flow.mlflow"),
            patch(
                "minivess.orchestration.flows.train_flow.update_estimated_cost_gauges"
            ) as mock_gauges,
        ):
            log_epoch0_cost_estimate(
                training_time_seconds=600.0,
                max_epochs=5,
                num_folds=3,
                hourly_rate_usd=0.19,
            )
            mock_gauges.assert_called_once_with(mock_estimate)

    def test_epoch0_cost_logged_with_correct_values(self) -> None:
        """Given known inputs, logged values should match estimate output."""
        from minivess.observability.infrastructure_timing import (
            estimate_cost_from_first_epoch,
        )
        from minivess.orchestration.flows.train_flow import log_epoch0_cost_estimate

        # Known inputs
        epoch_seconds = 120.0
        max_epochs = 5
        num_folds = 3
        hourly_rate = 0.19

        expected = estimate_cost_from_first_epoch(
            epoch_seconds=epoch_seconds,
            max_epochs=max_epochs,
            num_folds=num_folds,
            hourly_rate_usd=hourly_rate,
        )

        with (
            patch("minivess.orchestration.flows.train_flow.mlflow") as mock_mlflow,
            patch(
                "minivess.orchestration.flows.train_flow.update_estimated_cost_gauges"
            ),
        ):
            log_epoch0_cost_estimate(
                training_time_seconds=epoch_seconds * max_epochs,
                max_epochs=max_epochs,
                num_folds=num_folds,
                hourly_rate_usd=hourly_rate,
            )
            logged_dict = mock_mlflow.log_metrics.call_args[0][0]
            assert logged_dict["est/total_cost"] == expected["est/total_cost"]
            assert logged_dict["est/total_hours"] == expected["est/total_hours"]
