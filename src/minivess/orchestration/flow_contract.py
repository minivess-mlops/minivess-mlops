"""Inter-flow communication contract via MLflow.

Flows communicate ONLY through MLflow artifacts and Prefect artifacts.
No shared filesystem, no direct function calls between containers.

This module provides the FlowContract class for discovering upstream
run results and passing context between flows.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class FlowContract:
    """Inter-flow communication contract via MLflow.

    Provides methods for flows to discover upstream results without
    direct function calls. Each flow reads/writes MLflow artifacts
    as the communication medium.

    Parameters
    ----------
    tracking_uri:
        MLflow tracking URI.
    """

    def __init__(self, tracking_uri: str = "mlruns") -> None:
        self.tracking_uri = tracking_uri

    def find_upstream_run(
        self,
        *,
        experiment_name: str,
        upstream_flow: str,
        tags: dict[str, str] | None = None,
    ) -> dict[str, Any] | None:
        """Find the most recent successful run from an upstream flow.

        Parameters
        ----------
        experiment_name:
            MLflow experiment name to search.
        upstream_flow:
            Name of the upstream flow (used as tag filter).
        tags:
            Additional tag filters.

        Returns
        -------
        Dict with ``run_id``, ``status``, ``tags`` if found, else None.
        """
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient(tracking_uri=self.tracking_uri)
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                logger.info("Experiment '%s' not found", experiment_name)
                return None

            filter_parts = ["attributes.status = 'FINISHED'"]
            if tags:
                for key, value in tags.items():
                    filter_parts.append(f"tags.{key} = '{value}'")

            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=" AND ".join(filter_parts),
                max_results=1,
                order_by=["attributes.start_time DESC"],
            )

            if not runs:
                logger.info("No upstream runs found for %s", upstream_flow)
                return None

            run = runs[0]
            return {
                "run_id": run.info.run_id,
                "status": run.info.status,
                "tags": dict(run.data.tags),
            }

        except ImportError:
            logger.warning("MLflow not available, cannot find upstream runs")
            return None
        except Exception:
            logger.warning("Failed to find upstream run", exc_info=True)
            return None

    def log_flow_completion(
        self,
        *,
        flow_name: str,
        run_id: str,
        artifacts: list[str] | None = None,
    ) -> None:
        """Log flow completion metadata for downstream flows.

        Parameters
        ----------
        flow_name:
            Name of the completing flow.
        run_id:
            MLflow run ID.
        artifacts:
            List of artifact paths produced by this flow.
        """
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient(tracking_uri=self.tracking_uri)
            client.set_tag(run_id, "flow_name", flow_name)
            client.set_tag(run_id, "flow_status", "completed")
            if artifacts:
                client.set_tag(run_id, "flow_artifacts", ",".join(artifacts))
        except Exception:
            logger.warning("Failed to log flow completion", exc_info=True)
