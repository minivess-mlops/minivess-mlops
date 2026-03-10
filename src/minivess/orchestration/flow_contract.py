"""Inter-flow communication contract via MLflow.

Flows communicate ONLY through MLflow artifacts and Prefect artifacts.
No shared filesystem, no direct function calls between containers.

This module provides the FlowContract class for discovering upstream
run results and passing context between flows.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from minivess.observability.tracking import resolve_tracking_uri

logger = logging.getLogger(__name__)

# Sentinel written by log_flow_completion(); find_upstream_run() filters on it.
_FLOW_COMPLETE_TAG = "FLOW_COMPLETE"


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

    def __init__(self, tracking_uri: str | None = None) -> None:
        self.tracking_uri = (
            tracking_uri if tracking_uri is not None else resolve_tracking_uri()
        )
        # Read debug suffix at construction time so tests can monkeypatch the env
        # before instantiating FlowContract and get deterministic behavior.
        self._debug_suffix: str = os.environ.get("MINIVESS_DEBUG_SUFFIX", "")

    # ------------------------------------------------------------------
    # Experiment name helpers
    # ------------------------------------------------------------------

    def _resolve_experiment(self, base_name: str) -> str:
        """Return *base_name* with the debug suffix appended (may be empty)."""
        return f"{base_name}{self._debug_suffix}"

    # ------------------------------------------------------------------
    # Upstream run discovery
    # ------------------------------------------------------------------

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
            # Filter by flow_name tag so we find the correct upstream flow
            # (e.g. "training-flow" not "post-training-flow") (#586).
            filter_parts.append(f"tags.flow_name = '{upstream_flow}'")
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

    # ------------------------------------------------------------------
    # Fold checkpoint discovery
    # ------------------------------------------------------------------

    def find_fold_checkpoints(
        self,
        *,
        parent_run_id: str,
    ) -> list[dict[str, Any]]:
        """Return fold checkpoint info from a parent training run's MLflow tags.

        Reads tags of the form ``checkpoint_dir_fold_N`` from *parent_run_id*
        and returns one entry per fold found.

        Parameters
        ----------
        parent_run_id:
            MLflow run ID of the upstream training run.

        Returns
        -------
        List of dicts, each with:
        - ``fold_id`` (int)
        - ``run_id`` (str) — same as *parent_run_id* for now
        - ``checkpoint_dir`` (Path)
        """
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient(tracking_uri=self.tracking_uri)
            run_data = client.get_run(parent_run_id)
            tags = run_data.data.tags
        except Exception:
            logger.warning(
                "Failed to read tags for run %s", parent_run_id, exc_info=True
            )
            return []

        prefix = "checkpoint_dir_fold_"
        results: list[dict[str, Any]] = []
        for tag_key, tag_value in tags.items():
            if not tag_key.startswith(prefix):
                continue
            fold_str = tag_key[len(prefix) :]
            try:
                fold_id = int(fold_str)
            except ValueError:
                logger.warning("Unexpected checkpoint tag key: %s", tag_key)
                continue
            results.append(
                {
                    "fold_id": fold_id,
                    "run_id": parent_run_id,
                    "checkpoint_dir": Path(tag_value),
                }
            )

        results.sort(key=lambda r: r["fold_id"])
        return results

    # ------------------------------------------------------------------
    # Flow completion logging
    # ------------------------------------------------------------------

    def log_flow_completion(
        self,
        *,
        flow_name: str,
        run_id: str,
        artifacts: list[str] | None = None,
        checkpoint_dir: Path | None = None,
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
        checkpoint_dir:
            Optional top-level checkpoint directory (tagged for post-training
            discovery).
        """
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient(tracking_uri=self.tracking_uri)
            client.set_tag(run_id, "flow_name", flow_name)
            client.set_tag(run_id, "flow_status", _FLOW_COMPLETE_TAG)
            if artifacts:
                client.set_tag(run_id, "flow_artifacts", ",".join(artifacts))
            if checkpoint_dir is not None:
                client.set_tag(run_id, "checkpoint_dir", str(checkpoint_dir))
        except Exception:
            logger.warning("Failed to log flow completion", exc_info=True)
