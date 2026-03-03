"""SkyPilot compute provisioning wrapper.

Provides a Python SDK interface for launching training jobs and HPO
sweeps on multi-cloud GPU instances via SkyPilot.

SkyPilot handles:
- Spot instance preemption recovery
- Multi-cloud failover (AWS → GCP → RunPod → Lambda)
- On-prem K8s backend
- Cost optimization (3-6x savings with spot)

Install SkyPilot: ``uv add 'skypilot[aws,gcp,kubernetes]'``
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

SKYPILOT_AVAILABLE: bool = False

try:
    import sky  # noqa: F401

    SKYPILOT_AVAILABLE = True
except ImportError:
    SKYPILOT_AVAILABLE = False


class SkyPilotLauncher:
    """Wrapper around SkyPilot SDK for training job management.

    Provides methods for launching single training jobs, HPO sweeps,
    and checking job status. Falls back to dry-run mode when SkyPilot
    is not installed.
    """

    def __init__(self, *, cluster_name: str = "minivess") -> None:
        self.cluster_name = cluster_name
        if not SKYPILOT_AVAILABLE:
            logger.warning(
                "SkyPilot not installed. All launches will be dry-run. "
                "Install with: uv add 'skypilot[aws,gcp,kubernetes]'"
            )

    def launch_training_job(
        self,
        config: dict[str, Any],
        *,
        dry_run: bool = False,
        task_yaml: str = "deployment/skypilot/train_generic.yaml",
    ) -> dict[str, Any]:
        """Launch a single training job via SkyPilot.

        Parameters
        ----------
        config:
            Training configuration dict with loss_name, model_family, etc.
        dry_run:
            If True, report planned execution without launching.
        task_yaml:
            Path to SkyPilot task YAML file.

        Returns
        -------
        Dict with ``status``, ``job_id``, ``cluster_name``.
        """
        if dry_run or not SKYPILOT_AVAILABLE:
            logger.info(
                "[DRY RUN] Would launch training: %s on %s",
                config,
                task_yaml,
            )
            return {
                "status": "dry_run",
                "config": config,
                "task_yaml": task_yaml,
                "cluster_name": self.cluster_name,
            }

        import sky

        task = sky.Task.from_yaml(task_yaml)

        # Override envs from config
        env_overrides = {
            "LOSS_NAME": config.get("loss_name", "cbdice_cldice"),
            "MODEL_FAMILY": config.get("model_family", "dynunet"),
            "COMPUTE": config.get("compute", "gpu_low"),
        }
        task.update_envs(env_overrides)

        job_id = sky.jobs.launch(
            task, name=f"train-{config.get('loss_name', 'default')}"
        )

        logger.info(
            "Launched SkyPilot job: %s (cluster: %s)", job_id, self.cluster_name
        )
        return {
            "status": "launched",
            "job_id": str(job_id),
            "cluster_name": self.cluster_name,
        }

    def launch_hpo_sweep(
        self,
        configs: list[dict[str, Any]],
        *,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Launch parallel HPO sweep via SkyPilot managed jobs.

        Parameters
        ----------
        configs:
            List of training config dicts for each trial.
        dry_run:
            If True, report planned execution without launching.

        Returns
        -------
        Dict with ``status``, ``n_jobs``, ``job_ids``.
        """
        if dry_run or not SKYPILOT_AVAILABLE:
            logger.info("[DRY RUN] Would launch %d HPO trials", len(configs))
            return {
                "status": "dry_run",
                "n_jobs": len(configs),
                "job_ids": [],
            }

        job_ids = []
        for config in configs:
            result = self.launch_training_job(config, dry_run=False)
            if result.get("job_id"):
                job_ids.append(result["job_id"])

        return {
            "status": "launched",
            "n_jobs": len(job_ids),
            "job_ids": job_ids,
        }

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Get status of a SkyPilot managed job.

        Parameters
        ----------
        job_id:
            SkyPilot job ID.

        Returns
        -------
        Dict with ``job_id``, ``status``, ``cluster``.
        """
        if not SKYPILOT_AVAILABLE:
            return {
                "job_id": job_id,
                "status": "unknown",
                "message": "SkyPilot not installed",
            }

        try:
            import sky

            statuses = sky.jobs.queue()
            for job in statuses:
                if str(job.get("job_id")) == str(job_id):
                    return {
                        "job_id": job_id,
                        "status": job.get("status", "unknown"),
                        "cluster": job.get("cluster", "unknown"),
                    }
            return {"job_id": job_id, "status": "not_found"}
        except Exception:
            logger.warning("Failed to get job status", exc_info=True)
            return {"job_id": job_id, "status": "error"}
