"""Prefect-SkyPilot bridge tasks.

Provides Prefect @task wrappers around SkyPilotLauncher for use
within Prefect flows. Enables the training flow to dispatch jobs
to SkyPilot when compute_backend='skypilot'.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from prefect import task

logger = logging.getLogger(__name__)


@task(name="launch-sky-training")
def launch_sky_training(
    config: dict[str, Any],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Prefect task: launch training job via SkyPilot.

    Parameters
    ----------
    config:
        Training configuration dict.
    dry_run:
        If True, don't actually launch.

    Returns
    -------
    Launch result dict.
    """
    from minivess.compute.skypilot_launcher import SkyPilotLauncher

    launcher = SkyPilotLauncher()
    result = launcher.launch_training_job(config, dry_run=dry_run)
    logger.info("SkyPilot launch result: %s", result.get("status"))
    return result


@task(name="wait-sky-job")
def wait_sky_job(
    job_id: str,
    *,
    poll_interval: float = 30.0,
    max_wait: float = 86400.0,
) -> dict[str, Any]:
    """Prefect task: poll SkyPilot job until completion.

    Parameters
    ----------
    job_id:
        SkyPilot job ID to monitor.
    poll_interval:
        Seconds between status checks.
    max_wait:
        Maximum wait time in seconds (default: 24 hours).

    Returns
    -------
    Final job status dict.
    """
    from minivess.compute.skypilot_launcher import SkyPilotLauncher

    launcher = SkyPilotLauncher()
    elapsed = 0.0

    while elapsed < max_wait:
        status = launcher.get_job_status(job_id)
        job_status = status.get("status", "unknown")

        if job_status in ("SUCCEEDED", "FAILED", "CANCELLED", "not_found", "unknown"):
            logger.info("SkyPilot job %s finished: %s", job_id, job_status)
            return status

        logger.info("SkyPilot job %s: %s (%.0fs elapsed)", job_id, job_status, elapsed)
        time.sleep(poll_interval)
        elapsed += poll_interval

    logger.warning("SkyPilot job %s: timeout after %.0fs", job_id, max_wait)
    return {"job_id": job_id, "status": "timeout"}
