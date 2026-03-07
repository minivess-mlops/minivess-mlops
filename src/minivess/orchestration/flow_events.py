"""Flow event helpers — dashboard refresh triggers and inter-flow signaling.

When any flow completes, it can trigger a dashboard refresh via Prefect's
run_deployment() API. This is best-effort — trigger failure never blocks
the triggering flow.
"""

from __future__ import annotations

import logging

from minivess.orchestration.constants import FLOW_NAME_DASHBOARD

logger = logging.getLogger(__name__)


def trigger_dashboard_refresh(trigger_source: str = "flow_completion") -> None:
    """Trigger dashboard flow refresh after any flow completion.

    Best-effort: failure is logged at DEBUG level and never raises.

    Parameters
    ----------
    trigger_source:
        Descriptive source of the trigger (for logging).
    """
    try:
        from prefect.deployments import run_deployment

        run_deployment(
            name=f"{FLOW_NAME_DASHBOARD}/default",
            parameters={"trigger_source": trigger_source},
            timeout=0,  # Fire-and-forget
        )
        logger.info("Dashboard refresh triggered (source: %s)", trigger_source)
    except Exception:
        logger.debug(
            "Dashboard trigger failed (non-fatal, source: %s)",
            trigger_source,
            exc_info=True,
        )
