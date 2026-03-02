"""Pipeline trigger chain — cascades flows from data to dashboard.

Manages the 5-flow pipeline execution order:
data → train → analyze → deploy → dashboard.
Core flows (1-4) stop on failure; dashboard (5) is best-effort.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable  # noqa: TC003 — used at runtime in dataclass
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


def _noop(**kwargs: Any) -> None:  # noqa: ARG001
    """Default no-op callable for unregistered flows."""


@dataclass
class FlowTriggerConfig:
    """Configuration for the pipeline trigger chain.

    Attributes
    ----------
    skip_flows:
        List of flow names to skip.
    dashboard_always:
        If True, run dashboard even when core flows fail.
    dry_run:
        If True, report planned execution without running.
    """

    skip_flows: list[str] = field(default_factory=list)
    dashboard_always: bool = True
    dry_run: bool = False


@dataclass(frozen=True)
class FlowTriggerResult:
    """Result of a single flow execution in the trigger chain.

    Attributes
    ----------
    flow_name:
        Name of the flow.
    status:
        Execution status: ``"success"``, ``"failed"``, ``"skipped"``.
    duration_s:
        Execution time in seconds.
    error:
        Error message if failed, ``None`` otherwise.
    """

    flow_name: str
    status: str
    duration_s: float
    error: str | None


@dataclass
class _FlowEntry:
    """Internal entry for a registered flow."""

    name: str
    callable: Callable[..., Any]
    is_core: bool


class PipelineTriggerChain:
    """Orchestrates the 5-flow pipeline execution order.

    Registered flows run in order. Core flow failure stops execution
    of subsequent core flows. Dashboard is best-effort.

    Usage::

        chain = PipelineTriggerChain()
        chain.register_flow("data", my_data_flow, is_core=True)
        results = chain.run_chain(trigger_source="manual")
    """

    # Default flow order
    _DEFAULT_FLOWS = ["data", "train", "analyze", "deploy", "dashboard"]

    def __init__(self) -> None:
        self._flows: dict[str, _FlowEntry] = {}
        # Initialize with default flow names (no-op callables)
        for name in self._DEFAULT_FLOWS:
            is_core = name != "dashboard"
            self._flows[name] = _FlowEntry(name=name, callable=_noop, is_core=is_core)

    @property
    def flow_names(self) -> list[str]:
        """Return ordered list of flow names."""
        return list(self._flows.keys())

    def register_flow(
        self,
        name: str,
        callable: Callable[..., Any],
        is_core: bool = True,
    ) -> None:
        """Register or replace a flow callable.

        Parameters
        ----------
        name:
            Flow name (must be in the default flow order).
        callable:
            Function to call when this flow executes.
        is_core:
            If True, failure stops the chain.
        """
        if name in self._flows:
            self._flows[name] = _FlowEntry(
                name=name, callable=callable, is_core=is_core
            )
        else:
            self._flows[name] = _FlowEntry(
                name=name, callable=callable, is_core=is_core
            )

    def run_chain(
        self,
        trigger_source: str,
        config: FlowTriggerConfig | None = None,
    ) -> list[FlowTriggerResult]:
        """Execute all registered flows in order.

        Parameters
        ----------
        trigger_source:
            What triggered this chain (e.g., ``"manual"``,
            ``"dvc_version_change"``).
        config:
            Optional trigger configuration.

        Returns
        -------
        List of results, one per flow executed.
        """
        if config is None:
            config = FlowTriggerConfig()

        results: list[FlowTriggerResult] = []
        core_failed = False

        for name, entry in self._flows.items():
            # Skip if in skip list
            if name in config.skip_flows:
                logger.info("Skipping flow '%s' (in skip_flows)", name)
                continue

            # Dashboard handling: skip if not dashboard_always and core failed
            if name == "dashboard" and core_failed and not config.dashboard_always:
                results.append(
                    FlowTriggerResult(
                        flow_name=name,
                        status="skipped",
                        duration_s=0.0,
                        error="Core flow failed, dashboard skipped",
                    )
                )
                continue

            # Skip subsequent core flows after failure
            if core_failed and entry.is_core:
                results.append(
                    FlowTriggerResult(
                        flow_name=name,
                        status="skipped",
                        duration_s=0.0,
                        error="Skipped due to prior core failure",
                    )
                )
                continue

            # Dry run — just report planned execution
            if config.dry_run:
                results.append(
                    FlowTriggerResult(
                        flow_name=name,
                        status="skipped",
                        duration_s=0.0,
                        error=None,
                    )
                )
                continue

            # Execute the flow
            start = time.monotonic()
            try:
                entry.callable(trigger_source=trigger_source)
                duration = time.monotonic() - start
                results.append(
                    FlowTriggerResult(
                        flow_name=name,
                        status="success",
                        duration_s=duration,
                        error=None,
                    )
                )
                logger.info("Flow '%s' succeeded in %.2fs", name, duration)
            except Exception as exc:
                duration = time.monotonic() - start
                results.append(
                    FlowTriggerResult(
                        flow_name=name,
                        status="failed",
                        duration_s=duration,
                        error=str(exc),
                    )
                )
                logger.warning("Flow '%s' failed in %.2fs: %s", name, duration, exc)
                if entry.is_core:
                    core_failed = True

        return results
