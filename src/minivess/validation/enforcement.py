"""Configurable gate enforcement for data quality pipeline.

Reads gate severity from Dynaconf settings.toml and dispatches:
- 'error'   -> halt pipeline (raise DataQualityError)
- 'warning' -> tag MLflow + continue
- 'info'    -> log only, never halt

Escape hatch: MINIVESS_SKIP_QUALITY_GATE=1 bypasses ALL gates (pytest only).
"""

from __future__ import annotations

import logging
import os
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minivess.validation.gates import GateResult

logger = logging.getLogger(__name__)


class GateAction(StrEnum):
    """Action taken after evaluating a quality gate."""

    PASS = "pass"
    WARN = "warn"
    LOG = "log"
    HALT = "halt"
    SKIP = "skip"


class DataQualityError(RuntimeError):
    """Raised when a quality gate with severity='error' fails."""


def get_gate_severity(gate_name: str) -> str:
    """Read gate severity from Dynaconf settings.

    Looks up ``data_quality_gate_{gate_name}`` in settings.toml.
    Falls back to ``'warning'`` for unknown gate names.

    Parameters
    ----------
    gate_name:
        Short name of the gate (e.g., 'pandera', 'ge', 'datacare', 'deepchecks').

    Returns
    -------
    Severity string: 'error', 'warning', or 'info'.
    """
    try:
        from minivess.config.settings import get_settings

        settings = get_settings()
        key = f"data_quality_gate_{gate_name}"
        value = settings.get(key, "warning")
        if isinstance(value, str) and value in ("error", "warning", "info"):
            return value
    except ImportError:
        logger.debug("Dynaconf not installed — using default severity 'warning'")
    except Exception:
        logger.debug(
            "Failed to read Dynaconf setting for gate %s", gate_name, exc_info=True
        )

    return "warning"


def enforce_gate(
    gate_name: str,
    result: GateResult,
    severity: str | None = None,
) -> GateAction:
    """Enforce a quality gate based on its result and configured severity.

    Parameters
    ----------
    gate_name:
        Short name of the gate (e.g., 'pandera', 'ge').
    result:
        GateResult from the validation gate.
    severity:
        Override severity. When None, reads from Dynaconf settings.

    Returns
    -------
    GateAction indicating what happened.

    Raises
    ------
    DataQualityError:
        When severity is 'error' and the gate failed.
    """
    # Escape hatch: skip all gates when env var is set
    if os.environ.get("MINIVESS_SKIP_QUALITY_GATE") == "1":
        logger.info("MINIVESS_SKIP_QUALITY_GATE=1 — skipping gate '%s'", gate_name)
        return GateAction.SKIP

    # Gate passed — nothing to enforce
    if result.passed:
        logger.info("Gate '%s' PASSED", gate_name)
        return GateAction.PASS

    # Resolve severity
    resolved_severity = (
        severity if severity is not None else get_gate_severity(gate_name)
    )

    if resolved_severity == "error":
        msg = f"Gate '{gate_name}' FAILED (severity=error): {result.errors}"
        logger.error(msg)
        raise DataQualityError(msg)

    if resolved_severity == "warning":
        logger.warning(
            "Gate '%s' FAILED (severity=warning): %s", gate_name, result.errors
        )
        return GateAction.WARN

    # severity == 'info' or anything else
    logger.info("Gate '%s' FAILED (severity=info): %s", gate_name, result.errors)
    return GateAction.LOG
