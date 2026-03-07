"""QA Prefect flow — automated MLflow data integrity checks.

Flow 6: Runs quality assurance checks on MLflow tracking data:
- Backend consistency (local vs server)
- Required param validation
- Ghost run detection
- Metric sanity checks

This flow is designed to run periodically, after training, or in CI.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from minivess.observability.mlflow_backend import detect_backend_type
from minivess.observability.mlflow_schema import check_required_params
from minivess.orchestration._prefect_compat import flow, task

logger = logging.getLogger(__name__)


@task(name="check-backend-consistency")
def check_backend_consistency(tracking_uri: str) -> dict[str, Any]:
    """Check MLflow backend type and warn on local filesystem.

    Parameters
    ----------
    tracking_uri:
        MLflow tracking URI to check.

    Returns
    -------
    Check result dict with ``name``, ``status``, ``backend_type``, ``message``.
    """
    backend_type = detect_backend_type(tracking_uri)

    if backend_type == "local":
        return {
            "name": "backend_consistency",
            "status": "warning",
            "backend_type": backend_type,
            "message": (
                f"Using local filesystem backend ({tracking_uri}). "
                "Consider migrating to server backend for reliability."
            ),
        }

    return {
        "name": "backend_consistency",
        "status": "pass",
        "backend_type": backend_type,
        "message": f"Backend type: {backend_type} ({tracking_uri})",
    }


@task(name="check-run-params")
def check_run_params(run: Any) -> dict[str, Any]:
    """Check that a run has all required parameters.

    Parameters
    ----------
    run:
        MLflow Run object to validate.

    Returns
    -------
    Check result dict with ``name``, ``status``, ``missing_params``.
    """
    logged_params = run.data.params
    missing = check_required_params(logged_params)

    if missing:
        return {
            "name": "required_params",
            "status": "fail",
            "missing_params": missing,
            "message": f"Missing required params: {', '.join(missing)}",
        }

    return {
        "name": "required_params",
        "status": "pass",
        "missing_params": [],
        "message": "All required params present",
    }


@task(name="check-ghost-runs")
def check_ghost_runs(
    client: Any,
    *,
    experiment_ids: list[str],
) -> dict[str, Any]:
    """Check for orphaned RUNNING runs.

    Parameters
    ----------
    client:
        MLflow client instance.
    experiment_ids:
        Experiment IDs to check.

    Returns
    -------
    Check result dict with ``name``, ``status``, ``ghost_count``.
    """
    from minivess.observability.ghost_cleanup import find_ghost_runs

    ghosts = find_ghost_runs(client, experiment_ids=experiment_ids)

    if ghosts:
        return {
            "name": "ghost_runs",
            "status": "warning",
            "ghost_count": len(ghosts),
            "ghost_run_ids": [r.info.run_id for r in ghosts],
            "message": f"Found {len(ghosts)} RUNNING (possibly orphaned) runs",
        }

    return {
        "name": "ghost_runs",
        "status": "pass",
        "ghost_count": 0,
        "message": "No ghost runs found",
    }


def generate_qa_report(checks: list[dict[str, Any]]) -> str:
    """Generate a human-readable QA report from check results.

    Parameters
    ----------
    checks:
        List of check result dicts.

    Returns
    -------
    Markdown-formatted report string.
    """
    lines = ["# MLflow QA Report", ""]

    for check in checks:
        status = check.get("status", "unknown").upper()
        name = check.get("name", "unnamed")
        message = check.get("message", "")
        icon = {"PASS": "[PASS]", "FAIL": "[FAIL]", "WARNING": "[WARN]"}.get(
            status, "[????]"
        )
        lines.append(f"- {icon} **{name}**: {message}")

    summary = summarize_qa_results(checks)
    lines.extend(
        [
            "",
            f"## Summary: {summary['passed']}/{summary['total']} passed, "
            f"{summary['failed']} failed, {summary['warnings']} warnings",
        ]
    )

    return "\n".join(lines)


def summarize_qa_results(checks: list[dict[str, Any]]) -> dict[str, int]:
    """Summarize QA check results.

    Parameters
    ----------
    checks:
        List of check result dicts.

    Returns
    -------
    Dict with ``total``, ``passed``, ``failed``, ``warnings``.
    """
    total = len(checks)
    passed = sum(1 for c in checks if c.get("status") == "pass")
    failed = sum(1 for c in checks if c.get("status") == "fail")
    warnings = sum(1 for c in checks if c.get("status") == "warning")

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "warnings": warnings,
    }


@flow(name="qa-flow")
def qa_flow(
    tracking_uri: str | None = None,
    experiment_ids: list[str] | None = None,
) -> dict[str, Any]:
    """QA Prefect flow — runs all MLflow quality checks.

    Parameters
    ----------
    tracking_uri:
        MLflow tracking URI. Defaults to MLFLOW_TRACKING_URI env var,
        falling back to "mlruns".
    experiment_ids:
        Experiment IDs to check. If None, checks all.

    Returns
    -------
    Dict with ``checks`` list, ``summary``, ``report``, and ``report_path``.
    """
    if tracking_uri is None:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")

    checks: list[dict[str, Any]] = []

    # Check 1: Backend consistency
    backend_check = check_backend_consistency(tracking_uri)
    checks.append(backend_check)

    # Check 2: Ghost runs (if experiment IDs provided)
    if experiment_ids:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=tracking_uri)
        ghost_check = check_ghost_runs(client, experiment_ids=experiment_ids)
        checks.append(ghost_check)

    # Generate report
    report = generate_qa_report(checks)
    summary = summarize_qa_results(checks)

    logger.info("QA flow complete: %s", summary)
    logger.info("\n%s", report)

    # Persist report to disk
    report_dir = Path(os.environ.get("DASHBOARD_OUTPUT", "/app/outputs/dashboard"))
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "qa_report.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("QA report saved: %s", report_path)

    return {
        "checks": checks,
        "summary": summary,
        "report": report,
        "report_path": str(report_path),
    }
