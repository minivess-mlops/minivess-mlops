"""Knowledge reviewer orchestrator — runs all reviewer agents.

Usage:
  uv run python scripts/review_knowledge.py                # full review
  uv run python scripts/review_knowledge.py --quick         # link check + legacy only
  uv run python scripts/review_knowledge.py --prd           # PRD auditor only
  uv run python scripts/review_knowledge.py --staleness     # staleness scan only
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

# Add scripts/ to sys.path so we can import reviewers
sys.path.insert(0, str(Path(__file__).resolve().parent))

from review_knowledge_links import main as link_check
from review_legacy_artifacts import main as legacy_check
from review_prd_integrity import main as prd_audit
from review_staleness import main as staleness_scan


def run_all_reviewers(mode: str = "full") -> dict:
    """Run reviewer agents based on mode."""
    reviewers = {
        "full": [
            ("link_checker", lambda: link_check(quick=False)),
            ("legacy_detector", legacy_check),
            ("prd_auditor", prd_audit),
            ("staleness_scanner", staleness_scan),
        ],
        "quick": [
            ("link_checker", lambda: link_check(quick=True)),
            ("legacy_detector", legacy_check),
        ],
        "prd": [
            ("prd_auditor", prd_audit),
        ],
        "staleness": [
            ("staleness_scanner", staleness_scan),
        ],
    }

    agents = reviewers.get(mode, reviewers["full"])
    results = []

    for name, fn in agents:
        try:
            result = fn()
            results.append(result)
        except Exception as e:
            results.append(
                {
                    "agent_name": name,
                    "failures": 1,
                    "warnings": 0,
                    "total_checks": 1,
                    "checks": [
                        {
                            "check": "agent_error",
                            "severity": "ERROR",
                            "ok": False,
                            "message": f"Agent {name} crashed: {e}",
                        }
                    ],
                }
            )

    report = {
        "mode": mode,
        "timestamp": datetime.now(UTC).isoformat(),
        "agents_run": len(results),
        "total_failures": sum(r.get("failures", 0) for r in results),
        "total_warnings": sum(r.get("warnings", 0) for r in results),
        "total_checks": sum(r.get("total_checks", 0) for r in results),
        "agent_reports": results,
    }
    return report


def main() -> None:
    """CLI entry point."""
    mode = "full"
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            mode = arg.lstrip("-")
            break

    report = run_all_reviewers(mode)

    print(f"\n{'=' * 60}")
    print(f"Knowledge Graph Review ({mode} mode)")
    print(f"{'=' * 60}")
    print(f"Agents run:      {report['agents_run']}")
    print(f"Total checks:    {report['total_checks']}")
    print(f"Failures (ERROR): {report['total_failures']}")
    print(f"Warnings (WARN):  {report['total_warnings']}")
    print(f"{'=' * 60}")

    for agent_report in report["agent_reports"]:
        agent_name = agent_report.get("agent_name", "unknown")
        f = agent_report.get("failures", 0)
        w = agent_report.get("warnings", 0)
        t = agent_report.get("total_checks", 0)
        status = "PASS" if f == 0 else "FAIL"
        print(f"\n  [{status}] {agent_name}: {t} checks, {f} failures, {w} warnings")
        for check in agent_report.get("checks", []):
            if not check["ok"]:
                print(f"    [{check['severity']}] {check['message']}")
            elif check.get("severity") == "INFO":
                print(f"    [INFO] {check['message']}")

    print(f"\n{'=' * 60}")
    if report["total_failures"] > 0:
        print(f"RESULT: {report['total_failures']} ERROR(s) found — fix required")
        sys.exit(1)
    elif report["total_warnings"] > 0:
        print(f"RESULT: {report['total_warnings']} WARNING(s) — review recommended")
        sys.exit(0)
    else:
        print("RESULT: All checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
