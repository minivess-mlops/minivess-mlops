"""Staleness scanner — identifies outdated knowledge artifacts.

Usage:
  uv run python scripts/review_staleness.py
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
KG_DIR = REPO_ROOT / "knowledge-graph"
DECISIONS_DIR = KG_DIR / "decisions"
PLANNING_DIR = REPO_ROOT / "docs" / "planning"


def _get_last_modified(file_path: Path) -> datetime | None:
    """Get last modification date from git log."""
    result = subprocess.run(
        ["git", "log", "-1", "--format=%aI", "--", str(file_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        return datetime.fromisoformat(result.stdout.strip())
    except ValueError:
        return None


def _check_volatile_nodes_reviewed() -> list[dict]:
    """Check 4: PRD volatile nodes reviewed within review date."""
    checks = []
    now = datetime.now(UTC)

    if not DECISIONS_DIR.exists():
        return []

    for yaml_file in DECISIONS_DIR.rglob("*.yaml"):
        data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
        if not data:
            continue
        volatility = data.get("volatility", {})
        classification = volatility.get("classification", "")
        next_review = volatility.get("next_review", "")

        if not next_review or classification not in ("volatile", "evolving"):
            continue

        try:
            review_date = datetime.strptime(str(next_review), "%Y-%m-%d").replace(
                tzinfo=UTC
            )
        except ValueError:
            checks.append(
                {
                    "check": f"review_date:{data.get('decision_id', yaml_file.stem)}",
                    "severity": "WARN",
                    "ok": False,
                    "message": f"Unparseable next_review '{next_review}' in {yaml_file.name}",
                }
            )
            continue

        overdue = now > review_date
        if overdue:
            days_overdue = (now - review_date).days
            checks.append(
                {
                    "check": f"overdue:{data.get('decision_id', yaml_file.stem)}",
                    "severity": "WARN",
                    "ok": False,
                    "message": f"OVERDUE by {days_overdue} days: {data.get('decision_id')} ({classification}, due {next_review})",
                }
            )

    if not checks:
        checks.append(
            {
                "check": "volatile_reviews",
                "severity": "WARN",
                "ok": True,
                "message": "All volatile/evolving decisions within review window",
            }
        )
    return checks


def _check_domain_navigators_fresh() -> list[dict]:
    """Check 6: Domain navigators reviewed within 30 days."""
    checks = []
    now = datetime.now(UTC)
    domains_dir = KG_DIR / "domains"
    if not domains_dir.exists():
        return []

    for yaml_file in sorted(domains_dir.glob("*.yaml")):
        data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
        if not data:
            continue
        date_str = data.get("last_reviewed", "")
        if not date_str:
            checks.append(
                {
                    "check": f"domain_fresh:{yaml_file.name}",
                    "severity": "WARN",
                    "ok": False,
                    "message": f"No last_reviewed in {yaml_file.name}",
                }
            )
            continue
        try:
            reviewed = datetime.strptime(str(date_str), "%Y-%m-%d").replace(tzinfo=UTC)
            days_since = (now - reviewed).days
            if days_since > 30:
                checks.append(
                    {
                        "check": f"domain_fresh:{yaml_file.name}",
                        "severity": "WARN",
                        "ok": False,
                        "message": f"STALE: {yaml_file.name} last reviewed {days_since} days ago ({date_str})",
                    }
                )
        except ValueError:
            checks.append(
                {
                    "check": f"domain_fresh:{yaml_file.name}",
                    "severity": "WARN",
                    "ok": False,
                    "message": f"Unparseable date '{date_str}' in {yaml_file.name}",
                }
            )

    if not checks:
        checks.append(
            {
                "check": "domain_freshness",
                "severity": "WARN",
                "ok": True,
                "message": "All domain navigators reviewed within 30 days",
            }
        )
    return checks


def _check_planning_docs_have_frontmatter() -> list[dict]:
    """Check 8: Count planning docs without YAML frontmatter."""
    if not PLANNING_DIR.exists():
        return []

    total = 0
    with_fm = 0
    without_fm = []

    for md_file in sorted(PLANNING_DIR.glob("*.md")):
        total += 1
        content = md_file.read_text(encoding="utf-8")
        if content.startswith("---"):
            with_fm += 1
        else:
            without_fm.append(md_file.name)

    ok = len(without_fm) == 0
    return [
        {
            "check": "planning_frontmatter",
            "severity": "WARN",
            "ok": ok,
            "message": f"{with_fm}/{total} planning docs have frontmatter"
            + (f" ({len(without_fm)} missing)" if without_fm else ""),
        }
    ]


def _check_scenario_freshness() -> list[dict]:
    """Check: Active scenario file is current with implementation."""
    checks = []
    scenario_path = KG_DIR / "scenarios" / "learning-first-mvp.yaml"
    if not scenario_path.exists():
        return [
            {
                "check": "scenario_exists",
                "severity": "WARN",
                "ok": False,
                "message": "Active scenario learning-first-mvp.yaml not found",
            }
        ]

    last_mod = _get_last_modified(scenario_path)
    if last_mod:
        days_since = (datetime.now(UTC) - last_mod).days
        ok = days_since <= 30
        checks.append(
            {
                "check": "scenario_freshness",
                "severity": "WARN",
                "ok": ok,
                "message": f"Active scenario last modified {days_since} days ago"
                + (" (STALE — review needed)" if not ok else ""),
            }
        )
    return checks


def main() -> dict:
    """Run all staleness checks."""
    all_checks: list[dict] = []
    all_checks.extend(_check_volatile_nodes_reviewed())
    all_checks.extend(_check_domain_navigators_fresh())
    all_checks.extend(_check_planning_docs_have_frontmatter())
    all_checks.extend(_check_scenario_freshness())

    failures = sum(1 for c in all_checks if not c["ok"] and c["severity"] == "ERROR")
    warnings = sum(1 for c in all_checks if not c["ok"] and c["severity"] == "WARN")

    return {
        "agent_name": "staleness_scanner",
        "failures": failures,
        "warnings": warnings,
        "total_checks": len(all_checks),
        "checks": all_checks,
    }


if __name__ == "__main__":
    result = main()
    print(f"\n{'=' * 60}")
    print("Staleness Scanner")
    print(f"{'=' * 60}")
    print(f"Total checks: {result['total_checks']}")
    print(f"Failures (ERROR): {result['failures']}")
    print(f"Warnings (WARN):  {result['warnings']}")
    for check in result["checks"]:
        if not check["ok"]:
            print(f"  [{check['severity']}] {check['message']}")
    if result["failures"] > 0:
        sys.exit(1)
    else:
        print("\nStaleness scan complete!")
        sys.exit(0)
