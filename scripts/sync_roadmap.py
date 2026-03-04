"""Sync GitHub Project Roadmap timeline fields.

Sets Start date, Target date, Size, and Estimate on project items
that are missing them. Works in three modes:

  --mode backfill   Scan ALL items, set any missing fields
  --mode recent     Scan issues closed in last N days (default 7)
  --issue NUM       Sync a single issue by number

Requires: gh CLI authenticated with project write scope.
No pip dependencies — stdlib only.

Usage:
    python3 scripts/sync_roadmap.py --mode backfill
    python3 scripts/sync_roadmap.py --mode recent --days 7
    python3 scripts/sync_roadmap.py --issue 343
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta

# ── Project field IDs ──────────────────────────────────────────────
PROJECT_ID = "PVT_kwDOCPpnGc4AYSAM"
REPO = "minivess-mlops/minivess-mlops"

FIELD_START_DATE = "PVTF_lADOCPpnGc4AYSAMzgPhgso"
FIELD_TARGET_DATE = "PVTF_lADOCPpnGc4AYSAMzg-z7gU"
FIELD_SIZE = "PVTSSF_lADOCPpnGc4AYSAMzg-zBm8"
FIELD_ESTIMATE = "PVTF_lADOCPpnGc4AYSAMzg-zBns"
FIELD_STATUS = "PVTSSF_lADOCPpnGc4AYSAMzgPhgrw"
FIELD_PRIORITY = "PVTSSF_lADOCPpnGc4AYSAMzgPhgsk"
FIELD_ITERATION = "PVTIF_lADOCPpnGc4AYSAMzg-zB-I"

STATUS_DONE = "c82e9af3"
STATUS_READY = "56933ea5"
STATUS_BACKLOG = "14dd5c1c"

PRIO_MAP = {
    "P0-critical": "c128192a",
    "P1-high": "b419b06f",
    "P2-medium": "5bb35602",
    "P3-low": "4b1f5dd3",
}

SIZE_OPTIONS = {
    "XS": "62e469d2",
    "S": "7e3dee78",
    "M": "184d9e87",
    "L": "9252603f",
    "XL": "763b524a",
}

SIZE_ESTIMATE = {"XS": 1, "S": 2, "M": 3, "L": 5, "XL": 8}


# ── Helpers ────────────────────────────────────────────────────────
def _run(cmd: str) -> tuple[str, int]:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout.strip(), r.returncode


def _graphql(query: str) -> dict | None:
    out, rc = _run(f"gh api graphql -f query='{query}'")
    if rc != 0:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


def _set_date(item_id: str, field_id: str, date_str: str) -> bool:
    q = f'''mutation {{
      updateProjectV2ItemFieldValue(input: {{
        projectId: "{PROJECT_ID}"
        itemId: "{item_id}"
        fieldId: "{field_id}"
        value: {{ date: "{date_str}" }}
      }}) {{ projectV2Item {{ id }} }}
    }}'''
    return _graphql(q) is not None


def _set_single_select(item_id: str, field_id: str, option_id: str) -> bool:
    q = f'''mutation {{
      updateProjectV2ItemFieldValue(input: {{
        projectId: "{PROJECT_ID}"
        itemId: "{item_id}"
        fieldId: "{field_id}"
        value: {{ singleSelectOptionId: "{option_id}" }}
      }}) {{ projectV2Item {{ id }} }}
    }}'''
    return _graphql(q) is not None


def _set_number(item_id: str, field_id: str, value: float) -> bool:
    q = f'''mutation {{
      updateProjectV2ItemFieldValue(input: {{
        projectId: "{PROJECT_ID}"
        itemId: "{item_id}"
        fieldId: "{field_id}"
        value: {{ number: {value} }}
      }}) {{ projectV2Item {{ id }} }}
    }}'''
    return _graphql(q) is not None


# ── Size heuristic ─────────────────────────────────────────────────
def estimate_size(num: int, title: str, labels: list[str]) -> str:
    """Estimate issue size from title and labels."""
    title_lower = title.lower()

    if "bug" in labels:
        return "XS"
    if "v0.1-legacy" in labels:
        return "S"
    # Task sub-items
    task_markers = [
        "task ",
        "sdd ",
        "t1:",
        "t2:",
        "t3:",
        "t4:",
        "t5:",
        "sam-",
        "deploy task",
    ]
    if any(m in title_lower for m in task_markers):
        return "S"
    # Original PRD features
    if 3 <= num <= 22:
        return "L"
    # Scaffold phases
    if 23 <= num <= 32:
        return "M"
    if "research" in labels:
        return "M"
    return "S"


# ── Data fetching ──────────────────────────────────────────────────
def get_repo_issues(
    numbers: list[int] | None = None,
    *,
    since_days: int | None = None,
) -> dict[int, dict]:
    """Fetch issues from the repo. Optionally filter by number or recency."""
    out, _ = _run(
        f"gh issue list --repo {REPO} --state all --limit 500 "
        "--json number,title,state,createdAt,closedAt,labels "
        "-q '[.[] | {n:.number, t:.title, s:.state, c:.createdAt, "
        "cl:.closedAt, l:[.labels[].name]}]'"
    )
    all_issues = {i["n"]: i for i in json.loads(out)}

    if numbers:
        return {n: all_issues[n] for n in numbers if n in all_issues}

    if since_days:
        cutoff = (datetime.now(UTC) - timedelta(days=since_days)).isoformat()
        return {n: i for n, i in all_issues.items() if i["cl"] and i["cl"] >= cutoff}

    return all_issues


def get_project_items() -> dict[int, dict]:
    """Fetch all project items with their current field values."""
    items: dict[int, dict] = {}
    cursor = None
    while True:
        after = f', after: "{cursor}"' if cursor else ""
        q = f"""{{
          organization(login: "minivess-mlops") {{
            projectV2(number: 1) {{
              items(first: 100{after}) {{
                pageInfo {{ hasNextPage endCursor }}
                nodes {{
                  id
                  content {{ ... on Issue {{ number }} }}
                  startDate: fieldValueByName(name: "Start date") {{
                    ... on ProjectV2ItemFieldDateValue {{ date }}
                  }}
                  targetDate: fieldValueByName(name: "Target date") {{
                    ... on ProjectV2ItemFieldDateValue {{ date }}
                  }}
                  size: fieldValueByName(name: "Size") {{
                    ... on ProjectV2ItemFieldSingleSelectValue {{ name }}
                  }}
                }}
              }}
            }}
          }}
        }}"""
        result = _graphql(q)
        if not result:
            print("ERROR: GraphQL failed (rate limit?)", file=sys.stderr)
            sys.exit(1)

        page = result["data"]["organization"]["projectV2"]["items"]
        for node in page["nodes"]:
            content = node.get("content")
            if not content or not content.get("number"):
                continue
            num = content["number"]
            sd = node.get("startDate")
            td = node.get("targetDate")
            sz = node.get("size")
            items[num] = {
                "id": node["id"],
                "has_start": bool(sd and sd.get("date")),
                "has_target": bool(td and td.get("date")),
                "has_size": bool(sz and sz.get("name")),
            }
        if not page["pageInfo"]["hasNextPage"]:
            break
        cursor = page["pageInfo"]["endCursor"]

    return items


def add_issue_to_project(num: int) -> str | None:
    """Add an issue to the project if not already there. Returns item ID."""
    out, rc = _run(
        f"gh project item-add 1 --owner minivess-mlops "
        f'--url "https://github.com/{REPO}/issues/{num}" '
        f"--format json"
    )
    if rc != 0:
        return None
    try:
        return json.loads(out)["id"]
    except (json.JSONDecodeError, KeyError):
        return None


# ── Core sync logic ────────────────────────────────────────────────
def sync_issue(
    num: int,
    item_id: str,
    issue: dict,
    *,
    force_target: bool = False,
    has_start: bool = False,
    has_target: bool = False,
    has_size: bool = False,
) -> bool:
    """Set timeline fields on a single project item."""
    ok = True

    # Start date: issue creation date
    if not has_start and issue["c"]:
        start = issue["c"][:10]
        ok = _set_date(item_id, FIELD_START_DATE, start) and ok

    # Target date: close date or sprint end
    if (not has_target or force_target) and issue["cl"]:
        target = issue["cl"][:10]
        ok = _set_date(item_id, FIELD_TARGET_DATE, target) and ok
    elif not has_target and issue["s"] == "OPEN":
        # Open issue: no target date (leave blank — it's unknown)
        pass

    # Size + Estimate
    if not has_size:
        size = estimate_size(num, issue["t"], issue["l"])
        ok = _set_single_select(item_id, FIELD_SIZE, SIZE_OPTIONS[size]) and ok
        ok = _set_number(item_id, FIELD_ESTIMATE, SIZE_ESTIMATE[size]) and ok

    return ok


# ── Modes ──────────────────────────────────────────────────────────
def mode_backfill() -> None:
    """Scan all items and fill missing fields."""
    issues = get_repo_issues()
    items = get_project_items()

    # Also find issues not yet in the project
    missing = set(issues.keys()) - set(items.keys())
    if missing:
        print(f"Adding {len(missing)} issues not yet in project...")
        for num in sorted(missing):
            item_id = add_issue_to_project(num)
            if item_id:
                items[num] = {
                    "id": item_id,
                    "has_start": False,
                    "has_target": False,
                    "has_size": False,
                }

    # Filter to items needing updates
    needs_update = {
        num: info
        for num, info in items.items()
        if not info["has_start"] or not info["has_target"] or not info["has_size"]
    }
    print(f"Items needing updates: {len(needs_update)} / {len(items)}")

    ok_count = 0
    err_count = 0
    for i, (num, info) in enumerate(sorted(needs_update.items())):
        issue = issues.get(num)
        if not issue:
            continue
        success = sync_issue(
            num,
            info["id"],
            issue,
            has_start=info["has_start"],
            has_target=info["has_target"],
            has_size=info["has_size"],
        )
        if success:
            ok_count += 1
        else:
            err_count += 1

        if (i + 1) % 20 == 0:
            print(
                f"  Progress: {i + 1}/{len(needs_update)} (ok={ok_count}, err={err_count})"
            )
            time.sleep(1)
        elif (i + 1) % 10 == 0:
            time.sleep(0.5)

    print(f"Done: {ok_count} updated, {err_count} errors")


def mode_recent(days: int) -> None:
    """Update Target date for issues closed in the last N days."""
    issues = get_repo_issues(since_days=days)
    if not issues:
        print(f"No issues closed in the last {days} days.")
        return

    items = get_project_items()
    print(f"Recently closed issues: {len(issues)}")

    ok_count = 0
    for num, issue in sorted(issues.items()):
        info = items.get(num)
        if not info:
            # Add to project first
            item_id = add_issue_to_project(num)
            if not item_id:
                continue
            info = {
                "id": item_id,
                "has_start": False,
                "has_target": False,
                "has_size": False,
            }

        success = sync_issue(
            num,
            info["id"],
            issue,
            force_target=True,  # Always update target for recently closed
            has_start=info["has_start"],
            has_target=False,  # Force target update
            has_size=info["has_size"],
        )
        if success:
            ok_count += 1
        time.sleep(0.3)

    print(f"Done: {ok_count} updated")


def mode_single(num: int) -> None:
    """Sync a single issue."""
    issues = get_repo_issues(numbers=[num])
    if num not in issues:
        print(f"Issue #{num} not found in repo.", file=sys.stderr)
        sys.exit(1)

    items = get_project_items()
    info = items.get(num)
    if not info:
        print(f"Issue #{num} not in project, adding...")
        item_id = add_issue_to_project(num)
        if not item_id:
            print(f"Failed to add #{num} to project.", file=sys.stderr)
            sys.exit(1)
        info = {
            "id": item_id,
            "has_start": False,
            "has_target": False,
            "has_size": False,
        }

    issue = issues[num]
    success = sync_issue(
        num,
        info["id"],
        issue,
        force_target=issue["s"] == "CLOSED",
        has_start=False,  # Always set
        has_target=False,  # Always set
        has_size=False,  # Always set
    )

    status = "CLOSED" if issue["s"] == "CLOSED" else "OPEN"
    size = estimate_size(num, issue["t"], issue["l"])
    print(
        f"#{num} ({status}): Start={issue['c'][:10]}, "
        f"Target={issue['cl'][:10] if issue['cl'] else 'open'}, "
        f"Size={size}, Estimate={SIZE_ESTIMATE[size]}"
    )
    if not success:
        print("WARNING: Some field updates failed", file=sys.stderr)


# ── CLI ────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync GitHub Project Roadmap timeline fields.",
    )
    parser.add_argument(
        "--mode",
        choices=["backfill", "recent"],
        help="Sync mode: backfill (all items) or recent (recently closed)",
    )
    parser.add_argument(
        "--issue",
        type=int,
        help="Sync a single issue by number",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="For --mode recent: number of days to look back (default: 7)",
    )
    args = parser.parse_args()

    if args.issue:
        mode_single(args.issue)
    elif args.mode == "backfill":
        mode_backfill()
    elif args.mode == "recent":
        mode_recent(args.days)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
