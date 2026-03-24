#!/usr/bin/env python3
"""Setup GitHub Project #6 (vascadia) — all phases automated.

Phases:
1. Create milestones
2. Add all issues to project
3. Set Priority field from labels
4. Set Status field (closed→Done, open→Backlog)
5. Set Size/Estimate from heuristics
6. Assign milestones from labels/titles
7. Backfill Start date / Target Date
8. Assign iterations

Requires: gh CLI authenticated.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from datetime import datetime

# === Project field IDs (from GraphQL query 2026-03-24) ===
PROJECT_ID = "PVT_kwHOABAuos4BSs3r"

FIELD_STATUS = "PVTSSF_lAHOABAuos4BSs3rzhAKIHk"
STATUS_BACKLOG = "f75ad846"
STATUS_DONE = "98236657"
STATUS_IN_PROGRESS = "47fc9ee4"

FIELD_PRIORITY = "PVTSSF_lAHOABAuos4BSs3rzhAKIIs"
PRIORITY_P0 = "79628723"
PRIORITY_P1 = "0a877460"
PRIORITY_P2 = "da944a9c"

FIELD_SIZE = "PVTSSF_lAHOABAuos4BSs3rzhAKIIw"
SIZE_XS = "911790be"
SIZE_S = "b277fb01"
SIZE_M = "86db8eb3"
SIZE_L = "853c8207"
SIZE_XL = "2d0801e2"

FIELD_ESTIMATE = "PVTF_lAHOABAuos4BSs3rzhAKII0"
FIELD_START_DATE = "PVTF_lAHOABAuos4BSs3rzhAKLRw"
FIELD_TARGET_DATE = "PVTF_lAHOABAuos4BSs3rzhAKLVY"

FIELD_ITERATION = "PVTIF_lAHOABAuos4BSs3rzhAKII4"
ITER_FOUNDATIONS = "381c7c80"
ITER_QA_PUB_GATE = "54cf5c95"

SIZE_TO_ESTIMATE = {"XS": 1, "S": 2, "M": 3, "L": 5, "XL": 8}

# === Milestone mapping (label/title patterns → milestone name) ===
MILESTONE_RULES = [
    ("v0.2-infrastructure", r"Docker|SkyPilot|Pulumi|CI|cloud|GCP|RunPod|quota|infrastructure|docker|ci-cd|prefect"),
    ("v0.2-training", r"train|loss|checkpoint|epoch|fold|resume|training"),
    ("v0.2-data", r"\bdata\b|DVC|loader|augment|dataset|NIfTI|split|annotation"),
    ("v0.2-evaluation", r"analysis|ensemble|evaluat|comparison|champion|metrics.reloaded"),
    ("v0.2-biostatistics", r"biostat|ANOVA|ranking|pairwise|variance|figure|table|LaTeX"),
    ("v0.2-observability", r"MLflow|lineage|drift|profil|metric.key|tracking|observability|mlflow|monitoring"),
    ("v0.2-post-training", r"post.training|calibrat|conformal|SWAG|checkpoint.averag"),
    ("v0.2-deployment", r"deploy|ONNX|BentoML|serving|inference|model.registry"),
    ("v0.2-dashboard", r"dashboard|Gradio|health.check|QA.flow"),
    ("v0.2-compliance", r"audit|regulatory|TRIPOD|EU.AI|IEC|FDA|SaMD|compliance"),
    ("v0.2-models", r"SAM3|VesselFM|Mamba|DynUNet|adapter|model.family|models"),
]


def gh(args: list[str], input_data: str | None = None) -> str:
    """Run gh CLI command and return stdout."""
    result = subprocess.run(
        ["gh"] + args,
        capture_output=True, text=True, input=input_data, timeout=30
    )
    if result.returncode != 0 and "already exists" not in result.stderr:
        print(f"  WARN: gh {' '.join(args[:3])}... → {result.stderr[:200]}", file=sys.stderr)
    return result.stdout.strip()


def graphql(query: str, retries: int = 3) -> dict:
    """Execute GraphQL query via gh api."""
    for attempt in range(retries):
        result = subprocess.run(
            ["gh", "api", "graphql", "-f", f"query={query}"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
        if "rate limit" in result.stderr.lower() or "abuse" in result.stderr.lower():
            wait = 10 * (attempt + 1)
            print(f"  Rate limited, waiting {wait}s...")
            time.sleep(wait)
            continue
        if result.stderr:
            print(f"  GraphQL error: {result.stderr[:200]}", file=sys.stderr)
        return {}
    return {}


def set_field_single_select(item_id: str, field_id: str, option_id: str) -> None:
    graphql(f'''mutation {{
        updateProjectV2ItemFieldValue(input: {{
            projectId: "{PROJECT_ID}", itemId: "{item_id}",
            fieldId: "{field_id}",
            value: {{ singleSelectOptionId: "{option_id}" }}
        }}) {{ projectV2Item {{ id }} }}
    }}''')


def set_field_date(item_id: str, field_id: str, date_str: str) -> None:
    graphql(f'''mutation {{
        updateProjectV2ItemFieldValue(input: {{
            projectId: "{PROJECT_ID}", itemId: "{item_id}",
            fieldId: "{field_id}",
            value: {{ date: "{date_str}" }}
        }}) {{ projectV2Item {{ id }} }}
    }}''')


def set_field_number(item_id: str, field_id: str, value: float) -> None:
    graphql(f'''mutation {{
        updateProjectV2ItemFieldValue(input: {{
            projectId: "{PROJECT_ID}", itemId: "{item_id}",
            fieldId: "{field_id}",
            value: {{ number: {value} }}
        }}) {{ projectV2Item {{ id }} }}
    }}''')


def set_field_iteration(item_id: str, field_id: str, iteration_id: str) -> None:
    graphql(f'''mutation {{
        updateProjectV2ItemFieldValue(input: {{
            projectId: "{PROJECT_ID}", itemId: "{item_id}",
            fieldId: "{field_id}",
            value: {{ iterationId: "{iteration_id}" }}
        }}) {{ projectV2Item {{ id }} }}
    }}''')


def add_item_to_project(content_id: str) -> str | None:
    """Add issue/PR to project, return item ID."""
    result = graphql(f'''mutation {{
        addProjectV2ItemById(input: {{
            projectId: "{PROJECT_ID}", contentId: "{content_id}"
        }}) {{ item {{ id }} }}
    }}''')
    item = result.get("data", {}).get("addProjectV2ItemById", {}).get("item", {})
    return item.get("id")


# =====================================================================
# PHASE 1: Create milestones
# =====================================================================
def phase1_create_milestones() -> dict[str, int]:
    """Create milestones, return name→number mapping."""
    print("\n=== PHASE 1: Create Milestones ===")
    milestones = {
        "v0.2-infrastructure": "Docker, SkyPilot, Pulumi, CI, cloud setup",
        "v0.2-training": "Training flow, loss functions, model adapters, checkpointing",
        "v0.2-data": "Data loading, augmentation, DVC, external datasets",
        "v0.2-evaluation": "Analysis flow, ensemble builder, metrics, comparison",
        "v0.2-biostatistics": "Biostatistics flow, ANOVA, figures, tables, rankings",
        "v0.2-observability": "MLflow, lineage, profiling, drift detection",
        "v0.2-post-training": "Post-training flow, calibration, conformal, SWAG",
        "v0.2-deployment": "Deploy flow, ONNX, BentoML, model registry",
        "v0.2-dashboard": "Dashboard flow, Gradio, health checks, QA",
        "v0.2-compliance": "Audit trail, regulatory docs, TRIPOD, EU AI Act",
        "v0.2-models": "SAM3, VesselFM, MambaVesselNet, DynUNet adapters",
    }
    result = {}
    for title, desc in milestones.items():
        out = gh(["api", "repos/petteriTeikari/vascadia/milestones",
                  "-f", f"title={title}", "-f", f"description={desc}", "-f", "state=open"])
        if out:
            data = json.loads(out)
            result[title] = data.get("number", 0)
            print(f"  Created: {title} (#{data.get('number', '?')})")
        else:
            # Already exists — find it
            existing = gh(["api", "repos/petteriTeikari/vascadia/milestones",
                          "--jq", f'[.[] | select(.title=="{title}")][0].number'])
            if existing:
                result[title] = int(existing)
                print(f"  Exists: {title} (#{existing})")
    return result


# =====================================================================
# PHASE 2: Add all issues + PHASE 3-7: Set all fields
# =====================================================================
def fetch_all_issues() -> list[dict]:
    """Fetch all repo issues with labels, state, dates."""
    print("\n  Fetching all issues...")
    issues = []
    page = 1
    while True:
        out = gh(["issue", "list", "--repo", "petteriTeikari/vascadia",
                  "--state", "all", "--limit", "100", "--json",
                  "number,title,state,labels,createdAt,closedAt,id",
                  "--jq", f".[{(page-1)*100}:{page*100}]"])
        # Use paginated approach
        break  # gh issue list doesn't paginate well, use single call
    out = gh(["issue", "list", "--repo", "petteriTeikari/vascadia",
              "--state", "all", "--limit", "1000", "--json",
              "number,title,state,labels,createdAt,closedAt,id"])
    if out:
        issues = json.loads(out)
    print(f"  Found {len(issues)} issues")
    return issues


def fetch_project_items() -> dict[str, str]:
    """Fetch existing project items, return nodeId→itemId mapping."""
    print("  Fetching existing project items...")
    mapping = {}
    cursor = None
    while True:
        after = f', after: "{cursor}"' if cursor else ""
        result = graphql(f'''query {{
            user(login: "petteriTeikari") {{
                projectV2(number: 6) {{
                    items(first: 100{after}) {{
                        pageInfo {{ hasNextPage endCursor }}
                        nodes {{
                            id
                            content {{
                                ... on Issue {{ id }}
                                ... on PullRequest {{ id }}
                            }}
                        }}
                    }}
                }}
            }}
        }}''')
        items = result.get("data", {}).get("user", {}).get("projectV2", {}).get("items", {})
        for node in items.get("nodes", []):
            content = node.get("content")
            if content and content.get("id"):
                mapping[content["id"]] = node["id"]
        if not items.get("pageInfo", {}).get("hasNextPage"):
            break
        cursor = items["pageInfo"]["endCursor"]
    print(f"  Found {len(mapping)} existing project items")
    return mapping


def determine_priority(labels: list[str]) -> str | None:
    """Map label names to priority option ID."""
    label_names = {l if isinstance(l, str) else l.get("name", "") for l in labels}
    if "P0-critical" in label_names or "P0" in label_names:
        return PRIORITY_P0
    if "P1-high" in label_names or "P1" in label_names:
        return PRIORITY_P1
    if "P2-medium" in label_names or "P3-low" in label_names:
        return PRIORITY_P2
    return None


def determine_size(labels: list[str], title: str) -> tuple[str, int]:
    """Determine size and estimate from labels/title."""
    label_names = {l if isinstance(l, str) else l.get("name", "") for l in labels}
    if "bug" in label_names:
        return SIZE_S, 2
    if "refactor" in label_names or "tech-debt" in label_names:
        return SIZE_S, 2
    if "science" in label_names or "research" in label_names:
        return SIZE_L, 5
    if "infrastructure" in label_names:
        return SIZE_M, 3
    if "enhancement" in label_names:
        return SIZE_M, 3
    return SIZE_S, 2


def determine_milestone(labels: list[str], title: str) -> str | None:
    """Match issue to a milestone based on labels and title."""
    label_names = " ".join(l if isinstance(l, str) else l.get("name", "") for l in labels)
    search_text = f"{title} {label_names}"
    for ms_name, pattern in MILESTONE_RULES:
        if re.search(pattern, search_text, re.IGNORECASE):
            return ms_name
    return None


def determine_iteration(state: str, closed_at: str | None) -> str:
    """Assign to iteration based on close date."""
    if state == "CLOSED" and closed_at:
        close_date = datetime.fromisoformat(closed_at.replace("Z", "+00:00"))
        if close_date < datetime(2026, 3, 23, tzinfo=close_date.tzinfo):
            return ITER_FOUNDATIONS
    return ITER_QA_PUB_GATE


def run_all_phases(milestones: dict[str, int]) -> None:
    """Execute phases 2-7 for all issues."""
    issues = fetch_all_issues()
    existing_items = fetch_project_items()

    # Get milestone numbers
    ms_numbers = {}
    out = gh(["api", "repos/petteriTeikari/vascadia/milestones", "--jq",
              "[.[] | {title: .title, number: .number}]"])
    if out:
        for ms in json.loads(out):
            ms_numbers[ms["title"]] = ms["number"]

    print(f"\n=== Processing {len(issues)} issues ===")
    added = 0
    updated = 0
    errors = 0

    for i, issue in enumerate(issues):
        num = issue["number"]
        node_id = issue["id"]
        title = issue["title"]
        state = issue["state"]
        labels = issue.get("labels", [])
        created_at = issue.get("createdAt", "")
        closed_at = issue.get("closedAt")
        label_names = [l["name"] if isinstance(l, dict) else l for l in labels]

        # Progress
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(issues)} (added={added}, updated={updated})")

        # Phase 2: Add to project if not already there
        item_id = existing_items.get(node_id)
        if not item_id:
            item_id = add_item_to_project(node_id)
            if item_id:
                added += 1
                existing_items[node_id] = item_id
            else:
                errors += 1
                continue
            # Small delay to avoid rate limiting
            if added % 20 == 0:
                time.sleep(1)

        # Phase 3: Set Priority
        priority = determine_priority(label_names)
        if priority:
            set_field_single_select(item_id, FIELD_PRIORITY, priority)

        # Phase 4: Set Status
        if state == "CLOSED":
            set_field_single_select(item_id, FIELD_STATUS, STATUS_DONE)
        else:
            if any(p in label_names for p in ["P0-critical", "P0"]):
                set_field_single_select(item_id, FIELD_STATUS, STATUS_IN_PROGRESS)
            else:
                set_field_single_select(item_id, FIELD_STATUS, STATUS_BACKLOG)

        # Phase 5: Set Size/Estimate
        size_id, estimate = determine_size(label_names, title)
        set_field_single_select(item_id, FIELD_SIZE, size_id)
        set_field_number(item_id, FIELD_ESTIMATE, estimate)

        # Phase 6: Assign milestone
        ms_name = determine_milestone(label_names, title)
        if ms_name and ms_name in ms_numbers:
            gh(["issue", "edit", str(num), "--repo", "petteriTeikari/vascadia",
                "--milestone", ms_name])

        # Phase 7a: Set Start date
        if created_at:
            date_str = created_at[:10]  # YYYY-MM-DD
            set_field_date(item_id, FIELD_START_DATE, date_str)

        # Phase 7b: Set Target Date for closed issues
        if closed_at:
            date_str = closed_at[:10]
            set_field_date(item_id, FIELD_TARGET_DATE, date_str)

        # Phase 7c: Set Iteration
        iter_id = determine_iteration(state, closed_at)
        set_field_iteration(item_id, FIELD_ITERATION, iter_id)

        updated += 1

        # Rate limit: ~8 mutations per issue, stay under 5000/hr
        if updated % 10 == 0:
            time.sleep(0.5)

    print(f"\n=== DONE ===")
    print(f"  Added to project: {added}")
    print(f"  Updated: {updated}")
    print(f"  Errors: {errors}")


def main() -> None:
    print("=" * 60)
    print("GitHub Project #6 (vascadia) Setup")
    print("=" * 60)

    milestones = phase1_create_milestones()
    run_all_phases(milestones)


if __name__ == "__main__":
    main()
