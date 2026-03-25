"""Context management analytics — dashboards for metalearning, memory, and decision registry.

Generates reports on:
1. Metalearning violation frequency (most common failure patterns)
2. Memory topic file churn (active vs stale)
3. Decision registry coverage (how many questions are DO_NOT_RE_ASK)
4. Config graph coverage (YAML files with vs without Python consumers)

Usage:
    uv run python scripts/context_analytics.py              # Full report
    uv run python scripts/context_analytics.py --violations  # Metalearning only
    uv run python scripts/context_analytics.py --registry    # Registry only
    uv run python scripts/context_analytics.py --config      # Config graph only

Part of: Context Management Upgrade (Issue #906, Phase 5)
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import duckdb
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
METALEARNING_DIR = REPO_ROOT / ".claude" / "metalearning"
MEMORY_DIR = Path.home() / ".claude" / "projects" / "-home-petteri-Dropbox-github-personal-minivess-mlops" / "memory"
REGISTRY_PATH = REPO_ROOT / "knowledge-graph" / "decisions" / "registry.yaml"
METALEARNING_DB = METALEARNING_DIR / "metalearning_index.duckdb"
CONFIG_DB = REPO_ROOT / ".claude" / "config_graph.duckdb"


def report_metalearning_violations() -> None:
    """Report metalearning violation frequency and severity distribution."""
    print("\n" + "=" * 60)
    print("  METALEARNING VIOLATION FREQUENCY REPORT")
    print("=" * 60)

    md_files = sorted(METALEARNING_DIR.glob("*.md"))
    if not md_files:
        print("  No metalearning docs found")
        return

    severities: Counter[str] = Counter()
    dates: Counter[str] = Counter()
    categories: Counter[str] = Counter()

    for md_file in md_files:
        content = md_file.read_text(encoding="utf-8")

        # Extract severity
        for line in content.split("\n")[:15]:
            if "Severity" in line and ":" in line:
                sev = line.split(":", 1)[1].strip()
                # Normalize
                if "P0" in sev or "CRITICAL" in sev:
                    severities["P0-CRITICAL"] += 1
                elif "P1" in sev:
                    severities["P1-HIGH"] += 1
                else:
                    severities["OTHER"] += 1
                break
        else:
            severities["UNTAGGED"] += 1

        # Extract month
        date_str = md_file.stem[:7]  # YYYY-MM
        dates[date_str] += 1

        # Categorize by filename keywords
        stem = md_file.stem.lower()
        if "docker" in stem:
            categories["Docker"] += 1
        elif "factorial" in stem:
            categories["Factorial Design"] += 1
        elif "sam3" in stem or "sam" in stem:
            categories["SAM3/Models"] += 1
        elif "debug" in stem or "production" in stem:
            categories["Debug/Scope"] += 1
        elif "context" in stem or "amnesia" in stem or "metalearning" in stem:
            categories["Context Mgmt"] += 1
        elif "script" in stem or "standalone" in stem:
            categories["Infrastructure"] += 1
        elif "skip" in stem or "test" in stem or "failure" in stem:
            categories["Testing"] += 1
        elif "mlflow" in stem or "artifact" in stem:
            categories["MLflow/Tracking"] += 1
        else:
            categories["Other"] += 1

    print(f"\n  Total docs: {len(md_files)}")

    print("\n  Severity Distribution:")
    for sev, count in severities.most_common():
        bar = "#" * count
        print(f"    {sev:15s} {count:3d} {bar}")

    print("\n  Monthly Distribution:")
    for month, count in sorted(dates.items()):
        bar = "#" * count
        print(f"    {month} {count:3d} {bar}")

    print("\n  Category Distribution:")
    for cat, count in categories.most_common():
        bar = "#" * count
        print(f"    {cat:20s} {count:3d} {bar}")


def report_memory_churn() -> None:
    """Report memory topic file status."""
    print("\n" + "=" * 60)
    print("  MEMORY TOPIC FILE STATUS")
    print("=" * 60)

    if not MEMORY_DIR.exists():
        print("  Memory directory not found")
        return

    md_files = sorted(MEMORY_DIR.glob("*.md"))
    total = len(md_files)

    types: Counter[str] = Counter()
    for md_file in md_files:
        if md_file.name == "MEMORY.md":
            continue
        stem = md_file.stem
        if stem.startswith("feedback_"):
            types["feedback"] += 1
        elif stem.startswith("project_"):
            types["project"] += 1
        elif stem.startswith("user_"):
            types["user"] += 1
        elif stem.startswith("reference_"):
            types["reference"] += 1
        else:
            types["other"] += 1

    print(f"\n  Total topic files: {total - 1} (excluding MEMORY.md index)")

    print("\n  Type Distribution:")
    for typ, count in types.most_common():
        bar = "#" * count
        print(f"    {typ:15s} {count:3d} {bar}")


def report_decision_registry() -> None:
    """Report decision registry coverage."""
    print("\n" + "=" * 60)
    print("  DECISION REGISTRY COVERAGE")
    print("=" * 60)

    if not REGISTRY_PATH.exists():
        print("  Registry not found at", REGISTRY_PATH)
        return

    data = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    decisions = data.get("decisions", {})

    total = len(decisions)
    do_not_re_ask = sum(1 for d in decisions.values() if d.get("DO_NOT_RE_ASK"))

    print(f"\n  Total decisions: {total}")
    print(f"  DO_NOT_RE_ASK:  {do_not_re_ask}")
    print(f"  Coverage:       {do_not_re_ask}/{total} ({100*do_not_re_ask/max(total,1):.0f}%)")

    print("\n  Decisions:")
    for key, decision in decisions.items():
        flag = " [DO NOT RE-ASK]" if decision.get("DO_NOT_RE_ASK") else ""
        answer = decision.get("answer", "")
        # Truncate answer for display
        short_answer = answer[:80].replace("\n", " ").strip()
        if len(answer) > 80:
            short_answer += "..."
        print(f"    {key}: {short_answer}{flag}")


def report_config_graph() -> None:
    """Report config graph coverage."""
    print("\n" + "=" * 60)
    print("  CONFIG-TO-CODE GRAPH COVERAGE")
    print("=" * 60)

    if not CONFIG_DB.exists():
        print("  Config graph not built. Run: uv run python scripts/build_config_graph.py")
        return

    conn = duckdb.connect(str(CONFIG_DB), read_only=True)

    total_files = conn.execute("SELECT COUNT(*) FROM config_files").fetchone()[0]
    total_edges = conn.execute("SELECT COUNT(*) FROM config_edges").fetchone()[0]

    # Files with vs without consumers
    files_with_edges = conn.execute("""
        SELECT COUNT(DISTINCT cf.path)
        FROM config_files cf
        JOIN config_edges ce ON cf.path = ce.config_path
    """).fetchone()[0]

    orphan_files = total_files - files_with_edges

    print(f"\n  Total config files: {total_files}")
    print(f"  Total edges:        {total_edges}")
    print(f"  Files with Python consumers: {files_with_edges} ({100*files_with_edges/max(total_files,1):.0f}%)")
    print(f"  Orphan configs (no consumers): {orphan_files}")

    # Top config groups by edge count
    groups = conn.execute("""
        SELECT config_group, COUNT(*) as edges
        FROM config_edges
        GROUP BY config_group
        ORDER BY edges DESC
        LIMIT 10
    """).fetchall()

    print("\n  Top config groups by edge count:")
    for group, edges in groups:
        bar = "#" * min(edges, 40)
        print(f"    {group:20s} {edges:4d} {bar}")

    conn.close()


def full_report() -> None:
    """Generate complete analytics report."""
    print("\n" + "=" * 60)
    print("  CONTEXT MANAGEMENT ANALYTICS REPORT")
    print("  Date: 2026-03-22")
    print("  Issue: #906 (P0-CRITICAL)")
    print("=" * 60)

    report_metalearning_violations()
    report_memory_churn()
    report_decision_registry()
    report_config_graph()

    print("\n" + "=" * 60)
    print("  END OF REPORT")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Context management analytics")
    parser.add_argument("--violations", action="store_true", help="Metalearning only")
    parser.add_argument("--memory", action="store_true", help="Memory churn only")
    parser.add_argument("--registry", action="store_true", help="Registry only")
    parser.add_argument("--config", action="store_true", help="Config graph only")
    args = parser.parse_args()

    if args.violations:
        report_metalearning_violations()
    elif args.memory:
        report_memory_churn()
    elif args.registry:
        report_decision_registry()
    elif args.config:
        report_config_graph()
    else:
        full_report()
