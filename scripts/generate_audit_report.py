"""Generate audit report from plan archive theme summaries.

Usage:
    uv run python scripts/generate_audit_report.py
    uv run python scripts/generate_audit_report.py --phantom
    uv run python scripts/generate_audit_report.py --health

Reads theme summaries from docs/planning/v0-2_archive/themes/ and
generates docs/planning/v0-2_archive/audit-report.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import yaml

ARCHIVE_DIR = Path("docs/planning/v0-2_archive")
THEMES_DIR = ARCHIVE_DIR / "themes"
DB_PATH = ARCHIVE_DIR / "plan_archive.duckdb"
NAVIGATOR_PATH = ARCHIVE_DIR / "navigator.yaml"
OUTPUT_PATH = ARCHIVE_DIR / "audit-report.md"


def generate_health_dashboard() -> str:
    """Generate theme health dashboard from DuckDB."""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    stats = con.execute("""
        SELECT theme, COUNT(*) as total,
               COUNT(*) FILTER (WHERE doc_type = 'execution_plan') as plans,
               COUNT(*) FILTER (WHERE doc_type = 'research_report') as reports,
               COUNT(*) FILTER (WHERE doc_type IN ('cold_start', 'synthesis')) as meta,
               ROUND(SUM(char_count) / 1024.0, 1) as total_kb
        FROM plan_docs GROUP BY theme ORDER BY total DESC
    """).fetchall()
    total_row = con.execute(
        "SELECT COUNT(*), ROUND(SUM(char_count) / 1024.0, 1) FROM plan_docs"
    ).fetchone()
    con.close()

    lines = [
        "## Theme Health Dashboard\n",
        "| Theme | Total | Plans | Reports | Meta | Size (KB) | Summary Exists |",
        "|-------|-------|-------|---------|------|-----------|----------------|",
    ]
    for row in stats:
        theme = row[0]
        summary_exists = "YES" if (THEMES_DIR / f"{theme}.md").exists() else "NO"
        lines.append(
            f"| {theme} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | {summary_exists} |"
        )
    lines.append(f"\n**Total: {total_row[0]} docs, {total_row[1]} KB**\n")
    return "\n".join(lines)


def find_phantom_plans() -> str:
    """Scan theme summaries for phantom plans."""
    lines = ["## Phantom Plans (Planned But Never Implemented)\n"]
    theme_files = sorted(THEMES_DIR.glob("*.md"))
    if not theme_files:
        lines.append("*No theme summaries generated yet. Run Phase 1 first.*\n")
        return "\n".join(lines)

    phantom_count = 0
    for theme_file in theme_files:
        content = theme_file.read_text(encoding="utf-8")
        theme_name = theme_file.stem
        phantom_lines = []
        for line in content.split("\n"):
            lower = line.lower()
            if (
                "phantom" in lower or "not implemented" in lower or "missing" in lower
            ) and "|" in line:
                phantom_lines.append(line.strip())
        if phantom_lines:
            lines.append(f"### {theme_name} ({len(phantom_lines)} gap entries)\n")
            for pl in phantom_lines:
                lines.append(pl)
            lines.append("")
            phantom_count += len(phantom_lines)

    lines.insert(1, f"**Total gap entries across all themes: {phantom_count}**\n")
    return "\n".join(lines)


def generate_full_report() -> str:
    """Generate the complete audit report."""
    nav = yaml.safe_load(NAVIGATOR_PATH.read_text(encoding="utf-8"))
    report = [
        "---",
        "title: Plan Archive Audit Report",
        "generated: 2026-03-22",
        f"total_docs: {nav.get('total_docs', 'unknown')}",
        "---",
        "",
        "# Plan Archive Audit Report - v0.2-beta",
        "",
        generate_health_dashboard(),
        "",
        find_phantom_plans(),
        "",
        "## Theme Summaries\n",
    ]
    theme_files = sorted(THEMES_DIR.glob("*.md"))
    if theme_files:
        for tf in theme_files:
            size_kb = tf.stat().st_size / 1024
            report.append(f"- `{tf.name}` ({size_kb:.1f} KB)")
    else:
        report.append("*No theme summaries generated yet.*")
    return "\n".join(report)


def main() -> None:
    if "--phantom" in sys.argv:
        print(find_phantom_plans())
    elif "--health" in sys.argv:
        print(generate_health_dashboard())
    else:
        report = generate_full_report()
        OUTPUT_PATH.write_text(report, encoding="utf-8")
        print(f"Audit report written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
