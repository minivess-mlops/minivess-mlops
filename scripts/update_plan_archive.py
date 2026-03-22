"""Update the plan archive DuckDB index when new planning docs are written.

Usage:
    uv run python scripts/update_plan_archive.py --file docs/planning/v0-2_archive/original_docs/new-doc.md
    uv run python scripts/update_plan_archive.py --rebuild
    uv run python scripts/update_plan_archive.py --stale

Called by PostToolUse hook on writes to docs/planning/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from classify_plan_docs import classify_doc

DB_PATH = Path("docs/planning/v0-2_archive/plan_archive.duckdb")
NAVIGATOR_PATH = Path("docs/planning/v0-2_archive/navigator.yaml")
THEMES_DIR = Path("docs/planning/v0-2_archive/themes")
STALE_SUFFIX = ".stale"


def upsert_doc(filepath: Path) -> str:
    """Classify and upsert a single doc into the archive. Returns theme name."""
    theme = classify_doc(filepath)
    content = ""
    word_count = 0
    char_count = 0
    try:
        content = filepath.read_text(encoding="utf-8")
        word_count = len(content.split())
        char_count = len(content)
    except (UnicodeDecodeError, OSError):
        content = f"[Binary file: {filepath.name}]"

    con = duckdb.connect(str(DB_PATH))
    existing = con.execute(
        "SELECT filename FROM plan_docs WHERE filename = ?", [filepath.name]
    ).fetchone()

    if existing:
        con.execute(
            "UPDATE plan_docs SET theme=?, word_count=?, char_count=?, content=? WHERE filename=?",
            [theme, word_count, char_count, content, filepath.name],
        )
        action = "updated"
    else:
        stem = filepath.stem
        doc_type = _infer_doc_type(filepath)
        con.execute(
            "INSERT INTO plan_docs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                filepath.name,
                stem,
                theme,
                doc_type,
                filepath.suffix,
                word_count,
                char_count,
                content,
            ],
        )
        action = "inserted"

    con.execute("INSTALL fts")
    con.execute("LOAD fts")
    con.execute(
        "PRAGMA create_fts_index('plan_docs', 'filename', 'content', overwrite=1)"
    )
    con.close()

    stale_marker = THEMES_DIR / f"{theme}{STALE_SUFFIX}"
    stale_marker.touch()

    _update_navigator_counts()
    print(f"{action} {filepath.name} -> theme:{theme}")
    return theme


def _infer_doc_type(path: Path) -> str:
    """Infer document type from filename patterns."""
    stem = path.stem.lower()
    if path.suffix == ".xml":
        return "execution_plan"
    if "report" in stem:
        return "research_report"
    if "plan" in stem:
        return "plan"
    if "cold-start" in stem:
        return "cold_start"
    if "prompt" in stem:
        return "prompt"
    if "synthesis" in stem:
        return "synthesis"
    return "document"


def _update_navigator_counts() -> None:
    """Update doc_count per theme in navigator.yaml."""
    if not NAVIGATOR_PATH.exists():
        return
    con = duckdb.connect(str(DB_PATH), read_only=True)
    counts = dict(
        con.execute("SELECT theme, COUNT(*) FROM plan_docs GROUP BY theme").fetchall()
    )
    total = con.execute("SELECT COUNT(*) FROM plan_docs").fetchone()[0]
    con.close()

    nav = yaml.safe_load(NAVIGATOR_PATH.read_text(encoding="utf-8"))
    nav["total_docs"] = total
    for theme_name, theme_data in nav.get("themes", {}).items():
        theme_data["doc_count"] = counts.get(theme_name, 0)

    with NAVIGATOR_PATH.open("w", encoding="utf-8") as f:
        f.write("# Plan Archive Navigator - v0.2-beta (auto-updated)\n")
        yaml.dump(nav, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def show_stale() -> None:
    """Show which themes have stale summaries."""
    stale = sorted(THEMES_DIR.glob(f"*{STALE_SUFFIX}"))
    if stale:
        print("Stale themes (summaries need regeneration):")
        for s in stale:
            print(f"  - {s.stem}")
    else:
        print("No stale themes.")


def rebuild() -> None:
    """Rebuild entire index from scratch."""
    from build_plan_archive import build_index

    build_index()
    _update_navigator_counts()
    print("Rebuild complete.")


def main() -> None:
    if "--rebuild" in sys.argv:
        rebuild()
    elif "--stale" in sys.argv:
        show_stale()
    elif "--file" in sys.argv:
        idx = sys.argv.index("--file")
        if idx + 1 < len(sys.argv):
            filepath = Path(sys.argv[idx + 1])
            if not filepath.exists():
                print(f"ERROR: {filepath} does not exist", file=sys.stderr)
                sys.exit(1)
            upsert_doc(filepath)
        else:
            print("Usage: --file <path>", file=sys.stderr)
            sys.exit(1)
    else:
        print("Usage: --file <path> | --rebuild | --stale", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
