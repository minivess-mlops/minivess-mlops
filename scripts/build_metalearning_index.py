"""Build DuckDB full-text search index over metalearning documents.

Creates a searchable index of all .claude/metalearning/*.md files, enabling
fast keyword search for relevant failure patterns and lessons learned.

Usage:
    uv run python scripts/build_metalearning_index.py
    uv run python scripts/build_metalearning_index.py --query "factorial design"
    uv run python scripts/build_metalearning_index.py --query "docker" --top 3

Output:
    .claude/metalearning/metalearning_index.duckdb

Part of: Context Management Upgrade (Issue #906)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parent.parent
METALEARNING_DIR = REPO_ROOT / ".claude" / "metalearning"
INDEX_DB = METALEARNING_DIR / "metalearning_index.duckdb"


def build_index() -> int:
    """Build or rebuild the metalearning FTS index. Returns doc count."""
    md_files = sorted(METALEARNING_DIR.glob("*.md"))
    if not md_files:
        print("No metalearning docs found")
        return 0

    conn = duckdb.connect(str(INDEX_DB))

    # Drop and recreate
    conn.execute("DROP TABLE IF EXISTS metalearning")
    conn.execute("""
        CREATE TABLE metalearning (
            filename VARCHAR,
            date VARCHAR,
            severity VARCHAR,
            title VARCHAR,
            content VARCHAR,
            word_count INTEGER
        )
    """)

    for md_file in md_files:
        content = md_file.read_text(encoding="utf-8")

        # Extract metadata from content (no regex — Rule 16)
        date = md_file.stem[:10] if len(md_file.stem) >= 10 else ""
        title = ""
        severity = ""
        for line in content.split("\n")[:10]:
            if line.startswith("# "):
                title = line[2:].strip()
            elif "Severity" in line and ":" in line:
                severity = line.split(":", 1)[1].strip()

        word_count = len(content.split())

        conn.execute(
            "INSERT INTO metalearning VALUES (?, ?, ?, ?, ?, ?)",
            [md_file.name, date, severity, title, content, word_count],
        )

    # Create FTS index
    conn.execute("INSTALL fts")
    conn.execute("LOAD fts")
    conn.execute("DROP INDEX IF EXISTS fts_metalearning")
    conn.execute("""
        PRAGMA create_fts_index(
            'metalearning', 'filename',
            'title', 'content', 'severity',
            stemmer='english',
            stopwords='english'
        )
    """)

    count = conn.execute("SELECT COUNT(*) FROM metalearning").fetchone()[0]
    conn.close()
    print(f"Indexed {count} metalearning docs → {INDEX_DB}")
    return count


def search(query: str, top_n: int = 5) -> list[dict]:
    """Search metalearning docs by keyword. Returns ranked results."""
    if not INDEX_DB.exists():
        print("Index not found. Run: uv run python scripts/build_metalearning_index.py")
        return []

    conn = duckdb.connect(str(INDEX_DB), read_only=True)
    conn.execute("LOAD fts")

    results = conn.execute(
        """
        SELECT
            filename,
            date,
            severity,
            title,
            fts_main_metalearning.match_bm25(filename, ?, fields := 'title,content,severity') AS score
        FROM metalearning
        WHERE score IS NOT NULL
        ORDER BY score DESC
        LIMIT ?
        """,
        [query, top_n],
    ).fetchall()

    conn.close()

    output = []
    for row in results:
        output.append({
            "filename": row[0],
            "date": row[1],
            "severity": row[2],
            "title": row[3],
            "score": row[4],
        })
    return output


def search_and_print(query: str, top_n: int = 5) -> None:
    """Search and print results to stdout."""
    results = search(query, top_n)
    if not results:
        print(f"No results for: {query!r}")
        return

    print(f"\nTop {len(results)} results for: {query!r}\n")
    for i, r in enumerate(results, 1):
        severity_tag = f" [{r['severity']}]" if r['severity'] else ""
        print(f"  {i}. [{r['date']}]{severity_tag} {r['title']}")
        print(f"     File: {r['filename']} (score: {r['score']:.4f})")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Metalearning search index")
    parser.add_argument("--query", "-q", help="Search query")
    parser.add_argument("--top", "-n", type=int, default=5, help="Top N results")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild")
    args = parser.parse_args()

    if args.rebuild or not INDEX_DB.exists() or args.query is None:
        build_index()

    if args.query:
        search_and_print(args.query, args.top)
