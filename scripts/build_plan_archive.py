"""Build DuckDB plan archive index from classified planning docs.

Usage:
    uv run python scripts/build_plan_archive.py
    uv run python scripts/build_plan_archive.py --stats
    uv run python scripts/build_plan_archive.py --search "calibration conformal"

Creates: docs/planning/v0-2_archive/plan_archive.duckdb
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).parent))
from classify_plan_docs import ARCHIVE_DIR, classify_all

DB_PATH = Path("docs/planning/v0-2_archive/plan_archive.duckdb")


def build_index() -> None:
    """Build the DuckDB plan archive from classified docs."""
    if DB_PATH.exists():
        DB_PATH.unlink()

    con = duckdb.connect(str(DB_PATH))
    con.execute("""
        CREATE TABLE plan_docs (
            filename VARCHAR PRIMARY KEY,
            stem VARCHAR NOT NULL,
            theme VARCHAR NOT NULL,
            doc_type VARCHAR,
            extension VARCHAR,
            word_count INTEGER,
            char_count INTEGER,
            content TEXT
        )
    """)

    results = classify_all()
    total_inserted = 0

    for _theme, docs in results.items():
        for doc in docs:
            filepath = ARCHIVE_DIR / doc["filename"]
            content = ""
            word_count = 0
            char_count = 0
            try:
                content = filepath.read_text(encoding="utf-8")
                word_count = len(content.split())
                char_count = len(content)
            except (UnicodeDecodeError, OSError):
                content = f"[Binary file: {doc['filename']}]"

            con.execute(
                """INSERT INTO plan_docs VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    doc["filename"],
                    doc["stem"],
                    doc["theme"],
                    doc["doc_type"],
                    doc["extension"],
                    word_count,
                    char_count,
                    content,
                ],
            )
            total_inserted += 1

    con.execute("INSTALL fts")
    con.execute("LOAD fts")
    con.execute(
        "PRAGMA create_fts_index('plan_docs', 'filename', 'content', overwrite=1)"
    )
    print(f"Inserted {total_inserted} docs into {DB_PATH}")
    _print_stats(con)
    con.close()


def _print_stats(con: duckdb.DuckDBPyConnection) -> None:
    """Print theme statistics."""
    print("\n=== Plan Archive Statistics ===\n")
    result = con.execute("""
        SELECT theme, COUNT(*) as total, SUM(word_count) as total_words,
               ROUND(SUM(char_count) / 1024.0, 1) as total_kb
        FROM plan_docs GROUP BY theme ORDER BY total DESC
    """).fetchall()

    print(f"  {'Theme':20s} {'Docs':>5s} {'Words':>8s} {'Size (KB)':>10s}")
    print(f"  {'-' * 20} {'-' * 5} {'-' * 8} {'-' * 10}")
    for row in result:
        print(f"  {row[0]:20s} {row[1]:5d} {row[2]:8d} {row[3]:10.1f}")

    total = con.execute(
        "SELECT COUNT(*), SUM(word_count), ROUND(SUM(char_count)/1024.0, 1) FROM plan_docs"
    ).fetchone()
    print(f"\n  {'TOTAL':20s} {total[0]:5d} {total[1]:8d} {total[2]:10.1f}")


def search(query: str) -> None:
    """Full-text search across plan docs."""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    con.execute("LOAD fts")
    results = con.execute(
        """
        SELECT filename, theme, doc_type, word_count,
               fts_main_plan_docs.match_bm25(filename, ?) as score
        FROM plan_docs WHERE score IS NOT NULL ORDER BY score DESC LIMIT 20
    """,
        [query],
    ).fetchall()

    print(f"\n=== Search results for '{query}' ===\n")
    for row in results:
        print(f"  [{row[1]:15s}] {row[0]:60s} ({row[2]}, {row[3]} words)")
    con.close()


def main() -> None:
    if "--stats" in sys.argv:
        con = duckdb.connect(str(DB_PATH), read_only=True)
        _print_stats(con)
        con.close()
    elif "--search" in sys.argv:
        idx = sys.argv.index("--search")
        if idx + 1 < len(sys.argv):
            search(sys.argv[idx + 1])
        else:
            print("Usage: --search 'query'", file=sys.stderr)
            sys.exit(1)
    else:
        build_index()


if __name__ == "__main__":
    main()
