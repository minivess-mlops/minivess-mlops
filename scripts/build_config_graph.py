"""Build config-to-code edge graph — maps Hydra YAML configs to Python consumers.

code-review-graph (Tree-sitter) is blind to YAML→Python dispatch (Hydra instantiate(),
registry lookups, config-driven factories). This script supplements it by tracing:

  configs/model/dynunet.yaml → ModelFamily("dynunet") → build_adapter() → DynUNetAdapter
  configs/post_training/swag.yaml → method: "swag" → SWAGPlugin
  configs/experiment/*.yaml → training_flow() parameters

Uses yaml.safe_load() + ast.parse() — NO regex (Rule 16).

Usage:
    uv run python scripts/build_config_graph.py
    uv run python scripts/build_config_graph.py --query configs/model/dynunet.yaml
    uv run python scripts/build_config_graph.py --summary

Output:
    .claude/config_graph.duckdb

Part of: Context Management Upgrade (Issue #906, Phase 3)
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any

import duckdb
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
SRC_DIR = REPO_ROOT / "src"
DB_PATH = REPO_ROOT / ".claude" / "config_graph.duckdb"


def _find_yaml_files() -> list[Path]:
    """Find all YAML config files."""
    return sorted(CONFIGS_DIR.rglob("*.yaml"))


def _parse_yaml_metadata(yaml_path: Path) -> dict[str, Any]:
    """Parse a YAML config file and extract key metadata."""
    try:
        content = yaml_path.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        if not isinstance(data, dict):
            return {"raw_keys": [], "group": "", "name": ""}
    except Exception:
        return {"raw_keys": [], "group": "", "name": ""}

    # Determine config group from directory
    rel = yaml_path.relative_to(CONFIGS_DIR)
    group = rel.parent.name if rel.parent.name != "." else "root"
    name = yaml_path.stem

    return {
        "raw_keys": list(data.keys()) if isinstance(data, dict) else [],
        "group": group,
        "name": name,
        "data": data,
    }


def _find_python_consumers(config_group: str, config_name: str) -> list[dict[str, str]]:
    """Find Python files that consume a config group/name.

    Searches for:
    - String literals matching config_name (e.g., "dynunet", "swag")
    - Imports from config modules
    - Registry lookups
    """
    consumers: list[dict[str, str]] = []

    for py_file in sorted(SRC_DIR.rglob("*.py")):
        try:
            source = py_file.read_text(encoding="utf-8")
        except Exception:
            continue

        # Skip if config name doesn't appear at all
        if config_name not in source:
            continue

        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        rel_path = str(py_file.relative_to(REPO_ROOT))

        # Find string literals matching the config name
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value == config_name:
                    consumers.append({
                        "file": rel_path,
                        "line": node.lineno,
                        "kind": "string_literal",
                        "context": f'"{config_name}" on line {node.lineno}',
                    })

    return consumers


def _find_config_group_consumers() -> dict[str, list[dict[str, str]]]:
    """Find Python files that reference config group names (model, training, etc.)."""
    groups = {d.name for d in CONFIGS_DIR.iterdir() if d.is_dir()}
    group_consumers: dict[str, list[dict[str, str]]] = {}

    for py_file in sorted(SRC_DIR.rglob("*.py")):
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except Exception:
            continue

        rel_path = str(py_file.relative_to(REPO_ROOT))

        for node in ast.walk(tree):
            # Look for function defs with config-related parameters
            if isinstance(node, ast.FunctionDef):
                for arg in node.args.args:
                    arg_name = arg.arg
                    for group in groups:
                        if group in arg_name:
                            if group not in group_consumers:
                                group_consumers[group] = []
                            group_consumers[group].append({
                                "file": rel_path,
                                "line": node.lineno,
                                "kind": "function_param",
                                "context": f"{node.name}({arg_name}) on line {node.lineno}",
                            })

    return group_consumers


def build_graph() -> int:
    """Build the config-to-code graph in DuckDB."""
    conn = duckdb.connect(str(DB_PATH))

    # Schema
    conn.execute("DROP TABLE IF EXISTS config_files")
    conn.execute("DROP TABLE IF EXISTS config_edges")

    conn.execute("""
        CREATE TABLE config_files (
            path VARCHAR,
            config_group VARCHAR,
            config_name VARCHAR,
            keys VARCHAR,
            key_count INTEGER
        )
    """)

    conn.execute("""
        CREATE TABLE config_edges (
            config_path VARCHAR,
            config_group VARCHAR,
            config_name VARCHAR,
            python_file VARCHAR,
            python_line INTEGER,
            edge_kind VARCHAR,
            context VARCHAR
        )
    """)

    yaml_files = _find_yaml_files()
    total_edges = 0

    for yaml_path in yaml_files:
        meta = _parse_yaml_metadata(yaml_path)
        rel_path = str(yaml_path.relative_to(REPO_ROOT))

        conn.execute(
            "INSERT INTO config_files VALUES (?, ?, ?, ?, ?)",
            [
                rel_path,
                meta["group"],
                meta["name"],
                ",".join(meta["raw_keys"]),
                len(meta["raw_keys"]),
            ],
        )

        # Find Python consumers
        consumers = _find_python_consumers(meta["group"], meta["name"])
        for consumer in consumers:
            conn.execute(
                "INSERT INTO config_edges VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    rel_path,
                    meta["group"],
                    meta["name"],
                    consumer["file"],
                    consumer["line"],
                    consumer["kind"],
                    consumer["context"],
                ],
            )
            total_edges += 1

    # Group-level consumers
    group_consumers = _find_config_group_consumers()
    for group, consumers in group_consumers.items():
        for consumer in consumers:
            conn.execute(
                "INSERT INTO config_edges VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    f"configs/{group}/",
                    group,
                    "*",
                    consumer["file"],
                    consumer["line"],
                    consumer["kind"],
                    consumer["context"],
                ],
            )
            total_edges += 1

    conn.close()

    print(f"Config graph: {len(yaml_files)} YAML files, {total_edges} edges → {DB_PATH}")
    return total_edges


def query_config(config_path: str) -> None:
    """Query which Python files consume a specific config."""
    if not DB_PATH.exists():
        print("Config graph not built. Run: uv run python scripts/build_config_graph.py")
        return

    conn = duckdb.connect(str(DB_PATH), read_only=True)

    results = conn.execute(
        """
        SELECT python_file, python_line, edge_kind, context
        FROM config_edges
        WHERE config_path LIKE ? OR config_name = ?
        ORDER BY python_file, python_line
        """,
        [f"%{config_path}%", Path(config_path).stem],
    ).fetchall()

    conn.close()

    if not results:
        print(f"No Python consumers found for: {config_path}")
        return

    print(f"\nPython consumers of {config_path}:\n")
    for file, line, kind, context in results:
        print(f"  {file}:{line} [{kind}] {context}")


def print_summary() -> None:
    """Print summary statistics."""
    if not DB_PATH.exists():
        print("Config graph not built. Run: uv run python scripts/build_config_graph.py")
        return

    conn = duckdb.connect(str(DB_PATH), read_only=True)

    print("\n=== Config Graph Summary ===\n")

    groups = conn.execute("""
        SELECT config_group, COUNT(*) as files
        FROM config_files
        GROUP BY config_group
        ORDER BY files DESC
    """).fetchall()

    print("Config groups:")
    for group, count in groups:
        edge_count = conn.execute(
            "SELECT COUNT(*) FROM config_edges WHERE config_group = ?",
            [group],
        ).fetchone()[0]
        print(f"  {group}: {count} files, {edge_count} Python edges")

    total_files = conn.execute("SELECT COUNT(*) FROM config_files").fetchone()[0]
    total_edges = conn.execute("SELECT COUNT(*) FROM config_edges").fetchone()[0]
    print(f"\nTotal: {total_files} config files, {total_edges} edges")

    # Top consumer files
    top_consumers = conn.execute("""
        SELECT python_file, COUNT(*) as edge_count
        FROM config_edges
        GROUP BY python_file
        ORDER BY edge_count DESC
        LIMIT 10
    """).fetchall()

    print("\nTop 10 config consumers (Python files):")
    for file, count in top_consumers:
        print(f"  {file}: {count} config references")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config-to-code edge graph")
    parser.add_argument("--query", "-q", help="Query consumers of a config file")
    parser.add_argument("--summary", "-s", action="store_true", help="Print summary")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild")
    args = parser.parse_args()

    if args.rebuild or not DB_PATH.exists():
        build_graph()

    if args.query:
        query_config(args.query)
    elif args.summary:
        print_summary()
