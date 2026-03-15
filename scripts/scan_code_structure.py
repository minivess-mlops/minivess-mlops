"""Scan src/minivess/ source code and update knowledge-graph/code-structure/*.yaml.

Uses ast.parse() to discover Prefect @flow-decorated functions and ModelAdapter
subclasses. Writes/updates flows.yaml and adapters.yaml under knowledge-graph/.

Usage:
    uv run python scripts/scan_code_structure.py
    uv run python scripts/scan_code_structure.py --dry-run

CLAUDE.md Rule #16: import re is BANNED. Use ast.parse() for Python source.
CLAUDE.md Rule #6: Use pathlib.Path, never string concatenation.
"""

from __future__ import annotations

import argparse
import ast
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
SRC_FLOWS_DIRS = [
    REPO_ROOT / "src" / "minivess" / "orchestration" / "flows",
    REPO_ROOT / "src" / "minivess" / "orchestration",
]
SRC_ADAPTERS_DIR = REPO_ROOT / "src" / "minivess" / "adapters"
KG_CODE_STRUCTURE_DIR = REPO_ROOT / "knowledge-graph" / "code-structure"

# Names that indicate a class is a ModelAdapter subclass
ADAPTER_BASE_NAMES = {"ModelAdapter"}

# Decorator names that mark a Prefect flow function
FLOW_DECORATOR_NAMES = {"flow"}


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _get_decorator_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Return plain decorator names from a function node (no arguments needed)."""
    names: list[str] = []
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name):
            names.append(dec.id)
        elif isinstance(dec, ast.Attribute):
            names.append(dec.attr)
        elif isinstance(dec, ast.Call):
            # @flow(name=...) — the call's func
            func = dec.func
            if isinstance(func, ast.Name):
                names.append(func.id)
            elif isinstance(func, ast.Attribute):
                names.append(func.attr)
    return names


def _get_decorator_kwarg(decorator: ast.expr, kwarg_name: str) -> str | None:
    """Extract a keyword argument value from a decorator Call node."""
    if not isinstance(decorator, ast.Call):
        return None
    for kw in decorator.keywords:
        if kw.arg == kwarg_name and isinstance(kw.value, ast.Constant):
            return str(kw.value.value)
    return None


def _get_base_names(node: ast.ClassDef) -> list[str]:
    """Return the plain names of base classes for a ClassDef node."""
    names: list[str] = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            names.append(base.id)
        elif isinstance(base, ast.Attribute):
            names.append(base.attr)
    return names


# ---------------------------------------------------------------------------
# Public API: extract from a single AST tree
# ---------------------------------------------------------------------------


def extract_flow_functions(tree: ast.AST, source_file: Path) -> list[dict[str, Any]]:
    """Extract @flow-decorated functions from a parsed AST tree.

    Args:
        tree: Parsed AST from ast.parse().
        source_file: Path to the source file (used as 'file' field in output).

    Returns:
        List of dicts with 'function_name', 'flow_name', 'file' keys.
    """
    flows: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        dec_names = _get_decorator_names(node)
        if not FLOW_DECORATOR_NAMES.intersection(dec_names):
            continue
        # Extract flow name from @flow(name=...) if present
        flow_name: str | None = None
        for dec in node.decorator_list:
            flow_name = _get_decorator_kwarg(dec, "name")
            if flow_name:
                break
        flows.append(
            {
                "function_name": node.name,
                "flow_name": flow_name or node.name,
                "file": str(source_file),
            }
        )
    return flows


def extract_adapter_classes(tree: ast.AST, source_file: Path) -> list[dict[str, Any]]:
    """Extract ModelAdapter subclasses from a parsed AST tree.

    Args:
        tree: Parsed AST from ast.parse().
        source_file: Path to the source file.

    Returns:
        List of dicts with 'class_name', 'base_classes', 'file' keys.
    """
    adapters: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        base_names = _get_base_names(node)
        if not ADAPTER_BASE_NAMES.intersection(base_names):
            continue
        adapters.append(
            {
                "class_name": node.name,
                "base_classes": base_names,
                "file": str(source_file),
            }
        )
    return adapters


# ---------------------------------------------------------------------------
# Public API: scan directories
# ---------------------------------------------------------------------------


def scan_flows_dir(flows_dir: Path) -> list[dict[str, Any]]:
    """Scan a directory for Python files containing @flow-decorated functions.

    Args:
        flows_dir: Directory to scan (non-recursive).

    Returns:
        Flat list of flow dicts from all Python files in the directory.
    """
    all_flows: list[dict[str, Any]] = []
    if not flows_dir.exists():
        return all_flows
    for py_file in sorted(flows_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue
        all_flows.extend(extract_flow_functions(tree, source_file=py_file))
    return all_flows


def scan_adapters_dir(adapters_dir: Path) -> list[dict[str, Any]]:
    """Scan a directory for Python files containing ModelAdapter subclasses.

    Args:
        adapters_dir: Directory to scan (non-recursive).

    Returns:
        Flat list of adapter dicts from all Python files in the directory.
    """
    all_adapters: list[dict[str, Any]] = []
    if not adapters_dir.exists():
        return all_adapters
    for py_file in sorted(adapters_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue
        all_adapters.extend(extract_adapter_classes(tree, source_file=py_file))
    return all_adapters


# ---------------------------------------------------------------------------
# Public API: write YAML outputs
# ---------------------------------------------------------------------------


def write_flows_yaml(flows: list[dict[str, Any]], out_path: Path) -> None:
    """Write discovered flows to a YAML file.

    The output preserves a `_meta` section and a `flows` list. Running twice
    on the same inputs produces identical output (idempotent).

    Args:
        flows: List of flow dicts from extract_flow_functions / scan_flows_dir.
        out_path: Destination path for the YAML file.
    """
    # Normalise: make paths repo-relative strings for portability
    repo_root_str = str(REPO_ROOT)
    normalised: list[dict[str, Any]] = []
    for f in sorted(flows, key=lambda x: x["function_name"]):
        file_path = f["file"].replace(repo_root_str + "/", "")
        normalised.append(
            {
                "id": f["function_name"],
                "flow_name": f["flow_name"],
                "file": file_path,
            }
        )

    data: dict[str, Any] = {
        "_meta": {
            "generated_by": "scripts/scan_code_structure.py",
            "last_updated": datetime.now(UTC).strftime("%Y-%m-%d"),
            "note": "Auto-generated — do not edit manually. Run /kg-sync to regenerate.",
        },
        "flows": normalised,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def write_adapters_yaml(adapters: list[dict[str, Any]], out_path: Path) -> None:
    """Write discovered adapters to a YAML file (idempotent).

    Args:
        adapters: List of adapter dicts from extract_adapter_classes / scan_adapters_dir.
        out_path: Destination path for the YAML file.
    """
    repo_root_str = str(REPO_ROOT)
    normalised: list[dict[str, Any]] = []
    for a in sorted(adapters, key=lambda x: x["class_name"]):
        file_path = a["file"].replace(repo_root_str + "/", "")
        normalised.append(
            {
                "id": a["class_name"],
                "class_name": a["class_name"],
                "base_classes": a["base_classes"],
                "file": file_path,
            }
        )

    data: dict[str, Any] = {
        "_meta": {
            "generated_by": "scripts/scan_code_structure.py",
            "last_updated": datetime.now(UTC).strftime("%Y-%m-%d"),
            "note": "Auto-generated — do not edit manually. Run /kg-sync to regenerate.",
        },
        "adapters": normalised,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan src/minivess/ and update knowledge-graph/code-structure/*.yaml"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without writing files",
    )
    parser.add_argument(
        "--flows-out",
        type=Path,
        default=KG_CODE_STRUCTURE_DIR / "flows_generated.yaml",
        help="Output path for flows YAML (default: knowledge-graph/code-structure/flows_generated.yaml)",
    )
    parser.add_argument(
        "--adapters-out",
        type=Path,
        default=KG_CODE_STRUCTURE_DIR / "adapters_generated.yaml",
        help="Output path for adapters YAML",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Scan flows from all flows directories
    all_flows: list[dict[str, Any]] = []
    for flows_dir in SRC_FLOWS_DIRS:
        all_flows.extend(scan_flows_dir(flows_dir))

    # Scan adapters
    all_adapters = scan_adapters_dir(SRC_ADAPTERS_DIR)

    print(f"Found {len(all_flows)} flows in {[str(d) for d in SRC_FLOWS_DIRS]}")
    print(f"Found {len(all_adapters)} adapters in {SRC_ADAPTERS_DIR}")

    if args.dry_run:
        print("[dry-run] Would write:")
        print(f"  flows    → {args.flows_out}")
        print(f"  adapters → {args.adapters_out}")
        return

    write_flows_yaml(all_flows, args.flows_out)
    write_adapters_yaml(all_adapters, args.adapters_out)
    print(f"Written: {args.flows_out}")
    print(f"Written: {args.adapters_out}")


if __name__ == "__main__":
    main()
