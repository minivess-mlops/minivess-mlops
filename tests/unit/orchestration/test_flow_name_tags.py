"""Tests for flow_name tag consistency with @flow(name=...) (T-07, closes #410).

TDD RED phase: flow_name tags written to MLflow start_run() must match
the @flow(name=...) decorator value. Verified via AST parsing.
"""

from __future__ import annotations

import ast
from pathlib import Path

ORCH_DIR = Path("src/minivess/orchestration")
FLOWS_DIR = ORCH_DIR / "flows"

# Map of file → set of valid flow_name tag values.
# Files with multiple @flow functions (e.g., train_flow.py with parent +
# sub-flows) may have multiple valid tags.
EXPECTED_TAGS: dict[Path, set[str]] = {
    FLOWS_DIR / "train_flow.py": {
        "training-flow",           # parent flow + training subflow MLflow run
        "post-training-subflow",   # post-training subflow MLflow run
    },
    FLOWS_DIR / "analysis_flow.py": {"analysis-flow"},
    FLOWS_DIR / "data_flow.py": {"data-flow"},
    FLOWS_DIR / "deploy_flow.py": {"deploy-flow"},
}


def _extract_flow_name_from_dict(dict_node: ast.Dict) -> list[str]:
    """Extract flow_name values from an AST Dict node."""
    found: list[str] = []
    for key, val in zip(dict_node.keys, dict_node.values, strict=False):
        if not isinstance(key, ast.Constant):
            continue
        if not isinstance(key.value, str):
            continue
        if key.value != "flow_name":
            continue
        if isinstance(val, ast.Constant) and isinstance(val.value, str):
            found.append(val.value)
        elif isinstance(val, ast.Name):
            from minivess.orchestration import constants as _c

            resolved = getattr(_c, val.id, None)
            if isinstance(resolved, str):
                found.append(resolved)
    return found


def _find_start_run_flow_name_tags(source_path: Path) -> list[str]:
    """Extract flow_name values from mlflow.start_run(tags={...}) calls via AST.

    Handles both literal strings and constant references (e.g., FLOW_NAME_TRAIN).
    Also handles the pattern where tags dict is assigned to a variable first:
        run_tags = {"flow_name": CONST}
        mlflow.start_run(tags=run_tags)
    """
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    found: list[str] = []

    # First pass: collect variable assignments like `run_tags = {"flow_name": ...}`
    var_tags: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Dict):
                tags = _extract_flow_name_from_dict(node.value)
                if tags:
                    var_tags[target.id] = tags

    # Second pass: find mlflow.start_run(tags=...) calls
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "start_run"):
            continue
        for kw in node.keywords:
            if kw.arg != "tags":
                continue
            if isinstance(kw.value, ast.Dict):
                found.extend(_extract_flow_name_from_dict(kw.value))
            elif isinstance(kw.value, ast.Name) and kw.value.id in var_tags:
                found.extend(var_tags[kw.value.id])

    return found


class TestFlowNameTagConsistency:
    def test_start_run_flow_name_matches_flow_decorator(self) -> None:
        """flow_name tags in mlflow.start_run() must match @flow(name=...) decorator."""
        violations: list[str] = []
        for fpath, valid_tags in EXPECTED_TAGS.items():
            if not fpath.exists():
                continue
            tags = _find_start_run_flow_name_tags(fpath)
            for tag in tags:
                if tag not in valid_tags:
                    violations.append(
                        f"{fpath.name}: flow_name tag={tag!r}, "
                        f"expected one of {sorted(valid_tags)}"
                    )
        assert not violations, (
            "flow_name tags in start_run() do not match @flow decorator names:\n"
            + "\n".join(f"  {v}" for v in violations)
        )

    def test_every_flow_file_has_start_run_tag(self) -> None:
        """Every flow file must log at least one flow_name tag to MLflow."""
        missing: list[str] = []
        for fpath in EXPECTED_TAGS:
            if not fpath.exists():
                continue
            tags = _find_start_run_flow_name_tags(fpath)
            if not tags:
                missing.append(fpath.name)
        assert not missing, (
            "Flow files have no flow_name tag in start_run():\n"
            + "\n".join(f"  {m}" for m in missing)
        )
