"""Tests for flow_name tag consistency with @flow(name=...) (T-07, closes #410).

TDD RED phase: flow_name tags written to MLflow start_run() must match
the @flow(name=...) decorator value. Verified via AST parsing.
"""

from __future__ import annotations

import ast
from pathlib import Path

ORCH_DIR = Path("src/minivess/orchestration")
FLOWS_DIR = ORCH_DIR / "flows"

# Map of file → expected flow_name tag value (must match @flow(name=...))
EXPECTED_TAGS: dict[Path, str] = {
    FLOWS_DIR / "train_flow.py": "training-flow",
    FLOWS_DIR / "post_training_flow.py": "post-training-flow",
    FLOWS_DIR / "analysis_flow.py": "analysis-flow",
    FLOWS_DIR / "data_flow.py": "data-flow",
    ORCH_DIR / "deploy_flow.py": "deploy-flow",
}


def _find_start_run_flow_name_tags(source_path: Path) -> list[str]:
    """Extract flow_name values from mlflow.start_run(tags={...}) calls via AST."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    found: list[str] = []

    for node in ast.walk(tree):
        # Match Call nodes
        if not isinstance(node, ast.Call):
            continue
        # Match mlflow.start_run(...)
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "start_run"):
            continue
        # Find tags= keyword argument
        for kw in node.keywords:
            if kw.arg != "tags":
                continue
            # tags must be a dict literal
            if not isinstance(kw.value, ast.Dict):
                continue
            for key, val in zip(kw.value.keys, kw.value.values, strict=False):
                if not isinstance(key, ast.Constant):
                    continue
                if not isinstance(key.value, str):
                    continue
                if key.value != "flow_name":
                    continue
                # Extract the string value of the flow_name entry
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    found.append(val.value)

    return found


class TestFlowNameTagConsistency:
    def test_start_run_flow_name_matches_flow_decorator(self) -> None:
        """flow_name tags in mlflow.start_run() must match @flow(name=...) decorator."""
        violations: list[str] = []
        for fpath, expected in EXPECTED_TAGS.items():
            if not fpath.exists():
                continue
            tags = _find_start_run_flow_name_tags(fpath)
            for tag in tags:
                if tag != expected:
                    violations.append(
                        f"{fpath.name}: flow_name tag={tag!r}, expected={expected!r}"
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
