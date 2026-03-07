"""Tests for env var naming consistency in orchestration flows.

TDD RED phase for T-08 (closes #407): All output directory env vars must
use the _OUTPUT_DIR suffix for consistency.
"""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

FLOWS_DIR = Path("src/minivess/orchestration/flows")
COMPOSE_FILE = Path("deployment/docker-compose.flows.yml")
FLOW_FILES = [
    FLOWS_DIR / "analysis_flow.py",
    FLOWS_DIR / "dashboard_flow.py",
    FLOWS_DIR / "qa_flow.py",
    FLOWS_DIR / "post_training_flow.py",
    FLOWS_DIR / "train_flow.py",
    FLOWS_DIR / "data_flow.py",
]


def _get_environ_get_names(source_path: Path) -> list[str]:
    """Return all env var names passed to os.environ.get() using AST parsing."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        # Match: os.environ.get("VAR_NAME", ...)
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "get"
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "environ"
        ):
            continue
        if node.args and isinstance(node.args[0], ast.Constant):
            val = node.args[0].value
            if isinstance(val, str):
                names.append(val)
    return names


class TestOutputDirEnvVarNaming:
    def test_flow_files_output_vars_have_dir_suffix(self) -> None:
        """All env vars containing 'OUTPUT' in flow files must end with '_OUTPUT_DIR'."""
        violations: list[str] = []
        for fpath in FLOW_FILES:
            if not fpath.exists():
                continue
            env_names = _get_environ_get_names(fpath)
            for name in env_names:
                if "OUTPUT" in name and not name.endswith("_OUTPUT_DIR"):
                    violations.append(f"{fpath.name}: {name!r}")
        assert not violations, (
            "Found env var names with OUTPUT but missing _DIR suffix:\n"
            + "\n".join(f"  {v}" for v in violations)
        )

    def test_docker_compose_output_vars_have_dir_suffix(self) -> None:
        """All env vars containing 'OUTPUT' in docker-compose.flows.yml must end with '_OUTPUT_DIR'."""
        if not COMPOSE_FILE.exists():
            return
        compose = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
        violations: list[str] = []
        services = compose.get("services", {})
        for svc_name, svc in services.items():
            env = svc.get("environment", {})
            # environment can be dict or list
            if isinstance(env, dict):
                keys = list(env.keys())
            elif isinstance(env, list):
                keys = [item.split("=")[0] for item in env if "=" in item]
            else:
                continue
            for key in keys:
                if "OUTPUT" in key and not key.endswith("_OUTPUT_DIR"):
                    violations.append(f"service {svc_name!r}: {key!r}")
        assert not violations, (
            "Found docker-compose env vars with OUTPUT but missing _DIR suffix:\n"
            + "\n".join(f"  {v}" for v in violations)
        )
