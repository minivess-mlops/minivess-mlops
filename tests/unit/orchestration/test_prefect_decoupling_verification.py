"""Prefect decoupling invariant verification tests.

Encodes as permanent tests the decoupling properties verified by 4 subagents.
Uses ast.parse() for Python analysis (CLAUDE.md Rule #16 — no regex).
Uses yaml.safe_load() for docker-compose parsing.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
FLOWS_DIR = REPO_ROOT / "src" / "minivess" / "orchestration" / "flows"
DOCKER_COMPOSE_FLOWS = REPO_ROOT / "deployment" / "docker-compose.flows.yml"

# Core 6 flows that must always be present (per CLAUDE.md)
CORE_FLOW_NAMES = {
    "data_flow",
    "train_flow",
    "analysis_flow",
    "deploy_flow",
    "dashboard_flow",
    "qa_flow",
}

# Docker service names corresponding to each core flow
CORE_DOCKER_SERVICES = {
    "data",
    "train",
    "analyze",
    "deploy",
    "dashboard",
    "qa",
}


def _get_flow_files() -> list[Path]:
    """Return all *_flow.py files in the flows directory."""
    return sorted(FLOWS_DIR.glob("*_flow.py"))


def _parse_flow(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _get_all_names_in_module(tree: ast.Module) -> set[str]:
    """Collect all Name and Attribute values used in the module."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def _get_function_defs(
    tree: ast.Module,
) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]


def _get_imports(tree: ast.Module) -> list[tuple[str, str | None]]:
    """Return (module, name_or_None) for all imports."""
    imports: list[tuple[str, str | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name, None))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append((module, alias.name))
    return imports


class TestAllFlowsHaveDockerGate:
    """Every *_flow.py must call _require_docker_context."""

    @pytest.mark.parametrize("flow_file", _get_flow_files(), ids=lambda p: p.stem)
    def test_all_flows_have_docker_gate(self, flow_file: Path) -> None:
        tree = _parse_flow(flow_file)
        names = _get_all_names_in_module(tree)
        assert "_require_docker_context" in names, (
            f"{flow_file.name} does not call _require_docker_context(). "
            "Every flow must hard-gate against execution outside Docker. "
            "See CLAUDE.md Rule #19 (STOP Protocol)."
        )


class TestNoCrossFlowImports:
    """No *_flow.py should import from another *_flow.py.

    Flows communicate ONLY through MLflow artifacts + Prefect artifacts.
    Cross-flow Python imports create tight coupling that defeats decoupling.
    Exception: dashboard_sections is a helper module (not a standalone flow).
    """

    @pytest.mark.parametrize("flow_file", _get_flow_files(), ids=lambda p: p.stem)
    def test_no_cross_flow_imports(self, flow_file: Path) -> None:
        tree = _parse_flow(flow_file)
        imports = _get_imports(tree)
        flow_stems = {f.stem for f in _get_flow_files() if f != flow_file}
        # dashboard_sections is a helper module, not a standalone flow
        flow_stems.discard("dashboard_sections")

        violations = []
        for module, _name in imports:
            # Check if any part of the import path is another flow module
            parts = module.split(".")
            for part in parts:
                if part in flow_stems:
                    violations.append(f"  import from '{module}'")

        assert not violations, (
            f"{flow_file.name} has cross-flow imports:\n"
            + "\n".join(violations)
            + "\nFlows must communicate only through MLflow/Prefect artifacts."
        )


class TestAllFlowsHaveFlowNameConstant:
    """Every flow must import a FLOW_NAME_* constant from orchestration.constants."""

    @pytest.mark.parametrize("flow_file", _get_flow_files(), ids=lambda p: p.stem)
    def test_all_flows_have_flow_name_constant(self, flow_file: Path) -> None:
        tree = _parse_flow(flow_file)
        imports = _get_imports(tree)
        # Look for imports of FLOW_NAME_* from minivess.orchestration.constants
        flow_name_imports = [
            (m, n)
            for m, n in imports
            if m == "minivess.orchestration.constants"
            and n is not None
            and n.startswith("FLOW_NAME")
        ]
        assert flow_name_imports, (
            f"{flow_file.name} does not import a FLOW_NAME_* constant from "
            "minivess.orchestration.constants. Every flow must declare its canonical "
            "name via the shared constants module so Prefect deployments are "
            "consistently referenced. Add: "
            "from minivess.orchestration.constants import FLOW_NAME_<FLOWNAME>"
        )


class TestNoTmpInFlowFiles:
    """/tmp and tempfile are banned in flow files (artifacts must survive container)."""

    @pytest.mark.parametrize("flow_file", _get_flow_files(), ids=lambda p: p.stem)
    def test_no_tmp_in_flow_files(self, flow_file: Path) -> None:
        tree = _parse_flow(flow_file)
        imports = _get_imports(tree)

        # Check for tempfile imports
        tempfile_imports = [
            (m, n) for m, n in imports if m == "tempfile" or n == "tempfile"
        ]
        assert not tempfile_imports, (
            f"{flow_file.name} imports tempfile. "
            "Temporary files are forbidden in flow code — artifacts must be volume-mounted. "
            "Use explicit paths under /app/checkpoints, /mlruns, /logs, etc."
        )

        # Check for /tmp string literals
        tmp_literals = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith("/tmp")
        ]
        assert not tmp_literals, (
            f"{flow_file.name} contains '/tmp' string literal(s). "
            "Paths under /tmp are not volume-mounted and will not survive container exit."
        )


class TestDockerComposeHasAllFlowServices:
    """docker-compose.flows.yml must have a service entry for each core flow."""

    def _load_compose(self) -> dict:
        assert DOCKER_COMPOSE_FLOWS.exists(), (
            f"docker-compose.flows.yml not found at {DOCKER_COMPOSE_FLOWS}. "
            "Every flow must have a corresponding Docker service definition."
        )
        return yaml.safe_load(DOCKER_COMPOSE_FLOWS.read_text(encoding="utf-8"))

    def test_docker_compose_has_all_flow_services(self) -> None:
        compose = self._load_compose()
        services = set(compose.get("services", {}).keys())
        missing = CORE_DOCKER_SERVICES - services
        assert not missing, (
            f"docker-compose.flows.yml is missing service entries for: {missing}. "
            "Every core Prefect flow must have a corresponding Docker service. "
            "See CLAUDE.md Design Goal #2: Docker-per-flow isolation."
        )
