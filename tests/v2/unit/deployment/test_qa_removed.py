"""Tests that QA flow has been completely removed from the codebase.

QA was merged into dashboard health adapter (#342, PR #567).
All references to QA as a standalone flow must be removed.

Rule #16: No regex. Use ast, str methods, Path, yaml.safe_load, tomllib.
"""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent.parent.parent
DOCKER_DIR = ROOT / "deployment" / "docker"
ORCHESTRATION_DIR = ROOT / "src" / "minivess" / "orchestration"


class TestQaDockerRemoved:
    """Docker-level QA artifacts must not exist."""

    def test_dockerfile_qa_does_not_exist(self) -> None:
        path = DOCKER_DIR / "Dockerfile.qa"
        assert not path.exists(), f"Dockerfile.qa still exists at {path} — delete it"

    def test_compose_no_qa_service(self) -> None:
        compose_path = ROOT / "deployment" / "docker-compose.flows.yml"
        data = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
        services = data.get("services", {})
        assert "qa" not in services, (
            "docker-compose.flows.yml still has 'qa' service — remove it"
        )


class TestQaOrchestrationRemoved:
    """Orchestration code must not reference QA as a standalone flow."""

    def test_trigger_no_qa_in_default_flows(self) -> None:
        trigger_path = ORCHESTRATION_DIR / "trigger.py"
        tree = ast.parse(trigger_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == "qa":
                msg = "trigger.py still contains string literal 'qa' — remove from _DEFAULT_FLOWS"
                raise AssertionError(msg)

    def test_deployments_yaml_no_qa(self) -> None:
        deploy_yaml = ROOT / "deployment" / "prefect" / "deployments.yaml"
        if not deploy_yaml.exists():
            return
        content = deploy_yaml.read_text(encoding="utf-8")
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "qa" in stripped.lower() and "flow" in stripped.lower():
                msg = f"deployments.yaml still references qa: {stripped!r}"
                raise AssertionError(msg)

    def test_security_scan_no_qa(self) -> None:
        scan_path = ROOT / "scripts" / "weekly_security_scan.sh"
        if not scan_path.exists():
            return
        content = scan_path.read_text(encoding="utf-8")
        assert "minivess-qa" not in content, (
            "weekly_security_scan.sh still scans minivess-qa image"
        )

    def test_constants_no_qa(self) -> None:
        constants_path = ORCHESTRATION_DIR / "constants.py"
        tree = ast.parse(constants_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and "QA" in target.id:
                        msg = f"constants.py still defines {target.id} — remove it"
                        raise AssertionError(msg)

    def test_deployments_yaml_flow_list_no_qa(self) -> None:
        test_path = ROOT / "tests" / "v2" / "unit" / "test_deployments_yaml.py"
        if not test_path.exists():
            return
        content = test_path.read_text(encoding="utf-8")
        assert "qa-flow" not in content, (
            "test_deployments_yaml.py still lists 'qa-flow'"
        )

    def test_makefile_no_qa_in_scan(self) -> None:
        makefile = ROOT / "Makefile"
        if not makefile.exists():
            return
        content = makefile.read_text(encoding="utf-8")
        for line in content.splitlines():
            if "for flow in" in line and " qa " in f" {line} ":
                msg = f"Makefile scan target still lists 'qa': {line.strip()!r}"
                raise AssertionError(msg)

    def test_pipeline_flow_no_qa_import(self) -> None:
        pipeline_path = ORCHESTRATION_DIR / "flows" / "pipeline_flow.py"
        tree = ast.parse(pipeline_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    if "QA" in alias.name:
                        msg = f"pipeline_flow.py still imports {alias.name}"
                        raise AssertionError(msg)

    def test_flow_trigger_tests_no_qa(self) -> None:
        test_path = ROOT / "tests" / "v2" / "unit" / "test_flow_trigger.py"
        if not test_path.exists():
            return
        tree = ast.parse(test_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == "qa":
                msg = "test_flow_trigger.py still contains 'qa' string literal"
                raise AssertionError(msg)

    def test_constants_tests_no_qa(self) -> None:
        test_path = ROOT / "tests" / "unit" / "orchestration" / "test_constants.py"
        if not test_path.exists():
            return
        content = test_path.read_text(encoding="utf-8")
        assert "EXPERIMENT_QA" not in content, (
            "test_constants.py still imports EXPERIMENT_QA"
        )
        assert "FLOW_NAME_QA" not in content, (
            "test_constants.py still imports FLOW_NAME_QA"
        )

    def test_flow_names_tests_no_qa(self) -> None:
        test_path = ROOT / "tests" / "unit" / "orchestration" / "test_flow_names.py"
        if not test_path.exists():
            return
        content = test_path.read_text(encoding="utf-8")
        assert "FLOW_NAME_QA" not in content, (
            "test_flow_names.py still imports FLOW_NAME_QA"
        )
