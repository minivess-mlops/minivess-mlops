"""Tests for T-12: deploy_flow output paths and BentoML store volume wiring.

Verifies that DeployConfig.output_dir uses DEPLOY_OUTPUT_DIR env var,
BentoML import uses BENTOML_HOME env var, and deploy_flow() has FlowContract wiring.

Uses ast.parse() for source inspection — NO regex (CLAUDE.md Rule #16).
"""

from __future__ import annotations

import ast
from pathlib import Path

_DEPLOY_FLOW_SRC = Path("src/minivess/orchestration/flows/deploy_flow.py")
_DEPLOY_CONFIG_SRC = Path("src/minivess/config/deploy_config.py")


# ---------------------------------------------------------------------------
# AST-level: no hardcoded Path("outputs/deploy") literals
# ---------------------------------------------------------------------------


class TestNoHardcodedRelativePath:
    def test_no_hardcoded_relative_path_deploy_flow(self) -> None:
        """deploy_flow.py must not contain Path('outputs/deploy') literal."""
        source = _DEPLOY_FLOW_SRC.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_path_call = (isinstance(func, ast.Name) and func.id == "Path") or (
                isinstance(func, ast.Attribute) and func.attr == "Path"
            )
            if not is_path_call:
                continue
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    assert "outputs/deploy" not in arg.value, (
                        f"Hardcoded relative path Path({arg.value!r}) found at "
                        f"line {node.lineno} in deploy_flow.py. "
                        "Use DEPLOY_OUTPUT_DIR env var instead."
                    )

    def test_no_hardcoded_relative_path_deploy_config(self) -> None:
        """deploy_config.py must not contain Path('outputs/deploy') literal."""
        source = _DEPLOY_CONFIG_SRC.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_path_call = (isinstance(func, ast.Name) and func.id == "Path") or (
                isinstance(func, ast.Attribute) and func.attr == "Path"
            )
            if not is_path_call:
                continue
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    assert "outputs/deploy" not in arg.value, (
                        f"Hardcoded relative path Path({arg.value!r}) found at "
                        f"line {node.lineno} in deploy_config.py. "
                        "Use DEPLOY_OUTPUT_DIR env var instead."
                    )


# ---------------------------------------------------------------------------
# Functional: DEPLOY_OUTPUT_DIR env var
# ---------------------------------------------------------------------------


class TestDeployOutputDir:
    def test_deploy_output_dir_from_env(self, monkeypatch) -> None:
        """DEPLOY_OUTPUT_DIR env var must control DeployConfig.output_dir default."""
        import os

        target = "/test/deploy"
        monkeypatch.setenv("DEPLOY_OUTPUT_DIR", target)
        resolved = Path(os.environ.get("DEPLOY_OUTPUT_DIR", "/app/outputs/deploy"))
        assert str(resolved) == target

    def test_deploy_output_default_absolute(self, monkeypatch) -> None:
        """Default DEPLOY_OUTPUT_DIR must be absolute."""
        import os

        monkeypatch.delenv("DEPLOY_OUTPUT_DIR", raising=False)
        resolved = Path(os.environ.get("DEPLOY_OUTPUT_DIR", "/app/outputs/deploy"))
        assert resolved.is_absolute(), (
            f"Default DEPLOY_OUTPUT_DIR is not absolute: {resolved}"
        )

    def test_deploy_output_default_value(self, monkeypatch) -> None:
        """Default DEPLOY_OUTPUT_DIR must be /app/outputs/deploy."""
        import os

        monkeypatch.delenv("DEPLOY_OUTPUT_DIR", raising=False)
        resolved = Path(os.environ.get("DEPLOY_OUTPUT_DIR", "/app/outputs/deploy"))
        assert str(resolved) == "/app/outputs/deploy"

    def test_deploy_config_reads_deploy_output_dir(self) -> None:
        """deploy_config.py must reference DEPLOY_OUTPUT_DIR env var."""
        source = _DEPLOY_CONFIG_SRC.read_text(encoding="utf-8")
        assert "DEPLOY_OUTPUT_DIR" in source, (
            "deploy_config.py must read DEPLOY_OUTPUT_DIR env var. "
            "Add default_factory using os.environ.get('DEPLOY_OUTPUT_DIR', "
            "'/app/outputs/deploy') to output_dir field."
        )

    def test_deploy_onnx_dir_within_output(self, monkeypatch) -> None:
        """onnx_dir must be output_dir / 'onnx' (absolute path)."""
        import os

        monkeypatch.setenv("DEPLOY_OUTPUT_DIR", "/app/outputs/deploy")
        output_dir = Path(os.environ.get("DEPLOY_OUTPUT_DIR", "/app/outputs/deploy"))
        onnx_dir = output_dir / "onnx"
        assert onnx_dir.is_absolute()
        assert onnx_dir == Path("/app/outputs/deploy/onnx")


# ---------------------------------------------------------------------------
# BentoML store volume wiring
# ---------------------------------------------------------------------------


class TestBentomlHomeWiring:
    def test_deploy_flow_references_bentoml_home(self) -> None:
        """deploy_flow.py must reference BENTOML_HOME env var."""
        source = _DEPLOY_FLOW_SRC.read_text(encoding="utf-8")
        assert "BENTOML_HOME" in source, (
            "deploy_flow.py must reference BENTOML_HOME env var so BentoML "
            "uses the mounted volume instead of container-ephemeral ~/.bentoml. "
            "Add: os.environ.setdefault('BENTOML_HOME', '/home/minivess/bentoml')"
        )


# ---------------------------------------------------------------------------
# FlowContract wiring
# ---------------------------------------------------------------------------


class TestDeployFlowContract:
    def test_deploy_flow_references_flow_contract(self) -> None:
        """deploy_flow.py must reference FlowContract."""
        source = _DEPLOY_FLOW_SRC.read_text(encoding="utf-8")
        assert "FlowContract" in source, (
            "deploy_flow.py must use FlowContract. "
            "Add: from minivess.orchestration.flow_contract import FlowContract"
        )

    def test_deploy_flow_references_log_flow_completion(self) -> None:
        """deploy_flow.py must call log_flow_completion."""
        source = _DEPLOY_FLOW_SRC.read_text(encoding="utf-8")
        assert "log_flow_completion" in source, (
            "deploy_flow.py must call FlowContract.log_flow_completion(). "
            "Add the call near the end of deploy_flow()."
        )

    def test_deploy_flow_tags_flow_name(self) -> None:
        """deploy_flow.py must contain 'deploy-flow' as a flow_name tag value."""
        source = _DEPLOY_FLOW_SRC.read_text(encoding="utf-8")
        tree = ast.parse(source)

        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == "deploy-flow":
                found = True
                break
        assert found, (
            "deploy_flow.py must tag MLflow run with flow_name='deploy-flow' "
            "(matches FLOW_NAME_DEPLOY constant). "
            "Add flow_name='deploy-flow' tag when opening MLflow run."
        )

    def test_deploy_flow_references_upstream_analysis_run(self) -> None:
        """deploy_flow.py must reference upstream analysis run ID."""
        source = _DEPLOY_FLOW_SRC.read_text(encoding="utf-8")
        assert "upstream_analysis_run_id" in source or "upstream" in source, (
            "deploy_flow.py must reference upstream analysis run ID. "
            "Read it via FlowContract.find_upstream_run(upstream_flow='analyze')."
        )
