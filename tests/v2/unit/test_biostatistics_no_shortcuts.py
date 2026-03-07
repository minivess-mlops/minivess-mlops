"""Anti-shortcut guardrail tests for the biostatistics flow (Phase 0.3).

Deterministic, code-level enforcement that CANNOT be bypassed without
modifying this test file (which is visible in PR review).

Tests that inspect biostatistics_flow.py are SKIPPED if the file does
not yet exist. Once it exists, they MUST pass.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

# Absolute paths for the project root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_FLOW_FILE = (
    _PROJECT_ROOT
    / "src"
    / "minivess"
    / "orchestration"
    / "flows"
    / "biostatistics_flow.py"
)
_COMPOSE_FILE = _PROJECT_ROOT / "deployment" / "docker-compose.flows.yml"
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"


def _parse_flow_ast() -> ast.Module:
    """Parse the biostatistics_flow.py file into an AST."""
    return ast.parse(_FLOW_FILE.read_text(encoding="utf-8"), filename=str(_FLOW_FILE))


class TestNoCompatLayerImport:
    """G1: biostatistics_flow.py must NOT import from _prefect_compat."""

    @pytest.mark.skipif(
        not _FLOW_FILE.exists(), reason="biostatistics_flow.py not yet created"
    )
    def test_no_compat_layer_import(self) -> None:
        tree = _parse_flow_ast()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and "_prefect_compat" in node.module
            ):
                pytest.fail(
                    f"biostatistics_flow.py imports from _prefect_compat "
                    f"(line {node.lineno}). Use 'from prefect import flow, task' directly."
                )


class TestDockerContextCheckPresent:
    """G2: biostatistics_flow.py must call a Docker context check."""

    @pytest.mark.skipif(
        not _FLOW_FILE.exists(), reason="biostatistics_flow.py not yet created"
    )
    def test_docker_context_check_present(self) -> None:
        tree = _parse_flow_ast()
        found = False
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and (
                    "docker" in node.func.id.lower()
                    or "container" in node.func.id.lower()
                )
            ):
                found = True
                break
        assert found, (
            "biostatistics_flow.py must call _require_docker_context() or similar "
            "Docker gate function. No such call found in AST."
        )


class TestNoStandaloneScriptExists:
    """G3: No scripts/*biostatistic* or scripts/*biostats* files allowed."""

    def test_no_standalone_script_exists(self) -> None:
        biostat_scripts = list(_SCRIPTS_DIR.glob("*biostatistic*")) + list(
            _SCRIPTS_DIR.glob("*biostats*")
        )
        assert biostat_scripts == [], (
            f"Standalone biostatistics scripts found: {biostat_scripts}. "
            f"The ONLY entry points are Prefect deployment and Docker Compose."
        )


class TestNoPrefectDisabledInBiostatisticsScripts:
    """G4: No PREFECT_DISABLED in biostatistics-related shell scripts."""

    def test_no_prefect_disabled_in_biostatistics_scripts(self) -> None:
        for sh_file in _SCRIPTS_DIR.glob("*.sh"):
            content = sh_file.read_text(encoding="utf-8")
            if "biostatistic" in content.lower() and "PREFECT_DISABLED" in content:
                pytest.fail(
                    f"{sh_file.name} contains both 'biostatistic' and 'PREFECT_DISABLED'. "
                    f"The biostatistics flow requires Prefect. Remove PREFECT_DISABLED."
                )


class TestDockerComposeHasBiostatisticsService:
    """G5: docker-compose.flows.yml has a biostatistics service."""

    def test_docker_compose_has_biostatistics_service(self) -> None:
        compose = yaml.safe_load(_COMPOSE_FILE.read_text(encoding="utf-8"))
        services = compose.get("services", {})
        assert "biostatistics" in services, (
            "docker-compose.flows.yml missing 'biostatistics' service. "
            "Add it with mlruns:ro volume and biostatistics output volume."
        )

    def test_biostatistics_has_mlruns_volume_readonly(self) -> None:
        compose = yaml.safe_load(_COMPOSE_FILE.read_text(encoding="utf-8"))
        svc = compose["services"]["biostatistics"]
        volumes = svc.get("volumes", [])
        mlruns_vols = [v for v in volumes if "mlruns" in str(v)]
        assert any(":ro" in str(v) for v in mlruns_vols), (
            "biostatistics service must mount mlruns as read-only (:ro)"
        )

    def test_biostatistics_has_output_volume(self) -> None:
        compose = yaml.safe_load(_COMPOSE_FILE.read_text(encoding="utf-8"))
        svc = compose["services"]["biostatistics"]
        volumes = svc.get("volumes", [])
        output_vols = [
            v for v in volumes if "biostatistics" in str(v) and ":ro" not in str(v)
        ]
        assert len(output_vols) >= 1, (
            "biostatistics service must have a writable output volume for biostatistics artifacts"
        )

    def test_biostatistics_no_gpu_reservation(self) -> None:
        compose = yaml.safe_load(_COMPOSE_FILE.read_text(encoding="utf-8"))
        svc = compose["services"]["biostatistics"]
        deploy = svc.get("deploy", {})
        devices = deploy.get("resources", {}).get("reservations", {}).get("devices", [])
        for device in devices:
            assert "gpu" not in device.get("capabilities", []), (
                "biostatistics service must NOT reserve GPU — it is CPU-only"
            )


class TestNoTmpOrTempfileUsage:
    """G6: No tempfile import or /tmp in biostatistics code."""

    @pytest.mark.skipif(
        not _FLOW_FILE.exists(), reason="biostatistics_flow.py not yet created"
    )
    def test_no_tmp_or_tempfile_usage(self) -> None:
        # Check all biostatistics_*.py files in the pipeline directory
        pipeline_dir = _PROJECT_ROOT / "src" / "minivess" / "pipeline"
        biostat_files = list(pipeline_dir.glob("biostatistics_*.py")) + [_FLOW_FILE]

        for pyfile in biostat_files:
            if not pyfile.exists():
                continue
            tree = ast.parse(pyfile.read_text(encoding="utf-8"), filename=str(pyfile))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "tempfile":
                    pytest.fail(
                        f"{pyfile.name} imports tempfile. Use config.output_dir instead."
                    )
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "tempfile":
                            pytest.fail(
                                f"{pyfile.name} imports tempfile. Use config.output_dir."
                            )
            # Check for /tmp string literals
            source = pyfile.read_text(encoding="utf-8")
            for i, line in enumerate(source.splitlines(), 1):
                if "/tmp" in line and not line.strip().startswith("#"):
                    pytest.fail(
                        f"{pyfile.name}:{i} contains '/tmp'. "
                        f"Use config.output_dir (Docker volume-mounted)."
                    )


class TestFlowImportsRealPrefect:
    """G7: biostatistics_flow.py imports from the real prefect package."""

    @pytest.mark.skipif(
        not _FLOW_FILE.exists(), reason="biostatistics_flow.py not yet created"
    )
    def test_flow_imports_real_prefect(self) -> None:
        tree = _parse_flow_ast()
        found_prefect = False
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("prefect")
            ):
                found_prefect = True
                break
        assert found_prefect, (
            "biostatistics_flow.py must import from 'prefect' package directly. "
            "No _prefect_compat fallback allowed."
        )


class TestDockerfileExists:
    """Dockerfile.biostatistics must exist."""

    def test_dockerfile_biostatistics_exists(self) -> None:
        dockerfile = (
            _PROJECT_ROOT / "deployment" / "docker" / "Dockerfile.biostatistics"
        )
        assert dockerfile.exists(), (
            "deployment/docker/Dockerfile.biostatistics does not exist"
        )
