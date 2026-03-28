"""Tests for the deterministic health harness (L1 + L2 defense).

Verifies that the health regression gate, baseline file, and session
health check scripts exist and function correctly.

See: docs/planning/v0-2_archive/critical-failure-fixing-and-silent-failure-fix.md
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]


class TestHealthBaselineExists:
    """The health baseline file must exist and have valid structure."""

    def test_baseline_file_exists(self) -> None:
        path = _REPO_ROOT / "tests" / "health_baseline.json"
        assert path.exists(), f"health_baseline.json not found at {path}"

    def test_baseline_valid_json(self) -> None:
        path = _REPO_ROOT / "tests" / "health_baseline.json"
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_baseline_has_required_keys(self) -> None:
        path = _REPO_ROOT / "tests" / "health_baseline.json"
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        required = ["updated", "updated_by_commit", "test_staging", "ruff", "deployment_state"]
        for key in required:
            assert key in data, f"health_baseline.json missing required key: {key}"

    def test_baseline_test_staging_structure(self) -> None:
        path = _REPO_ROOT / "tests" / "health_baseline.json"
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        staging = data["test_staging"]
        for key in ["collected", "passed", "failed", "skipped"]:
            assert key in staging, f"test_staging missing: {key}"
        assert staging["failed"] == 0, f"Baseline has {staging['failed']} failures — must be 0"
        assert staging["skipped"] == 0, f"Baseline has {staging['skipped']} skips — must be 0"

    def test_baseline_ruff_zero_errors(self) -> None:
        path = _REPO_ROOT / "tests" / "health_baseline.json"
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        assert data["ruff"]["error_count"] == 0, "Baseline must have 0 ruff errors"

    def test_baseline_deployment_state_has_region(self) -> None:
        path = _REPO_ROOT / "tests" / "health_baseline.json"
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        deploy = data["deployment_state"]
        assert deploy.get("gar_region") == "europe-west4", (
            f"Baseline GAR region must be europe-west4, got {deploy.get('gar_region')}"
        )


class TestHealthRegressionGateExists:
    """The regression gate script must exist and be importable."""

    def test_gate_script_exists(self) -> None:
        path = _REPO_ROOT / "scripts" / "health_regression_gate.py"
        assert path.exists(), "health_regression_gate.py not found"

    def test_gate_script_has_main(self) -> None:
        import ast

        path = _REPO_ROOT / "scripts" / "health_regression_gate.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        func_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        assert "main" in func_names, "health_regression_gate.py must have a main() function"

    def test_gate_script_runs_without_error(self) -> None:
        result = subprocess.run(
            ["uv", "run", "python", "scripts/health_regression_gate.py"],
            capture_output=True, text=True, timeout=180,
            cwd=str(_REPO_ROOT),
        )
        assert result.returncode == 0, (
            f"health_regression_gate.py failed: {result.stdout}\n{result.stderr}"
        )


class TestSessionHealthCheckExists:
    """The session health check script must exist and be executable."""

    def test_script_exists(self) -> None:
        path = _REPO_ROOT / "scripts" / "session_health_check.sh"
        assert path.exists(), "session_health_check.sh not found"

    def test_script_is_executable(self) -> None:
        import os

        path = _REPO_ROOT / "scripts" / "session_health_check.sh"
        assert os.access(path, os.X_OK), "session_health_check.sh must be executable"

    def test_script_has_shebang(self) -> None:
        path = _REPO_ROOT / "scripts" / "session_health_check.sh"
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
        assert first_line.startswith("#!/"), "session_health_check.sh must have a shebang"


class TestUpdateBaselineExists:
    """The baseline update script must exist."""

    def test_script_exists(self) -> None:
        path = _REPO_ROOT / "scripts" / "update_health_baseline.py"
        assert path.exists(), "update_health_baseline.py not found"


class TestPreCommitHasHealthGate:
    """Pre-commit config must include the health regression gate."""

    def test_health_gate_in_precommit(self) -> None:
        import yaml

        path = _REPO_ROOT / ".pre-commit-config.yaml"
        with path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f)
        hook_ids = []
        for repo in config.get("repos", []):
            for hook in repo.get("hooks", []):
                hook_ids.append(hook.get("id", ""))
        assert "health-regression-gate" in hook_ids, (
            "Pre-commit config must include health-regression-gate hook"
        )

    def test_ruff_strict_gate_in_precommit(self) -> None:
        import yaml

        path = _REPO_ROOT / ".pre-commit-config.yaml"
        with path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f)
        hook_ids = []
        for repo in config.get("repos", []):
            for hook in repo.get("hooks", []):
                hook_ids.append(hook.get("id", ""))
        assert "ruff-strict-gate" in hook_ids, (
            "Pre-commit config must include ruff-strict-gate hook"
        )
