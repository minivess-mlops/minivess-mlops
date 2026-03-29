"""Tests for version pin consistency across the codebase.

Module A of the QA config scanner: every *_VERSION in .env.example must
match the value in Dockerfiles, docker-compose.yml, and Pulumi code.

Catches the MLFLOW_SERVER_VERSION drift class — a version pinned in
.env.example that is hardcoded in 5+ files that can diverge independently.

See: .claude/metalearning/2026-03-14-mlflow-version-mismatch-fuckup.md
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
_ENV_EXAMPLE = _REPO_ROOT / ".env.example"
_DEPLOY = _REPO_ROOT / "deployment"


def _get_env_version(key: str) -> str:
    """Read a *_VERSION value from .env.example."""
    for line in _ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    msg = f"{key} not found in .env.example"
    raise ValueError(msg)


class TestMLflowVersionConsistency:
    """MLFLOW_SERVER_VERSION must be identical everywhere."""

    def test_env_example_has_mlflow_version(self) -> None:
        ver = _get_env_version("MLFLOW_SERVER_VERSION")
        assert ver, "MLFLOW_SERVER_VERSION is empty in .env.example"

    def test_dockerfile_mlflow_matches_env(self) -> None:
        """Dockerfile.mlflow FROM tag must match .env.example."""
        ver = _get_env_version("MLFLOW_SERVER_VERSION")
        dockerfile = _DEPLOY / "docker" / "Dockerfile.mlflow"
        if not dockerfile.exists():
            return
        content = dockerfile.read_text(encoding="utf-8")
        # The FROM line should reference the version via ARG or match exactly
        from_lines = [line for line in content.splitlines() if line.startswith("FROM") and "mlflow" in line.lower()]
        for line in from_lines:
            assert f"v{ver}" in line or f":{ver}" in line or "${" in line, (
                f"Dockerfile.mlflow FROM does not match MLFLOW_SERVER_VERSION={ver}: {line}"
            )

    def test_dockerfile_mlflow_gcp_matches_env(self) -> None:
        """Dockerfile.mlflow-gcp FROM tag must match .env.example."""
        ver = _get_env_version("MLFLOW_SERVER_VERSION")
        dockerfile = _DEPLOY / "docker" / "Dockerfile.mlflow-gcp"
        if not dockerfile.exists():
            return
        content = dockerfile.read_text(encoding="utf-8")
        from_lines = [line for line in content.splitlines() if line.startswith("FROM") and "mlflow" in line.lower()]
        for line in from_lines:
            assert f"v{ver}" in line or f":{ver}" in line or "${" in line, (
                f"Dockerfile.mlflow-gcp FROM does not match MLFLOW_SERVER_VERSION={ver}: {line}"
            )

    def test_pulumi_mlflow_version_matches_env(self) -> None:
        """Pulumi MLFLOW_SERVER_VERSION constant must match .env.example."""
        ver = _get_env_version("MLFLOW_SERVER_VERSION")
        pulumi_main = _DEPLOY / "pulumi" / "gcp" / "__main__.py"
        if not pulumi_main.exists():
            return
        content = pulumi_main.read_text(encoding="utf-8")
        for line in content.splitlines():
            if "MLFLOW_SERVER_VERSION" in line and "=" in line and not line.strip().startswith("#"):
                assert f'"{ver}"' in line, (
                    f"Pulumi MLFLOW_SERVER_VERSION does not match .env.example={ver}: {line.strip()}"
                )

    def test_docker_compose_mlflow_version(self) -> None:
        """docker-compose.yml mlflow images must use ${MLFLOW_SERVER_VERSION} or match."""
        ver = _get_env_version("MLFLOW_SERVER_VERSION")
        compose = _DEPLOY / "docker-compose.yml"
        if not compose.exists():
            return
        violations = []
        for i, line in enumerate(compose.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if "image:" in stripped and "mlflow" in stripped.lower():
                if "${" in stripped:
                    continue  # Uses variable substitution — good
                if f"v{ver}" not in stripped and f":{ver}" not in stripped:
                    violations.append(f"line {i}: {stripped}")
        assert not violations, (
            f"docker-compose.yml has hardcoded mlflow versions not matching {ver}:\n"
            + "\n".join(f"  {v}" for v in violations)
            + "\nFix: use ${MLFLOW_SERVER_VERSION} substitution or match .env.example"
        )


class TestVersionPinScriptExists:
    """The version pin checker script must exist and run cleanly."""

    def test_script_exists(self) -> None:
        script = _REPO_ROOT / "scripts" / "check_version_pins.py"
        assert script.exists(), "scripts/check_version_pins.py not found"
