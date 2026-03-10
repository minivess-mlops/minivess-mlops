"""Tests for the 3-tier base Docker image hierarchy.

Verifies Dockerfile.base-cpu and Dockerfile.base-light exist with correct
structure: multi-stage build, python:3.13-slim-bookworm, no CUDA, non-root
user, HEALTHCHECK, OCI labels, requirements-{tier}.txt usage.

Rule #16: No regex. Use str methods only.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent.parent
DOCKER_DIR = ROOT / "deployment" / "docker"


def _read_dockerfile(name: str) -> str:
    """Read a Dockerfile from the docker directory."""
    path = DOCKER_DIR / name
    return path.read_text(encoding="utf-8")


class TestAllBaseDockerfilesExist:
    """All three base Dockerfiles must exist."""

    def test_base_gpu_exists(self) -> None:
        assert (DOCKER_DIR / "Dockerfile.base").exists()

    def test_base_cpu_exists(self) -> None:
        assert (DOCKER_DIR / "Dockerfile.base-cpu").exists(), (
            "Dockerfile.base-cpu not found — create it for Tier B (CPU)"
        )

    def test_base_light_exists(self) -> None:
        assert (DOCKER_DIR / "Dockerfile.base-light").exists(), (
            "Dockerfile.base-light not found — create it for Tier C (Light)"
        )

    def test_only_gpu_base_has_nvidia(self) -> None:
        """Only Dockerfile.base should have nvidia/cuda in FROM lines."""
        gpu_content = _read_dockerfile("Dockerfile.base")
        assert "nvidia/cuda" in gpu_content, "Dockerfile.base must use nvidia/cuda"

        for name in ("Dockerfile.base-cpu", "Dockerfile.base-light"):
            if (DOCKER_DIR / name).exists():
                content = _read_dockerfile(name)
                # Check only non-comment lines for nvidia/cuda references
                for line in content.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        continue
                    assert "nvidia" not in stripped.lower(), (
                        f"{name} non-comment line references nvidia: {stripped!r}"
                    )
                    assert "cuda" not in stripped.lower(), (
                        f"{name} non-comment line references cuda: {stripped!r}"
                    )


class TestDockerfileCpuStructure:
    """Dockerfile.base-cpu must follow H3+H4 hardening pattern."""

    def test_uses_python_313_builder(self) -> None:
        content = _read_dockerfile("Dockerfile.base-cpu")
        assert "python:3.13-slim-bookworm" in content, (
            "Dockerfile.base-cpu must use python:3.13-slim-bookworm"
        )

    def test_has_multi_stage(self) -> None:
        content = _read_dockerfile("Dockerfile.base-cpu")
        assert "AS builder" in content, "Missing builder stage"
        assert "AS runner" in content, "Missing runner stage"

    def test_has_tier_label(self) -> None:
        content = _read_dockerfile("Dockerfile.base-cpu")
        found = any(
            line.strip().startswith('LABEL tier="cpu"') for line in content.splitlines()
        )
        assert found, 'Missing LABEL tier="cpu"'

    def test_has_nonroot_user(self) -> None:
        content = _read_dockerfile("Dockerfile.base-cpu")
        assert "useradd" in content, "Missing non-root user creation"
        assert "minivess" in content, "User must be named 'minivess'"

    def test_has_healthcheck(self) -> None:
        content = _read_dockerfile("Dockerfile.base-cpu")
        assert "HEALTHCHECK" in content, "Missing HEALTHCHECK"

    def test_uses_requirements_cpu(self) -> None:
        content = _read_dockerfile("Dockerfile.base-cpu")
        assert "requirements-cpu.txt" in content, (
            "Must use pre-generated requirements-cpu.txt"
        )

    def test_uses_uv_pip_install(self) -> None:
        content = _read_dockerfile("Dockerfile.base-cpu")
        assert "uv pip install" in content, "Must use 'uv pip install' (not 'uv sync')"

    def test_no_uv_sync(self) -> None:
        content = _read_dockerfile("Dockerfile.base-cpu")
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert "uv sync" not in stripped, (
                f"Dockerfile.base-cpu must NOT use 'uv sync': {stripped!r}"
            )

    def test_has_pythonpath(self) -> None:
        content = _read_dockerfile("Dockerfile.base-cpu")
        assert "PYTHONPATH=/app/src" in content, "Missing ENV PYTHONPATH=/app/src"


class TestDockerfileLightStructure:
    """Dockerfile.base-light must follow same pattern as cpu."""

    def test_uses_python_313_builder(self) -> None:
        content = _read_dockerfile("Dockerfile.base-light")
        assert "python:3.13-slim-bookworm" in content

    def test_has_multi_stage(self) -> None:
        content = _read_dockerfile("Dockerfile.base-light")
        assert "AS builder" in content
        assert "AS runner" in content

    def test_has_tier_label(self) -> None:
        content = _read_dockerfile("Dockerfile.base-light")
        found = any(
            line.strip().startswith('LABEL tier="light"')
            for line in content.splitlines()
        )
        assert found, 'Missing LABEL tier="light"'

    def test_has_nonroot_user(self) -> None:
        content = _read_dockerfile("Dockerfile.base-light")
        assert "useradd" in content
        assert "minivess" in content

    def test_has_healthcheck(self) -> None:
        content = _read_dockerfile("Dockerfile.base-light")
        assert "HEALTHCHECK" in content

    def test_uses_requirements_light(self) -> None:
        content = _read_dockerfile("Dockerfile.base-light")
        assert "requirements-light.txt" in content

    def test_uses_uv_pip_install(self) -> None:
        content = _read_dockerfile("Dockerfile.base-light")
        assert "uv pip install" in content

    def test_no_uv_sync(self) -> None:
        content = _read_dockerfile("Dockerfile.base-light")
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert "uv sync" not in stripped

    def test_has_pythonpath(self) -> None:
        content = _read_dockerfile("Dockerfile.base-light")
        assert "PYTHONPATH=/app/src" in content
