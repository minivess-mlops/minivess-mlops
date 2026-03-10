"""Tests for PEP 735 [dependency-groups] tier isolation.

Verifies that pyproject.toml defines cpu and light dependency groups
for the 3-tier Docker base image hierarchy, and that generated
requirements files exclude GPU dependencies.

Rule #16: No regex. Use tomllib and str methods.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent.parent
PYPROJECT = ROOT / "pyproject.toml"
DOCKER_DIR = ROOT / "deployment" / "docker"

# Packages that MUST NOT appear in non-GPU tiers
_GPU_PACKAGES = frozenset(
    {
        "torch",
        "torchvision",
        "torchio",
        "torchmetrics",
        "monai",
    }
)


def _load_dependency_groups() -> dict[str, list[str]]:
    """Load [dependency-groups] from pyproject.toml."""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return data.get("dependency-groups", {})


class TestDependencyGroupsExist:
    """pyproject.toml must have cpu and light dependency groups."""

    def test_cpu_group_exists(self) -> None:
        groups = _load_dependency_groups()
        assert "cpu" in groups, (
            "[dependency-groups] missing 'cpu' group in pyproject.toml"
        )

    def test_light_group_exists(self) -> None:
        groups = _load_dependency_groups()
        assert "light" in groups, (
            "[dependency-groups] missing 'light' group in pyproject.toml"
        )


class TestCpuGroupContents:
    """CPU group must have stats/analytics deps but NO GPU deps."""

    def test_cpu_has_scipy(self) -> None:
        groups = _load_dependency_groups()
        cpu_deps = " ".join(groups.get("cpu", []))
        assert "scipy" in cpu_deps, "cpu group missing scipy"

    def test_cpu_has_pandas(self) -> None:
        groups = _load_dependency_groups()
        cpu_deps = " ".join(groups.get("cpu", []))
        assert "pandas" in cpu_deps, "cpu group missing pandas"

    def test_cpu_has_matplotlib(self) -> None:
        groups = _load_dependency_groups()
        cpu_deps = " ".join(groups.get("cpu", []))
        assert "matplotlib" in cpu_deps, "cpu group missing matplotlib"

    def test_cpu_has_duckdb(self) -> None:
        groups = _load_dependency_groups()
        cpu_deps = " ".join(groups.get("cpu", []))
        assert "duckdb" in cpu_deps, "cpu group missing duckdb"

    def test_cpu_has_mlflow(self) -> None:
        groups = _load_dependency_groups()
        cpu_deps = " ".join(groups.get("cpu", []))
        assert "mlflow" in cpu_deps, "cpu group missing mlflow"

    def test_cpu_excludes_gpu_packages(self) -> None:
        groups = _load_dependency_groups()
        cpu_deps = groups.get("cpu", [])
        for dep in cpu_deps:
            pkg_name = (
                dep.split(">")[0].split("<")[0].split("=")[0].split("[")[0].strip()
            )
            assert pkg_name not in _GPU_PACKAGES, (
                f"cpu group must NOT contain GPU package: {pkg_name}"
            )


class TestLightGroupContents:
    """Light group must have orchestration/web deps but NO GPU or heavy deps."""

    def test_light_has_prefect(self) -> None:
        groups = _load_dependency_groups()
        light_deps = " ".join(groups.get("light", []))
        assert "prefect" in light_deps, "light group missing prefect"

    def test_light_has_fastapi(self) -> None:
        groups = _load_dependency_groups()
        light_deps = " ".join(groups.get("light", []))
        assert "fastapi" in light_deps, "light group missing fastapi"

    def test_light_has_mlflow(self) -> None:
        groups = _load_dependency_groups()
        light_deps = " ".join(groups.get("light", []))
        assert "mlflow" in light_deps, "light group missing mlflow"

    def test_light_has_pydantic(self) -> None:
        groups = _load_dependency_groups()
        light_deps = " ".join(groups.get("light", []))
        assert "pydantic" in light_deps, "light group missing pydantic"

    def test_light_excludes_gpu_packages(self) -> None:
        groups = _load_dependency_groups()
        light_deps = groups.get("light", [])
        for dep in light_deps:
            pkg_name = (
                dep.split(">")[0].split("<")[0].split("=")[0].split("[")[0].strip()
            )
            assert pkg_name not in _GPU_PACKAGES, (
                f"light group must NOT contain GPU package: {pkg_name}"
            )


class TestRequirementsFiles:
    """Generated requirements files must exist and exclude torch."""

    def test_requirements_cpu_exists(self) -> None:
        path = DOCKER_DIR / "requirements-cpu.txt"
        assert path.exists(), (
            f"requirements-cpu.txt not found at {path}. "
            "Run: uv export --frozen --only-group cpu --output-file deployment/docker/requirements-cpu.txt"
        )

    def test_requirements_light_exists(self) -> None:
        path = DOCKER_DIR / "requirements-light.txt"
        assert path.exists(), (
            f"requirements-light.txt not found at {path}. "
            "Run: uv export --frozen --only-group light --output-file deployment/docker/requirements-light.txt"
        )

    def test_requirements_cpu_no_torch(self) -> None:
        path = DOCKER_DIR / "requirements-cpu.txt"
        if not path.exists():
            return  # covered by test_requirements_cpu_exists
        content = path.read_text(encoding="utf-8")
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            pkg_name = stripped.split("=")[0].split(">")[0].split("<")[0].strip()
            assert pkg_name != "torch", (
                f"requirements-cpu.txt contains torch: {stripped}"
            )

    def test_requirements_light_no_torch(self) -> None:
        path = DOCKER_DIR / "requirements-light.txt"
        if not path.exists():
            return  # covered by test_requirements_light_exists
        content = path.read_text(encoding="utf-8")
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            pkg_name = stripped.split("=")[0].split(">")[0].split("<")[0].strip()
            assert pkg_name != "torch", (
                f"requirements-light.txt contains torch: {stripped}"
            )
