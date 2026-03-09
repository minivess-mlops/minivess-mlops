"""Tests for per-flow Dockerfile compliance.

Flow Dockerfiles must:
- Inherit from minivess-base:latest (never from upstream cuda directly)
- Have LABEL flow= for traceability
- Have LABEL description=
- Never run apt-get (all system deps belong in base)
- Never run uv sync (all Python deps belong in base)

Rule #16: No regex. Use str methods and Path operations.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent.parent
DOCKERFILES_DIR = ROOT / "deployment" / "docker"

# Dockerfile.base is the base image, not a flow — exclude from flow checks
# Dockerfile.mlflow is infrastructure (not a Prefect flow) — exclude
_EXCLUDED = {"Dockerfile.base", "Dockerfile.mlflow"}


def _flow_dockerfiles() -> list[Path]:
    """Return all per-flow Dockerfiles (excluding Dockerfile.base)."""
    return [
        p
        for p in sorted(DOCKERFILES_DIR.glob("Dockerfile.*"))
        if p.name not in _EXCLUDED
    ]


def test_flow_dockerfiles_exist() -> None:
    files = _flow_dockerfiles()
    assert len(files) >= 5, f"Expected >=5 flow Dockerfiles, got {len(files)}"


def test_all_flow_dockerfiles_have_flow_label() -> None:
    """Every flow Dockerfile must have LABEL flow= for traceability."""
    failures = []
    for df in _flow_dockerfiles():
        content = df.read_text(encoding="utf-8")
        has_flow = any(
            line.strip().startswith("LABEL flow=") for line in content.splitlines()
        )
        if not has_flow:
            failures.append(df.name)

    assert not failures, (
        f"Missing 'LABEL flow=...' in: {failures}. Add: LABEL flow=\"{{flow_name}}\""
    )


def test_all_flow_dockerfiles_have_description_label() -> None:
    """Every flow Dockerfile must have LABEL description=."""
    failures = []
    for df in _flow_dockerfiles():
        content = df.read_text(encoding="utf-8")
        has_desc = any(
            line.strip().startswith("LABEL description=")
            for line in content.splitlines()
        )
        if not has_desc:
            failures.append(df.name)

    assert not failures, (
        f"Missing 'LABEL description=...' in: {failures}. "
        f'Add: LABEL description="{{description}}"'
    )


def test_no_flow_dockerfile_runs_apt() -> None:
    """Flow Dockerfiles must not run apt-get (all system deps belong in base)."""
    failures = []
    for df in _flow_dockerfiles():
        content = df.read_text(encoding="utf-8")
        for line in content.splitlines():
            stripped = line.strip()
            if "apt-get" in stripped and not stripped.startswith("#"):
                failures.append(f"{df.name}: {stripped!r}")
                break

    assert not failures, (
        "Flow Dockerfiles contain apt-get (system deps belong in Dockerfile.base):\n"
        + "\n".join(failures)
    )


def test_no_flow_dockerfile_runs_uv() -> None:
    """Flow Dockerfiles must not run uv sync (Python deps belong in base)."""
    failures = []
    for df in _flow_dockerfiles():
        content = df.read_text(encoding="utf-8")
        for line in content.splitlines():
            stripped = line.strip()
            if "uv sync" in stripped and not stripped.startswith("#"):
                failures.append(f"{df.name}: {stripped!r}")
                break

    assert not failures, (
        "Flow Dockerfiles contain 'uv sync' (Python deps belong in Dockerfile.base):\n"
        + "\n".join(failures)
    )


def test_all_flow_dockerfiles_inherit_from_base() -> None:
    """All flow Dockerfiles must use 'FROM minivess-base:latest'."""
    failures = []
    for df in _flow_dockerfiles():
        content = df.read_text(encoding="utf-8")
        has_base = any(
            line.strip() == "FROM minivess-base:latest" for line in content.splitlines()
        )
        if not has_base:
            # Also accept FROM with trailing comment
            has_base = any(
                line.strip().startswith("FROM minivess-base:latest")
                for line in content.splitlines()
            )
        if not has_base:
            failures.append(df.name)

    assert not failures, (
        f"Flow Dockerfiles do not inherit from minivess-base:latest: {failures}. "
        f"Use: FROM minivess-base:latest"
    )
