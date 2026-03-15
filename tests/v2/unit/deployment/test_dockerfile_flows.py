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
# Dockerfile.dashboard-ui is React/nginx frontend (not Python, no base inheritance) — exclude
_EXCLUDED = {
    "Dockerfile.base",
    "Dockerfile.base-cpu",
    "Dockerfile.base-light",
    "Dockerfile.mlflow",
    "Dockerfile.mlflow-gcp",
    "Dockerfile.dashboard-ui",
}


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


_VALID_BASES = (
    "FROM minivess-base:latest",
    "FROM minivess-base-cpu:latest",
    "FROM minivess-base-light:latest",
)


def test_all_flow_dockerfiles_inherit_from_base() -> None:
    """All flow Dockerfiles must use a valid minivess base image."""
    failures = []
    for df in _flow_dockerfiles():
        content = df.read_text(encoding="utf-8")
        has_base = any(
            any(line.strip().startswith(base) for base in _VALID_BASES)
            for line in content.splitlines()
        )
        if not has_base:
            failures.append(df.name)

    assert not failures, (
        f"Flow Dockerfiles do not inherit from a minivess base: {failures}. "
        f"Use: FROM minivess-base:latest"
    )


# Canonical tier mapping — single source of truth for which base each flow uses
TIER_MAPPING: dict[str, str] = {
    "Dockerfile.train": "minivess-base:latest",
    "Dockerfile.hpo": "minivess-base:latest",
    "Dockerfile.post_training": "minivess-base:latest",
    "Dockerfile.analyze": "minivess-base:latest",
    "Dockerfile.deploy": "minivess-base:latest",
    "Dockerfile.data": "minivess-base:latest",
    "Dockerfile.annotation": "minivess-base:latest",
    "Dockerfile.monailabel": "minivess-base:latest",
    "Dockerfile.acquisition": "minivess-base:latest",
    "Dockerfile.biostatistics": "minivess-base-cpu:latest",
    "Dockerfile.dashboard": "minivess-base-light:latest",
    "Dockerfile.dashboard-api": "minivess-base-light:latest",
    "Dockerfile.pipeline": "minivess-base-light:latest",
}


def test_flow_dockerfiles_use_correct_tier() -> None:
    """Each flow Dockerfile must use its tier-correct base image."""
    failures = []
    for dockerfile_name, expected_base in TIER_MAPPING.items():
        path = DOCKERFILES_DIR / dockerfile_name
        if not path.exists():
            failures.append(f"{dockerfile_name}: file not found")
            continue
        content = path.read_text(encoding="utf-8")
        has_correct_base = any(
            line.strip().startswith(f"FROM {expected_base}")
            for line in content.splitlines()
        )
        if not has_correct_base:
            failures.append(f"{dockerfile_name}: expected FROM {expected_base}")

    assert not failures, "Flow Dockerfiles use wrong base image:\n" + "\n".join(
        failures
    )


def test_flow_dockerfiles_have_warning_suppression() -> None:
    """Flow Dockerfiles with Python CMD must have -W flags for noise suppression (DG1.7).

    MetricsReloaded SyntaxWarnings fire during compilation, FutureWarning from CUDA,
    UserWarning from MONAI — all non-actionable. Must be suppressed at interpreter level.
    """
    failures = []
    for df in _flow_dockerfiles():
        content = df.read_text(encoding="utf-8")
        # Only check Dockerfiles that have python CMD
        has_python_cmd = any(
            "python" in line and "CMD" in line and not line.strip().startswith("#")
            for line in content.splitlines()
        )
        if not has_python_cmd:
            continue
        has_w_flag = "-W" in content
        if not has_w_flag:
            failures.append(df.name)

    assert not failures, (
        f"Flow Dockerfiles with Python CMD missing -W warning flags (DG1.7): {failures}. "
        "Add: -W ignore::SyntaxWarning -W ignore::FutureWarning"
    )
