"""Tests for Dockerfile.base multi-stage H3+H4 compliance.

These tests define the desired state. They FAIL until Dockerfile.base is updated.
Rule #16: No regex. Use str methods: splitlines(), partition(), split(), "in" operator.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent.parent
DOCKERFILE_BASE = ROOT / "deployment" / "docker" / "Dockerfile.base"


def _content() -> str:
    return DOCKERFILE_BASE.read_text(encoding="utf-8")


def _from_lines() -> list[str]:
    """Extract all FROM lines (case-insensitive start, strip comments)."""
    lines = []
    for line in _content().splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("FROM"):
            lines.append(stripped)
    return lines


def _stages() -> list[str]:
    """Return content split at FROM boundaries.

    Each element starts with a FROM line.
    Pre-FROM preamble (comments) is excluded.
    stages[0] = builder stage, stages[1] = runner stage.
    """
    content = _content()
    stages = []
    current: list[str] = []
    in_stage = False
    for line in content.splitlines():
        if line.strip().upper().startswith("FROM"):
            if in_stage and current:
                stages.append("\n".join(current))
            current = [line]
            in_stage = True
        elif in_stage:
            current.append(line)
    if in_stage and current:
        stages.append("\n".join(current))
    return stages


def test_dockerfile_base_uses_two_stages() -> None:
    """Dockerfile.base must have at least 2 FROM lines (builder + runner)."""
    from_lines = _from_lines()
    assert len(from_lines) >= 2, (
        f"Expected >=2 FROM lines (multi-stage build), got {len(from_lines)}. "
        f"Dockerfile.base must use builder+runner stages (H4)."
    )


def test_dockerfile_base_builder_uses_devel() -> None:
    """First FROM (builder) must use the devel CUDA image."""
    from_lines = _from_lines()
    assert from_lines, "No FROM lines found in Dockerfile.base"
    first = from_lines[0]
    assert "devel" in first, (
        f"Builder FROM must reference 'devel' image for CUDA headers. Got: {first!r}"
    )


def test_dockerfile_base_builder_has_as_builder() -> None:
    """Builder stage must be named 'builder' via AS clause."""
    from_lines = _from_lines()
    assert from_lines, "No FROM lines found"
    first_upper = from_lines[0].upper()
    assert "AS BUILDER" in first_upper, (
        f"First FROM must have 'AS builder'. Got: {from_lines[0]!r}"
    )


def test_dockerfile_base_runner_uses_runtime() -> None:
    """Second FROM (runner) must use runtime image, not devel."""
    from_lines = _from_lines()
    assert len(from_lines) >= 2, "Expected multi-stage FROM"
    second = from_lines[1]
    assert "runtime" in second, f"Runner FROM must use 'runtime' image. Got: {second!r}"
    assert "devel" not in second, f"Runner FROM must NOT use 'devel'. Got: {second!r}"


def test_dockerfile_base_no_git_in_runner() -> None:
    """Runner stage must not install git (belongs in builder only)."""
    stages = _stages()
    if len(stages) < 2:
        raise AssertionError("Expected multi-stage Dockerfile")
    runner_stage = stages[1]  # Runner is stages[1] (builder=stages[0])
    # Check no apt-get install of git in runner stage
    for line in runner_stage.splitlines():
        stripped = line.strip()
        if "apt-get install" in stripped and "git" in stripped:
            raise AssertionError(
                f"Runner stage contains apt-get install of 'git'. "
                f"git belongs in builder stage only. Line: {stripped!r}"
            )


def test_dockerfile_base_no_uv_in_runner() -> None:
    """Runner stage must not COPY uv binary."""
    stages = _stages()
    if len(stages) < 2:
        raise AssertionError("Expected multi-stage Dockerfile")
    runner_stage = stages[1]
    for line in runner_stage.splitlines():
        if "COPY" in line and "astral-sh/uv" in line:
            raise AssertionError(
                f"Runner stage COPYs uv from ghcr.io/astral-sh/uv. "
                f"uv belongs in builder only — runner only gets the .venv. "
                f"Line: {line.strip()!r}"
            )


def test_dockerfile_base_no_pip_in_runner() -> None:
    """Runner stage must not install python3-pip."""
    stages = _stages()
    if len(stages) < 2:
        raise AssertionError("Expected multi-stage Dockerfile")
    runner_stage = stages[1]
    for line in runner_stage.splitlines():
        stripped = line.strip()
        if "apt-get install" in stripped and "python3-pip" in stripped:
            raise AssertionError(
                f"Runner stage installs python3-pip. "
                f"pip must not be in the production image. Line: {stripped!r}"
            )


def test_dockerfile_base_has_oci_labels() -> None:
    """Dockerfile.base must contain OCI standard image labels."""
    content = _content()
    assert "org.opencontainers.image.source" in content, (
        "Missing OCI label: org.opencontainers.image.source"
    )
    assert "org.opencontainers.image.revision" in content, (
        "Missing OCI label: org.opencontainers.image.revision"
    )


def test_dockerfile_base_has_buildkit_cache_mount() -> None:
    """Dockerfile.base must use BuildKit cache mount for uv."""
    content = _content()
    assert "--mount=type=cache,target=/root/.cache/uv" in content, (
        "Missing BuildKit cache mount for uv. "
        "Add: RUN --mount=type=cache,target=/root/.cache/uv \\"
    )


def test_dockerfile_base_uv_sync_frozen() -> None:
    """uv sync must use --frozen flag to treat uv.lock as immutable."""
    content = _content()
    # Check that some RUN line contains both uv sync and --frozen
    for line in content.splitlines():
        stripped = line.strip()
        if "uv sync" in stripped and "--frozen" in stripped:
            return
    raise AssertionError(
        "No 'uv sync --frozen' found in Dockerfile.base. "
        "Add --frozen to uv sync to enforce lock file immutability."
    )


def test_dockerfile_base_pythondontwritebytecode() -> None:
    """Dockerfile.base must set PYTHONDONTWRITEBYTECODE env var."""
    content = _content()
    assert "PYTHONDONTWRITEBYTECODE" in content, (
        "Missing ENV PYTHONDONTWRITEBYTECODE=1 in Dockerfile.base"
    )


def test_dockerfile_base_uv_lock_exact() -> None:
    """uv.lock must appear as exact filename, not glob 'uv.lock*'."""
    content = _content()
    assert "uv.lock*" not in content, (
        "Found 'uv.lock*' glob in Dockerfile.base. "
        "Use exact 'uv.lock' — glob is unnecessary and surprising."
    )


def test_dockerfile_base_venv_group_only() -> None:
    """chmod for .venv must use g+rX (group-readable), not a+rX (world-readable)."""
    content = _content()
    # Check that no chmod uses a+rX on venv (world-readable is too permissive)
    for line in content.splitlines():
        stripped = line.strip()
        if "chmod" in stripped and "a+rX" in stripped and ".venv" in stripped:
            raise AssertionError(
                f"Found world-readable chmod (a+rX) on .venv. "
                f"Use g+rX (group-readable only). Line: {stripped!r}"
            )


def test_dockerfile_base_has_uv_env_vars() -> None:
    """Dockerfile.base must set all three required uv env vars."""
    content = _content()
    assert "UV_COMPILE_BYTECODE" in content, "Missing UV_COMPILE_BYTECODE env var"
    assert "UV_LINK_MODE" in content, "Missing UV_LINK_MODE env var"
    assert "UV_PYTHON_DOWNLOADS" in content, "Missing UV_PYTHON_DOWNLOADS env var"
