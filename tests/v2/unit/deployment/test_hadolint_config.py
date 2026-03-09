"""Tests for Hadolint Dockerfile linting configuration.

Rule #16: No regex. Parse with yaml.safe_load() and str methods.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parent.parent.parent.parent.parent
DOCKERFILES_DIR = ROOT / "deployment" / "docker"


def _hadolint_installed() -> bool:
    try:
        subprocess.run(
            ["hadolint", "--version"],
            capture_output=True,
            check=False,
            timeout=5,
        )
        return True
    except FileNotFoundError:
        return False


_skip_no_hadolint = pytest.mark.skipif(
    not _hadolint_installed(), reason="hadolint not installed"
)


def test_hadolint_config_exists() -> None:
    assert (ROOT / ".hadolint.yaml").exists(), ".hadolint.yaml not found at repo root"


def test_pre_commit_includes_hadolint() -> None:
    pre_commit_path = ROOT / ".pre-commit-config.yaml"
    assert pre_commit_path.exists(), ".pre-commit-config.yaml not found"
    cfg = yaml.safe_load(pre_commit_path.read_text(encoding="utf-8"))

    repos = cfg.get("repos", [])
    hook_ids = []
    for repo in repos:
        for hook in repo.get("hooks", []):
            hook_ids.append(hook.get("id", ""))

    assert "hadolint" in hook_ids, (
        "hadolint hook not found in .pre-commit-config.yaml. "
        "Add it from https://github.com/hadolint/hadolint"
    )


@_skip_no_hadolint
def test_dockerfiles_pass_hadolint() -> None:
    """All Dockerfiles in deployment/docker/ must pass hadolint."""
    dockerfiles = list(DOCKERFILES_DIR.glob("Dockerfile*"))
    assert dockerfiles, f"No Dockerfiles found in {DOCKERFILES_DIR}"

    failures = []
    for dockerfile in sorted(dockerfiles):
        result = subprocess.run(
            ["hadolint", "--config", str(ROOT / ".hadolint.yaml"), str(dockerfile)],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if result.returncode != 0:
            failures.append(f"{dockerfile.name}:\n{result.stdout}{result.stderr}")

    assert not failures, "Hadolint failures:\n" + "\n\n".join(failures)
