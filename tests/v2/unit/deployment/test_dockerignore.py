"""Tests for .dockerignore correctness.

A missing .env in build context can leak credentials into image layers.
Rule #16: No regex. Use Path.read_text().splitlines() and str methods.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent.parent


def _lines() -> set[str]:
    di_path = ROOT / ".dockerignore"
    return set(di_path.read_text(encoding="utf-8").splitlines())


def test_dockerignore_exists() -> None:
    assert (ROOT / ".dockerignore").exists(), ".dockerignore not found at repo root"


def test_dockerignore_excludes_env() -> None:
    lines = _lines()
    assert ".env" in lines, (
        ".dockerignore must contain '.env' to prevent credential leaks"
    )


def test_dockerignore_excludes_git() -> None:
    lines = _lines()
    assert ".git" in lines, ".dockerignore must contain '.git'"


def test_dockerignore_excludes_mlruns() -> None:
    lines = _lines()
    assert "mlruns/" in lines, (
        ".dockerignore must exclude 'mlruns/' to reduce build context"
    )


def test_dockerignore_excludes_pycache() -> None:
    lines = _lines()
    assert "__pycache__/" in lines, ".dockerignore must exclude '__pycache__/'"


def test_dockerignore_excludes_pytest_cache() -> None:
    lines = _lines()
    assert ".pytest_cache/" in lines, ".dockerignore must exclude '.pytest_cache/'"
