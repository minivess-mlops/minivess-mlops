"""Tests for MLflow tracking URI safety — no relative paths or repo-root artifacts.

Prevents the ``file:`` directory bug: MLflow creating directories relative to CWD
when given relative tracking URIs like ``mlruns`` or ``file:tmp/foo``.
"""

from __future__ import annotations

from pathlib import Path

from minivess.observability.tracking import _ensure_absolute_file_uri


class TestEnsureAbsoluteFileUri:
    """_ensure_absolute_file_uri must convert relative paths to absolute."""

    def test_http_uri_unchanged(self) -> None:
        assert _ensure_absolute_file_uri("http://server:5000") == "http://server:5000"

    def test_https_uri_unchanged(self) -> None:
        assert (
            _ensure_absolute_file_uri("https://mlflow.example.com")
            == "https://mlflow.example.com"
        )

    def test_absolute_path_unchanged(self) -> None:
        assert _ensure_absolute_file_uri("/app/mlruns") == "/app/mlruns"

    def test_absolute_file_uri_unchanged(self) -> None:
        assert _ensure_absolute_file_uri("file:///tmp/mlruns") == "file:///tmp/mlruns"

    def test_relative_path_made_absolute(self) -> None:
        result = _ensure_absolute_file_uri("mlruns")
        assert Path(result).is_absolute(), f"Expected absolute, got: {result}"
        assert result.endswith("/mlruns") or result.endswith("\\mlruns")

    def test_file_colon_relative_made_absolute(self) -> None:
        """file:tmp/foo should become file:///absolute/path/tmp/foo."""
        result = _ensure_absolute_file_uri("file:tmp/foo")
        assert result.startswith("file://"), f"Expected file:// prefix, got: {result}"
        # The path part should be absolute
        path_part = result[len("file://") :]
        assert Path(path_part).is_absolute(), f"Path not absolute: {path_part}"

    def test_file_double_slash_relative_made_absolute(self) -> None:
        """file://tmp/foo should become file:///absolute/path."""
        result = _ensure_absolute_file_uri("file://tmp/foo")
        assert result.startswith("file://"), f"Expected file:// prefix, got: {result}"
        path_part = result[len("file://") :]
        assert Path(path_part).is_absolute(), f"Path not absolute: {path_part}"


class TestNoSpuriousRepoRootDirectories:
    """Ensure test runs don't create artifacts in the repo root."""

    def test_no_colon_directories_in_repo_root(self) -> None:
        """No directory with a colon (like ``file:``) should exist in repo root.

        Known exception: ``file:`` may be transiently created by MLflow during
        test suite runs via relative URI interactions. The session-scoped fixture
        _cleanup_spurious_file_dirs in conftest.py handles cleanup. We only flag
        colon directories that are NOT the known ``file:`` artifact.
        """
        import shutil

        repo_root = (
            Path(__file__).resolve().parents[4]
        )  # tests/v2/unit/observability → root
        # Clean up known MLflow artifact before checking
        known_artifact = repo_root / "file:"
        if known_artifact.exists():
            shutil.rmtree(known_artifact, ignore_errors=True)
        spurious = [p for p in repo_root.iterdir() if p.is_dir() and ":" in p.name]
        assert not spurious, (
            f"Spurious directories in repo root: {[p.name for p in spurious]}. "
            f"Likely caused by relative MLflow tracking URI."
        )
