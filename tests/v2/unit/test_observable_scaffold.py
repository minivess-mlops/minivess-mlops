"""Tests for Observable Framework dashboard scaffolding (Phase 4).

Validates the project structure and configuration files exist.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
OBSERVABLE_DIR = REPO_ROOT / "observable"


class TestObservableScaffold:
    """Tests for the Observable Framework project structure."""

    def test_observable_dir_exists(self) -> None:
        """observable/ directory exists in repo root."""
        assert OBSERVABLE_DIR.is_dir(), f"Missing {OBSERVABLE_DIR}"

    def test_package_json_exists(self) -> None:
        """observable/package.json exists."""
        assert (OBSERVABLE_DIR / "package.json").is_file()

    def test_config_exists(self) -> None:
        """observable/observablehq.config.js exists."""
        assert (OBSERVABLE_DIR / "observablehq.config.js").is_file()

    def test_src_dir_exists(self) -> None:
        """observable/src/ directory exists."""
        assert (OBSERVABLE_DIR / "src").is_dir()

    def test_index_page_exists(self) -> None:
        """observable/src/index.md exists."""
        assert (OBSERVABLE_DIR / "src" / "index.md").is_file()

    def test_data_dir_exists(self) -> None:
        """observable/src/data/ directory exists."""
        assert (OBSERVABLE_DIR / "src" / "data").is_dir()
