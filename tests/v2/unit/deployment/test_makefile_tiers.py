"""Tests for Makefile tier-aware build, scan, and requirements targets.

Rule #16: No regex. Use str methods only.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent.parent
MAKEFILE = ROOT / "Makefile"


def _makefile_content() -> str:
    return MAKEFILE.read_text(encoding="utf-8")


class TestMakefileTierTargets:
    """Makefile must have tier-aware build and management targets."""

    def test_has_build_base_cpu(self) -> None:
        content = _makefile_content()
        assert "build-base-cpu:" in content, "Makefile missing 'build-base-cpu' target"

    def test_has_build_base_light(self) -> None:
        content = _makefile_content()
        assert "build-base-light:" in content, (
            "Makefile missing 'build-base-light' target"
        )

    def test_has_build_bases(self) -> None:
        content = _makefile_content()
        assert "build-bases:" in content, "Makefile missing 'build-bases' target"

    def test_scan_references_all_base_images(self) -> None:
        content = _makefile_content()
        assert "minivess-base-cpu" in content, (
            "Makefile scan should reference minivess-base-cpu"
        )
        assert "minivess-base-light" in content, (
            "Makefile scan should reference minivess-base-light"
        )

    def test_has_requirements_tiers(self) -> None:
        content = _makefile_content()
        assert "requirements-tiers:" in content, (
            "Makefile missing 'requirements-tiers' target"
        )
