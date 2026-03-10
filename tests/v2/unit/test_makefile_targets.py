"""Tests for Makefile targets existence and correctness.

E2E Plan Phase 4, Task T4.3: Verify test-e2e target in Makefile.
"""

from __future__ import annotations

from pathlib import Path


def _read_makefile() -> str:
    """Read the Makefile content."""
    makefile = Path(__file__).resolve().parents[3] / "Makefile"
    return makefile.read_text(encoding="utf-8")


class TestMakefileTargets:
    """Verify Makefile has required test targets."""

    def test_makefile_has_test_e2e_target(self) -> None:
        """Parse Makefile, verify test-e2e target exists."""
        content = _read_makefile()
        assert "test-e2e:" in content, (
            "Makefile missing test-e2e target. Required for full e2e pipeline testing."
        )

    def test_makefile_help_lists_test_e2e(self) -> None:
        """Verify test-e2e appears in help output."""
        content = _read_makefile()
        assert "test-e2e" in content, "Makefile help section doesn't mention test-e2e"

    def test_test_e2e_uses_correct_timeout(self) -> None:
        """Verify test-e2e target has --timeout=3600."""
        content = _read_makefile()
        # Find the test-e2e section
        lines = content.split("\n")
        in_target = False
        timeout_found = False
        for line in lines:
            if line.startswith("test-e2e:"):
                in_target = True
                continue
            if in_target:
                if line.startswith("\t"):
                    if "--timeout=3600" in line:
                        timeout_found = True
                else:
                    break

        assert timeout_found, (
            "test-e2e target missing --timeout=3600. E2E pipeline needs 1 hour timeout."
        )

    def test_makefile_has_test_staging_target(self) -> None:
        """Verify test-staging target exists."""
        content = _read_makefile()
        assert "test-staging:" in content

    def test_makefile_has_test_prod_target(self) -> None:
        """Verify test-prod target exists."""
        content = _read_makefile()
        assert "test-prod:" in content

    def test_makefile_has_test_gpu_target(self) -> None:
        """Verify test-gpu target exists."""
        content = _read_makefile()
        assert "test-gpu:" in content
