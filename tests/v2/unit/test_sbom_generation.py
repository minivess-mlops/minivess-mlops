"""Tests for CycloneDX SBOM generation from Python environment.

Phase 2 (P1) of fda-insights-second-pass-executable.xml.
Validates that cyclonedx-bom is installed and can produce valid CycloneDX JSON.
"""

from __future__ import annotations

import json
import subprocess

import pytest

pytest.importorskip("cyclonedx", reason="cyclonedx not installed")


class TestCycloneDXInstalled:
    """CycloneDX SBOM generator must be importable."""

    def test_cyclonedx_bom_importable(self) -> None:
        """cyclonedx-bom package must be installed."""
        import cyclonedx  # noqa: F401

    def test_cyclonedx_cli_available(self) -> None:
        """cyclonedx-py CLI must be available via uv run."""
        result = subprocess.run(
            ["uv", "run", "cyclonedx-py", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"cyclonedx-py --help failed: {result.stderr}"


class TestCycloneDXOutput:
    """CycloneDX must produce valid JSON with expected schema fields."""

    def test_cyclonedx_produces_json(self, tmp_path: object) -> None:
        """cyclonedx-py environment produces valid CycloneDX JSON."""
        from pathlib import Path

        outfile = Path(str(tmp_path)) / "sbom.json"
        result = subprocess.run(
            [
                "uv",
                "run",
                "cyclonedx-py",
                "environment",
                "--output-format",
                "json",
                "-o",
                str(outfile),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"cyclonedx-py failed: {result.stderr}"
        assert outfile.exists(), "SBOM JSON file not created"

        data = json.loads(outfile.read_text(encoding="utf-8"))
        assert "bomFormat" in data, "Missing bomFormat field"
        assert data["bomFormat"] == "CycloneDX", f"Wrong format: {data['bomFormat']}"
        assert "specVersion" in data, "Missing specVersion field"
        assert "components" in data, "Missing components field"
        assert len(data["components"]) > 10, (
            f"Too few components ({len(data['components'])}); expected >10 for a real project"
        )
