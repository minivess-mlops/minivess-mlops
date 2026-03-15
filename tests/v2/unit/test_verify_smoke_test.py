"""Unit tests for smoke test validation scripts (#637, T3.1).

Tests validate_smoke_test_env.py and verify_smoke_test.py using
monkeypatch — no real cloud connectivity needed.

Run: uv run pytest tests/v2/unit/test_verify_smoke_test.py -v
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

ROOT = Path(__file__).parent.parent.parent.parent


class TestValidateSmokeTestEnv:
    """Test scripts/validate_smoke_test_env.py."""

    def test_script_exists(self) -> None:
        """Script must exist."""
        assert (ROOT / "scripts" / "validate_smoke_test_env.py").exists()

    def test_script_importable(self) -> None:
        """Script must be valid Python."""
        path = ROOT / "scripts" / "validate_smoke_test_env.py"
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_exits_1_when_env_vars_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Returns 1 when required env vars are missing."""
        # Clear all required vars
        for var in (
            "DVC_S3_ENDPOINT_URL",
            "DVC_S3_ACCESS_KEY",
            "DVC_S3_SECRET_KEY",
            "DVC_S3_BUCKET",
            "RUNPOD_API_KEY",
            "MLFLOW_CLOUD_URI",
            "MLFLOW_CLOUD_USERNAME",
            "MLFLOW_CLOUD_PASSWORD",
        ):
            monkeypatch.delenv(var, raising=False)

        from scripts.validate_smoke_test_env import _check_env_vars

        missing = _check_env_vars()
        assert len(missing) > 0


class TestVerifySmokeTest:
    """Test scripts/verify_smoke_test.py."""

    def test_script_exists(self) -> None:
        """Script must exist."""
        assert (ROOT / "scripts" / "verify_smoke_test.py").exists()

    def test_script_importable(self) -> None:
        """Script must be valid Python."""
        path = ROOT / "scripts" / "verify_smoke_test.py"
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


class TestMakefileTargets:
    """Verify Makefile has smoke test targets."""

    def test_makefile_has_smoke_test_targets(self) -> None:
        """Makefile must define smoke test targets."""
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        for target in (
            "smoke-test-preflight",
            "smoke-test-gpu",
            "smoke-test-all",
            "verify-smoke-test",
        ):
            assert f"{target}:" in makefile, f"Makefile missing target: {target}"
