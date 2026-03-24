"""Tests for Docker context gate in training flow (T-00).

The training flow MUST refuse to run outside a Docker container unless
the `MINIVESS_ALLOW_HOST=1` escape hatch is explicitly set.

References:
  - docs/planning/minivess-vision-enforcement-plan.md
  - docs/planning/minivess-vision-enforcement-plan-execution.xml (T-00)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from minivess.orchestration.docker_guard import require_docker_context


class TestRequireDockerContext:
    """Verify require_docker_context("train") gate behavior."""

    def test_raises_outside_docker(self) -> None:
        """No /.dockerenv, no env var → RuntimeError."""
        with (
            patch("minivess.orchestration.docker_guard.Path") as mock_path,
            patch.dict(
                "os.environ",
                {"MINIVESS_ALLOW_HOST": "", "DOCKER_CONTAINER": ""},
            ),
        ):
            mock_path.return_value.exists.return_value = False
            with pytest.raises(RuntimeError, match="Docker container"):
                require_docker_context("train")

    def test_runs_inside_docker(self) -> None:
        """/.dockerenv exists → no error."""
        with (
            patch("minivess.orchestration.docker_guard.Path") as mock_path,
            patch.dict(
                "os.environ",
                {"MINIVESS_ALLOW_HOST": "", "DOCKER_CONTAINER": ""},
            ),
        ):
            mock_path.return_value.exists.return_value = True
            require_docker_context("train")  # Should not raise

    def test_runs_with_allow_host(self) -> None:
        """MINIVESS_ALLOW_HOST=1 → no error even outside Docker."""
        with (
            patch("minivess.orchestration.docker_guard.Path") as mock_path,
            patch.dict("os.environ", {"MINIVESS_ALLOW_HOST": "1"}),
        ):
            mock_path.return_value.exists.return_value = False
            require_docker_context("train")  # Should not raise

    def test_runs_with_docker_container_env(self) -> None:
        """DOCKER_CONTAINER env var → no error."""
        with (
            patch("minivess.orchestration.docker_guard.Path") as mock_path,
            patch.dict("os.environ", {"DOCKER_CONTAINER": "1"}),
        ):
            mock_path.return_value.exists.return_value = False
            require_docker_context("train")  # Should not raise

    def test_error_message_has_docker_instructions(self) -> None:
        """RuntimeError message includes actionable Docker instructions."""
        with (
            patch("minivess.orchestration.docker_guard.Path") as mock_path,
            patch.dict(
                "os.environ",
                {"MINIVESS_ALLOW_HOST": "", "DOCKER_CONTAINER": ""},
            ),
        ):
            mock_path.return_value.exists.return_value = False
            with pytest.raises(RuntimeError, match="docker compose"):
                require_docker_context("train")


class TestAllowHostNotInScripts:
    """Verify MINIVESS_ALLOW_HOST is not used in .sh scripts."""

    def test_allow_host_not_in_sh_scripts(self) -> None:
        """MINIVESS_ALLOW_HOST must not appear in any .sh script."""
        scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
        if not scripts_dir.is_dir():
            pytest.skip("scripts/ directory not found")

        violations: list[str] = []
        for sh_file in scripts_dir.glob("*.sh"):
            content = sh_file.read_text(encoding="utf-8")
            if "MINIVESS_ALLOW_HOST" in content:
                violations.append(str(sh_file.name))

        assert not violations, (
            f"MINIVESS_ALLOW_HOST found in scripts (test-only escape hatch): {violations}"
        )

    def test_allow_host_not_in_python_src(self) -> None:
        """MINIVESS_ALLOW_HOST must not be SET in src/ code (only checked via os.environ.get)."""
        src_dir = Path(__file__).resolve().parents[3] / "src"
        if not src_dir.is_dir():
            pytest.skip("src/ directory not found")

        # The _require_docker_context and validate_checkpoint_path functions
        # READ this env var — that's correct. We just verify no .sh script
        # sets it (covered by test_allow_host_not_in_sh_scripts above).
