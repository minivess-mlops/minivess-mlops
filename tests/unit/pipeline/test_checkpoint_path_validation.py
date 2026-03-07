"""Tests for checkpoint path validation in SegmentationTrainer (T-02).

The trainer must reject repo-relative checkpoint paths and only accept:
- Docker paths (/app/checkpoints/...)
- pytest tmp_path
- Explicitly allowed via MINIVESS_ALLOW_HOST=1

References:
  - docs/planning/minivess-vision-enforcement-plan-execution.xml (T-02)
  - CLAUDE.md Rule #18 (volume mounts), Rule #19 (STOP protocol)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from minivess.pipeline.trainer import validate_checkpoint_path


class TestCheckpointPathValidation:
    """Verify checkpoint path validation logic."""

    def test_rejects_repo_relative_path(self) -> None:
        """checkpoint_dir=Path('checkpoints') -> ValueError."""
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ValueError, match="volume-mounted"),
        ):
            validate_checkpoint_path(Path("checkpoints"))

    def test_rejects_dot_relative_path(self) -> None:
        """checkpoint_dir=Path('./checkpoints') -> ValueError."""
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ValueError, match="volume-mounted"),
        ):
            validate_checkpoint_path(Path("./checkpoints"))

    def test_rejects_repo_absolute_path(self) -> None:
        """Absolute path under repo root -> ValueError."""
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ValueError, match="volume-mounted"),
        ):
            validate_checkpoint_path(Path("/home/user/repo/checkpoints"))

    def test_accepts_docker_path(self) -> None:
        """checkpoint_dir=Path('/app/checkpoints/fold_0') -> OK."""
        with patch.dict("os.environ", {}, clear=True):
            validate_checkpoint_path(Path("/app/checkpoints/fold_0"))

    def test_accepts_pytest_tmp_path(self, tmp_path: Path) -> None:
        """checkpoint_dir=tmp_path / 'fold_0' -> OK."""
        with patch.dict("os.environ", {}, clear=True):
            validate_checkpoint_path(tmp_path / "fold_0")

    def test_accepts_with_allow_host(self) -> None:
        """MINIVESS_ALLOW_HOST=1 -> any path OK."""
        with patch.dict("os.environ", {"MINIVESS_ALLOW_HOST": "1"}, clear=True):
            validate_checkpoint_path(Path("checkpoints"))  # Should not raise

    def test_accepts_tmp_path_like(self) -> None:
        """Paths containing '/tmp/' are accepted (pytest tmp dirs)."""
        with patch.dict("os.environ", {}, clear=True):
            validate_checkpoint_path(Path("/tmp/pytest-123/fold_0"))
