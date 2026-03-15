"""Tests for atomic checkpoint writes.

T0.5: Verify torch.save uses tmp + os.replace pattern for atomicity.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch


class TestAtomicSave:
    """_atomic_torch_save should write to tmp then atomically rename."""

    def test_produces_valid_checkpoint(self, tmp_path: Path) -> None:
        """Atomic save produces a loadable checkpoint file."""
        from minivess.pipeline.checkpoint_utils import atomic_torch_save

        state = {"weight": torch.randn(4, 4), "epoch": 10}
        path = tmp_path / "model.pth"
        atomic_torch_save(state, path)

        assert path.exists()
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        assert loaded["epoch"] == 10
        assert torch.equal(loaded["weight"], state["weight"])

    def test_no_tmp_file_after_success(self, tmp_path: Path) -> None:
        """Temporary file is cleaned up after successful save."""
        from minivess.pipeline.checkpoint_utils import atomic_torch_save

        path = tmp_path / "model.pth"
        atomic_torch_save({"x": 1}, path)

        tmp_path_check = path.with_suffix(".pth.tmp")
        assert not tmp_path_check.exists()

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """atomic_torch_save creates parent directories if missing."""
        from minivess.pipeline.checkpoint_utils import atomic_torch_save

        path = tmp_path / "deep" / "dir" / "model.pth"
        atomic_torch_save({"x": 1}, path)
        assert path.exists()

    def test_original_preserved_on_failure(self, tmp_path: Path) -> None:
        """If save fails, original file is untouched."""
        from minivess.pipeline.checkpoint_utils import atomic_torch_save

        path = tmp_path / "model.pth"
        # Write original
        atomic_torch_save({"epoch": 1}, path)

        # Force a failure during save
        with (
            patch("torch.save", side_effect=OSError("disk full")),
            pytest.raises(OSError, match="disk full"),
        ):
            atomic_torch_save({"epoch": 2}, path)

        # Original should still be readable with epoch=1
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        assert loaded["epoch"] == 1
