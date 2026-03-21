"""Tests for atomic checkpoint writes + SHA256 integrity (#714, #707).

Phase A3+A4 of the pre-GCP fixes plan.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn


class _SimpleAdapter(nn.Module):
    """Minimal adapter for checkpoint testing."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3, padding=1)

    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    def save_checkpoint(self, path: Path) -> None:
        from minivess.adapters.base import ModelAdapter

        ModelAdapter.save_checkpoint(self, path)

    def load_checkpoint(self, path: Path) -> None:
        from minivess.adapters.base import ModelAdapter

        ModelAdapter.load_checkpoint(self, path)


class TestAtomicCheckpointWrite:
    """Checkpoints must use sync→rename atomic pattern (#714)."""

    def test_no_tmp_file_after_save(self, tmp_path: Path) -> None:
        """After save, no .tmp file should remain."""
        adapter = _SimpleAdapter()
        ckpt_path = tmp_path / "model.pth"
        adapter.save_checkpoint(ckpt_path)

        tmp_file = ckpt_path.with_suffix(".tmp")
        assert not tmp_file.exists(), ".tmp file should be removed after atomic rename"
        assert ckpt_path.exists(), "Checkpoint file should exist after save"

    def test_checkpoint_loadable_after_save(self, tmp_path: Path) -> None:
        """Saved checkpoint must be loadable."""
        adapter = _SimpleAdapter()
        ckpt_path = tmp_path / "model.pth"
        adapter.save_checkpoint(ckpt_path)

        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        assert "model_state_dict" in loaded


class TestSHA256Sidecar:
    """Checkpoint saves must create SHA256 sidecar (#707)."""

    def test_sha256_sidecar_created(self, tmp_path: Path) -> None:
        """After save, a .sha256 sidecar file must exist."""
        adapter = _SimpleAdapter()
        ckpt_path = tmp_path / "model.pth"
        adapter.save_checkpoint(ckpt_path)

        sha_path = ckpt_path.with_suffix(".pth.sha256")
        assert sha_path.exists(), "SHA256 sidecar must be created"

    def test_sha256_content_matches(self, tmp_path: Path) -> None:
        """Sidecar SHA256 must match actual file hash."""
        import hashlib

        adapter = _SimpleAdapter()
        ckpt_path = tmp_path / "model.pth"
        adapter.save_checkpoint(ckpt_path)

        sha_path = ckpt_path.with_suffix(".pth.sha256")
        expected = sha_path.read_text(encoding="utf-8").strip()
        actual = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
        assert expected == actual

    def test_load_verifies_sha256(self, tmp_path: Path) -> None:
        """Load must verify SHA256 if sidecar exists."""
        adapter = _SimpleAdapter()
        ckpt_path = tmp_path / "model.pth"
        adapter.save_checkpoint(ckpt_path)

        # Corrupt the sidecar
        sha_path = ckpt_path.with_suffix(".pth.sha256")
        sha_path.write_text("badhash", encoding="utf-8")

        with pytest.raises(RuntimeError, match="integrity check FAILED"):
            adapter.load_checkpoint(ckpt_path)

    def test_load_works_without_sidecar(self, tmp_path: Path) -> None:
        """Load must work when no sidecar exists (backward compat)."""
        adapter = _SimpleAdapter()
        ckpt_path = tmp_path / "model.pth"
        torch.save({"model_state_dict": adapter.state_dict()}, ckpt_path)

        # No sidecar — should still load
        adapter2 = _SimpleAdapter()
        adapter2.load_checkpoint(ckpt_path)  # should not raise
