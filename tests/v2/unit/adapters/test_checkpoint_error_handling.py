"""Tests for checkpoint load error handling across SAM3 adapters.

T0.3: Verify load_checkpoint raises FileNotFoundError for missing paths.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from minivess.adapters.base import ModelAdapter


class _MinimalAdapter(ModelAdapter):
    """Minimal adapter for testing base class checkpoint methods."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(4, 2)

    def forward(self, images: torch.Tensor, **kwargs):  # type: ignore[override]
        return self.net(images)

    def get_config(self):  # type: ignore[override]
        return None

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TestCheckpointErrorHandling:
    """Checkpoint loading should fail clearly for missing files."""

    def test_load_nonexistent_raises_file_not_found(self, tmp_path: Path) -> None:
        """load_checkpoint raises FileNotFoundError for missing path."""
        adapter = _MinimalAdapter()
        missing = tmp_path / "nonexistent.pth"
        with pytest.raises(FileNotFoundError, match="No checkpoint"):
            adapter.load_checkpoint(missing)

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """save_checkpoint + load_checkpoint roundtrip works."""
        adapter = _MinimalAdapter()
        path = tmp_path / "test.pth"
        adapter.save_checkpoint(path)
        assert path.exists()

        adapter2 = _MinimalAdapter()
        adapter2.load_checkpoint(path)

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        """save_checkpoint creates parent directories."""
        adapter = _MinimalAdapter()
        path = tmp_path / "deep" / "nested" / "checkpoint.pth"
        adapter.save_checkpoint(path)
        assert path.exists()
