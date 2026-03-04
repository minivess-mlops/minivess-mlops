"""Tests for automated dataset downloaders.

Phase 2, Task 2.3 of flow-data-acquisition-plan.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from pathlib import Path


class TestDownloadVesselNN:
    """download_vesselnn uses git clone to fetch the dataset."""

    def test_calls_git_clone(self, tmp_path: Path) -> None:
        from minivess.data.downloaders import download_vesselnn

        target = tmp_path / "vesselnn"
        with patch("minivess.data.downloaders.subprocess") as mock_sub:
            mock_sub.run.return_value.returncode = 0
            result = download_vesselnn(target_dir=target)

        mock_sub.run.assert_called_once()
        args = mock_sub.run.call_args
        cmd = args[0][0]
        assert cmd[0] == "git"
        assert "clone" in cmd
        assert "--depth" in cmd
        assert result == target

    def test_raises_on_clone_failure(self, tmp_path: Path) -> None:
        import pytest

        from minivess.data.downloaders import download_vesselnn

        target = tmp_path / "vesselnn"
        with (
            patch("minivess.data.downloaders.subprocess") as mock_sub,
            pytest.raises(RuntimeError, match="git clone failed"),
        ):
            mock_sub.run.return_value.returncode = 1
            mock_sub.run.return_value.stderr = "fatal: error"
            download_vesselnn(target_dir=target)

    def test_skips_if_exists(self, tmp_path: Path) -> None:
        from minivess.data.downloaders import download_vesselnn

        target = tmp_path / "vesselnn"
        target.mkdir()
        (target / ".git").mkdir()  # Simulate existing clone

        with patch("minivess.data.downloaders.subprocess") as mock_sub:
            result = download_vesselnn(target_dir=target, skip_existing=True)

        mock_sub.run.assert_not_called()
        assert result == target


class TestGetDownloader:
    """get_downloader returns the right function for each dataset."""

    def test_vesselnn_has_downloader(self) -> None:
        from minivess.data.downloaders import get_downloader

        func = get_downloader("vesselnn")
        assert func is not None
        assert callable(func)

    def test_manual_datasets_return_none(self) -> None:
        from minivess.data.downloaders import get_downloader

        assert get_downloader("minivess") is None
        assert get_downloader("deepvess") is None
        assert get_downloader("tubenet_2pm") is None

    def test_unknown_dataset_returns_none(self) -> None:
        from minivess.data.downloaders import get_downloader

        assert get_downloader("nonexistent") is None
