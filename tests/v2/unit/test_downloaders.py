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


class TestDownloadDeepVess:
    """download_deepvess downloads ZIP from eCommons and extracts."""

    def test_downloads_and_extracts(self, tmp_path: Path) -> None:
        import io
        import zipfile

        from minivess.data.downloaders import download_deepvess

        target = tmp_path / "deepvess"

        # Create a mock ZIP containing TIFF-like files
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("HaftJavaherian_DeepVess2018_Images/sample.tif", b"TIFF_DATA")
            zf.writestr(
                "HaftJavaherian_DeepVess2018_Images/sample_seg.tif", b"SEG_DATA"
            )
        zip_bytes = zip_buffer.getvalue()

        # Mock httpx response
        with patch("minivess.data.downloaders.httpx") as mock_httpx:
            mock_response = mock_httpx.get.return_value
            mock_response.status_code = 200
            mock_response.content = zip_bytes
            mock_response.raise_for_status = lambda: None
            result = download_deepvess(target_dir=target)

        mock_httpx.get.assert_called_once()
        assert result == target

    def test_skips_if_exists(self, tmp_path: Path) -> None:
        from minivess.data.downloaders import download_deepvess

        target = tmp_path / "deepvess"
        target.mkdir()
        # Create marker files to simulate existing download
        (target / "images").mkdir()
        (target / "labels").mkdir()
        (target / "images" / "vol1.tif").write_bytes(b"x")

        with patch("minivess.data.downloaders.httpx") as mock_httpx:
            result = download_deepvess(target_dir=target, skip_existing=True)

        mock_httpx.get.assert_not_called()
        assert result == target

    def test_raises_on_http_error(self, tmp_path: Path) -> None:
        import pytest

        from minivess.data.downloaders import download_deepvess

        target = tmp_path / "deepvess"
        with (
            patch("minivess.data.downloaders.httpx") as mock_httpx,
            pytest.raises(RuntimeError, match="DeepVess download failed"),
        ):
            mock_httpx.get.side_effect = Exception("Connection refused")
            download_deepvess(target_dir=target)


class TestDeepVessURLFormat:
    """DeepVess URL format validation."""

    def test_deepvess_url_is_valid_dspace7_bitstream(self) -> None:
        """_DEEPVESS_ZIP_URL follows Cornell eCommons DSpace 7 bitstream pattern."""
        from minivess.data.downloaders import _DEEPVESS_ZIP_URL

        assert "ecommons.cornell.edu/server/api/core/bitstreams/" in _DEEPVESS_ZIP_URL
        assert _DEEPVESS_ZIP_URL.endswith("/content")

    def test_deepvess_downloads_if_images_dir_empty(self, tmp_path: Path) -> None:
        """Empty images/ dir does NOT count as existing — download proceeds."""
        import io
        import zipfile

        from minivess.data.downloaders import download_deepvess

        target = tmp_path / "deepvess"
        target.mkdir()
        (target / "images").mkdir()  # Empty dir

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("sample.tif", b"TIFF_DATA")
        zip_bytes = zip_buffer.getvalue()

        with patch("minivess.data.downloaders.httpx") as mock_httpx:
            mock_response = mock_httpx.get.return_value
            mock_response.status_code = 200
            mock_response.content = zip_bytes
            mock_response.raise_for_status = lambda: None
            download_deepvess(target_dir=target, skip_existing=True)

        mock_httpx.get.assert_called_once()  # Download DID happen

    def test_deepvess_uses_httpx_not_requests(self) -> None:
        """download_deepvess imports httpx, not requests or urllib."""
        import ast
        from pathlib import Path as P

        src = P(__file__).resolve().parents[3] / "src" / "minivess" / "data" / "downloaders.py"
        tree = ast.parse(src.read_text(encoding="utf-8"))

        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])

        assert "httpx" in imports
        assert "requests" not in imports
        assert "urllib" not in imports


class TestGetDownloader:
    """get_downloader returns the right function for each dataset."""

    def test_vesselnn_has_downloader(self) -> None:
        from minivess.data.downloaders import get_downloader

        func = get_downloader("vesselnn")
        assert func is not None
        assert callable(func)

    def test_deepvess_has_downloader(self) -> None:
        from minivess.data.downloaders import get_downloader

        func = get_downloader("deepvess")
        assert func is not None
        assert callable(func)

    def test_manual_datasets_return_none(self) -> None:
        from minivess.data.downloaders import get_downloader

        assert get_downloader("minivess") is None
        # tubenet_2pm excluded from project

    def test_unknown_dataset_returns_none(self) -> None:
        from minivess.data.downloaders import get_downloader

        assert get_downloader("nonexistent") is None
