"""Tests for DVC version change detection.

Covers Task 5.2 of data-engineering-improvement-plan.xml.
Closes #180 (partial).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class TestDataVersionInfo:
    """DataVersionInfo dataclass."""

    def test_version_info_has_fields(self) -> None:
        from minivess.data.versioning import DataVersionInfo

        info = DataVersionInfo(
            dataset="deepvess",
            version="0.1.0",
            git_tag="data/deepvess/v0.1.0",
            changed_files=[],
        )
        assert info.dataset == "deepvess"
        assert info.version == "0.1.0"
        assert info.git_tag == "data/deepvess/v0.1.0"


class TestDetectDVCChange:
    """detect_dvc_change checks for .dvc file changes."""

    def test_detect_no_change_returns_false(self, tmp_path: Path) -> None:
        from minivess.data.versioning import detect_dvc_change

        # No .dvc files → no change
        result = detect_dvc_change(data_dir=tmp_path)
        assert result is False

    def test_detect_change_with_dvc_file(self, tmp_path: Path) -> None:
        from minivess.data.versioning import detect_dvc_change

        # Create a .dvc file (simulates DVC tracking)
        dvc_file = tmp_path / "test.dvc"
        dvc_file.write_text("md5: abc123\n", encoding="utf-8")
        result = detect_dvc_change(data_dir=tmp_path, reference_hash=None)
        # With no reference hash, any .dvc file is a "change"
        assert result is True

    def test_detect_no_change_matching_hash(self, tmp_path: Path) -> None:
        from minivess.data.versioning import (
            compute_dvc_hash,
            detect_dvc_change,
        )

        dvc_file = tmp_path / "test.dvc"
        dvc_file.write_text("md5: abc123\n", encoding="utf-8")
        current_hash = compute_dvc_hash(tmp_path)
        result = detect_dvc_change(data_dir=tmp_path, reference_hash=current_hash)
        assert result is False


class TestDataVersionTag:
    """Data version tag format."""

    def test_tag_format(self) -> None:
        from minivess.data.versioning import create_data_version_tag

        tag = create_data_version_tag("deepvess", "0.1.0")
        assert tag == "data/deepvess/v0.1.0"

    def test_tag_format_minivess(self) -> None:
        from minivess.data.versioning import create_data_version_tag

        tag = create_data_version_tag("minivess", "1.0.0")
        assert tag == "data/minivess/v1.0.0"


class TestGetCurrentDataVersion:
    """get_current_data_version reads from DVC config."""

    def test_get_version_returns_string_or_none(self) -> None:
        from minivess.data.versioning import get_current_data_version

        # For a dataset not in DVC_CONFIGS this should return None
        result = get_current_data_version("nonexistent_dataset_xyz")
        assert result is None

    def test_get_version_for_known_dataset(self) -> None:
        from minivess.data.versioning import get_current_data_version

        result = get_current_data_version("deepvess")
        assert isinstance(result, str)


class TestComputeDVCHash:
    """compute_dvc_hash hashes .dvc files for change detection."""

    def test_hash_is_deterministic(self, tmp_path: Path) -> None:
        from minivess.data.versioning import compute_dvc_hash

        dvc_file = tmp_path / "test.dvc"
        dvc_file.write_text("md5: abc123\n", encoding="utf-8")
        h1 = compute_dvc_hash(tmp_path)
        h2 = compute_dvc_hash(tmp_path)
        assert h1 == h2

    def test_hash_changes_with_content(self, tmp_path: Path) -> None:
        from minivess.data.versioning import compute_dvc_hash

        dvc_file = tmp_path / "test.dvc"
        dvc_file.write_text("md5: abc123\n", encoding="utf-8")
        h1 = compute_dvc_hash(tmp_path)
        dvc_file.write_text("md5: def456\n", encoding="utf-8")
        h2 = compute_dvc_hash(tmp_path)
        assert h1 != h2

    def test_hash_empty_dir_returns_string(self, tmp_path: Path) -> None:
        from minivess.data.versioning import compute_dvc_hash

        h = compute_dvc_hash(tmp_path)
        assert isinstance(h, str)
        assert len(h) > 0
