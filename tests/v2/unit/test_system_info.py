"""Tests for system info collection module.

RED phase: all tests written before implementation exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path


def _gpu_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


class TestGetSystemParams:
    """Tests for get_system_params()."""

    def test_returns_dict_with_expected_keys(self) -> None:
        from minivess.observability.system_info import get_system_params

        result = get_system_params()
        expected_keys = {
            "sys_python_version",
            "sys_os",
            "sys_os_kernel",
            "sys_hostname",
            "sys_total_ram_gb",
            "sys_cpu_model",
        }
        assert expected_keys.issubset(result.keys())

    def test_all_values_are_strings(self) -> None:
        from minivess.observability.system_info import get_system_params

        result = get_system_params()
        for key, value in result.items():
            assert isinstance(value, str), f"{key} is {type(value)}, expected str"

    def test_python_version_not_empty(self) -> None:
        from minivess.observability.system_info import get_system_params

        result = get_system_params()
        assert result["sys_python_version"]
        assert "." in result["sys_python_version"]  # e.g. "3.13.2"

    def test_ram_is_numeric_string(self) -> None:
        from minivess.observability.system_info import get_system_params

        result = get_system_params()
        ram_str = result["sys_total_ram_gb"]
        # Should be parseable as float, e.g. "62.7"
        assert float(ram_str) > 0

    def test_cpu_model_fallback_when_proc_unavailable(self) -> None:
        from minivess.observability.system_info import _get_cpu_model

        with patch(
            "minivess.observability.system_info.open", side_effect=FileNotFoundError
        ):
            result = _get_cpu_model()
        # Should still return a string, possibly from platform.processor()
        assert isinstance(result, str)
        assert result  # should not be empty


class TestGetLibraryVersions:
    """Tests for get_library_versions()."""

    def test_returns_dict_with_expected_keys(self) -> None:
        from minivess.observability.system_info import get_library_versions

        result = get_library_versions()
        expected_keys = {
            "sys_torch_version",
            "sys_monai_version",
            "sys_cuda_version",
            "sys_cudnn_version",
            "sys_mlflow_version",
            "sys_numpy_version",
        }
        assert expected_keys.issubset(result.keys())

    def test_all_values_are_strings(self) -> None:
        from minivess.observability.system_info import get_library_versions

        result = get_library_versions()
        for key, value in result.items():
            assert isinstance(value, str), f"{key} is {type(value)}, expected str"

    def test_torch_version_present(self) -> None:
        from minivess.observability.system_info import get_library_versions

        result = get_library_versions()
        assert result["sys_torch_version"] != "not_installed"

    def test_graceful_degradation_when_library_missing(self) -> None:
        """When a library can't be imported, return 'not_installed'."""
        from minivess.observability.system_info import get_library_versions

        # Normal case: monai should be present
        normal = get_library_versions()
        assert normal["sys_monai_version"] != "not_installed"

        # When monai is not importable, should return "not_installed"
        with patch.dict("sys.modules", {"monai": None}):
            degraded = get_library_versions()
        assert degraded["sys_monai_version"] == "not_installed"


class TestGetGpuInfo:
    """Tests for get_gpu_info()."""

    def test_returns_dict_with_expected_keys(self) -> None:
        from minivess.observability.system_info import get_gpu_info

        result = get_gpu_info()
        assert "sys_gpu_count" in result
        assert "sys_gpu_model" in result
        assert "sys_gpu_vram_mb" in result

    def test_all_values_are_strings(self) -> None:
        from minivess.observability.system_info import get_gpu_info

        result = get_gpu_info()
        for key, value in result.items():
            assert isinstance(value, str), f"{key} is {type(value)}, expected str"

    def test_no_gpu_returns_zero_count(self) -> None:
        from minivess.observability.system_info import get_gpu_info

        with patch("torch.cuda.is_available", return_value=False):
            result = get_gpu_info()
        assert result["sys_gpu_count"] == "0"
        assert result["sys_gpu_model"] == "N/A"

    @pytest.mark.skipif(
        not _gpu_available(),
        reason="GPU not available",
    )
    def test_gpu_present_returns_model_name(self) -> None:
        from minivess.observability.system_info import get_gpu_info

        result = get_gpu_info()
        assert result["sys_gpu_count"] != "0"
        assert result["sys_gpu_model"] != "N/A"
        assert int(result["sys_gpu_vram_mb"]) > 0


class TestGetGitInfo:
    """Tests for get_git_info()."""

    def test_returns_dict_with_expected_keys(self) -> None:
        from minivess.observability.system_info import get_git_info

        result = get_git_info()
        expected_keys = {
            "sys_git_commit",
            "sys_git_commit_short",
            "sys_git_branch",
            "sys_git_dirty",
        }
        assert expected_keys == set(result.keys())

    def test_all_values_are_strings(self) -> None:
        from minivess.observability.system_info import get_git_info

        result = get_git_info()
        for key, value in result.items():
            assert isinstance(value, str), f"{key} is {type(value)}, expected str"

    def test_commit_is_hex_hash(self) -> None:
        from minivess.observability.system_info import get_git_info

        result = get_git_info()
        commit = result["sys_git_commit"]
        if commit != "unknown":
            assert len(commit) == 40
            assert all(c in "0123456789abcdef" for c in commit)

    def test_short_commit_is_prefix(self) -> None:
        from minivess.observability.system_info import get_git_info

        result = get_git_info()
        if result["sys_git_commit"] != "unknown":
            assert result["sys_git_commit"].startswith(result["sys_git_commit_short"])

    def test_dirty_is_boolean_string(self) -> None:
        from minivess.observability.system_info import get_git_info

        result = get_git_info()
        assert result["sys_git_dirty"] in ("true", "false", "unknown")

    def test_no_git_returns_unknown(self) -> None:
        from minivess.observability.system_info import get_git_info

        with patch(
            "subprocess.run",
            side_effect=FileNotFoundError("git not found"),
        ):
            result = get_git_info()
        assert result["sys_git_commit"] == "unknown"
        assert result["sys_git_branch"] == "unknown"

    def test_detached_head_returns_descriptive_branch(self) -> None:
        """In detached HEAD state, branch should not be just 'HEAD'."""
        from minivess.observability.system_info import get_git_info

        def mock_run(cmd, **kwargs):
            mock_result = MagicMock()
            mock_result.returncode = 0
            if "rev-parse" in cmd and "HEAD" in cmd and "--abbrev-ref" not in cmd:
                if "--short" in cmd:
                    mock_result.stdout = "abc1234\n"
                else:
                    mock_result.stdout = "abc1234567890abcdef1234567890abcdef123456\n"
            elif "--abbrev-ref" in cmd:
                mock_result.stdout = "HEAD\n"  # detached HEAD returns literal "HEAD"
            elif "diff" in cmd and "--quiet" in cmd:
                mock_result.returncode = 0  # clean
            else:
                mock_result.stdout = "\n"
            return mock_result

        with patch("subprocess.run", side_effect=mock_run):
            result = get_git_info()
        # Should include commit info, not just "HEAD"
        assert (
            "detached" in result["sys_git_branch"].lower()
            or "abc1234" in result["sys_git_branch"]
        )


class TestGetDvcInfo:
    """Tests for get_dvc_info()."""

    def test_returns_dict_with_expected_keys(self) -> None:
        from minivess.observability.system_info import get_dvc_info

        result = get_dvc_info()
        assert "sys_dvc_version" in result
        assert "sys_dvc_data_hash" in result
        assert "sys_dvc_data_nfiles" in result

    def test_all_values_are_strings(self) -> None:
        from minivess.observability.system_info import get_dvc_info

        result = get_dvc_info()
        for key, value in result.items():
            assert isinstance(value, str), f"{key} is {type(value)}, expected str"

    def test_dvc_version_is_present(self) -> None:
        """DVC is a Python dependency, so version should be available."""
        from minivess.observability.system_info import get_dvc_info

        result = get_dvc_info()
        assert result["sys_dvc_version"] != "not_installed"

    def test_graceful_when_no_dvc_file(self, tmp_path: Path) -> None:
        """When .dvc file doesn't exist, returns 'unknown' for data hash."""
        from minivess.observability.system_info import get_dvc_info

        with patch(
            "minivess.observability.system_info._DVC_FILE_PATH",
            tmp_path / "nonexistent.dvc",
        ):
            result = get_dvc_info()
        assert result["sys_dvc_data_hash"] == "unknown"


class TestGetAllSystemInfo:
    """Tests for get_all_system_info() — the combined function."""

    def test_combines_all_sections(self) -> None:
        from minivess.observability.system_info import get_all_system_info

        result = get_all_system_info()
        # Should have keys from all four functions
        assert "sys_python_version" in result  # from get_system_params
        assert "sys_torch_version" in result  # from get_library_versions
        assert "sys_gpu_count" in result  # from get_gpu_info
        assert "sys_git_commit" in result  # from get_git_info
        assert "sys_dvc_version" in result  # from get_dvc_info

    def test_all_values_are_strings(self) -> None:
        from minivess.observability.system_info import get_all_system_info

        result = get_all_system_info()
        for key, value in result.items():
            assert isinstance(value, str), f"{key} is {type(value)}, expected str"

    def test_all_keys_have_sys_prefix(self) -> None:
        from minivess.observability.system_info import get_all_system_info

        result = get_all_system_info()
        for key in result:
            assert key.startswith("sys_"), f"Key {key} missing sys_ prefix"

    def test_no_empty_values(self) -> None:
        from minivess.observability.system_info import get_all_system_info

        result = get_all_system_info()
        for key, value in result.items():
            assert value, f"{key} has empty value"
