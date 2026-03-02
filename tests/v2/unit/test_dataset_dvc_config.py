"""Tests for per-dataset DVC tracking configuration.

Covers Task 1.2 of data-engineering-improvement-plan.xml.
"""

from __future__ import annotations

from minivess.data.external_datasets import (
    DVC_CONFIGS,
    DatasetDVCConfig,
    get_dvc_tracked_datasets,
)


class TestDatasetDVCConfig:
    """DVC tracking configuration per external dataset."""

    def test_dvc_config_exists_for_deepvess(self) -> None:
        assert "deepvess" in DVC_CONFIGS
        cfg = DVC_CONFIGS["deepvess"]
        assert isinstance(cfg, DatasetDVCConfig)

    def test_dvc_config_exists_for_tubenet(self) -> None:
        assert "tubenet_2pm" in DVC_CONFIGS
        cfg = DVC_CONFIGS["tubenet_2pm"]
        assert isinstance(cfg, DatasetDVCConfig)

    def test_dvc_config_has_git_tag_format(self) -> None:
        for name, cfg in DVC_CONFIGS.items():
            assert "data/" in cfg.git_tag_format, (
                f"{name} tag format missing data/ prefix"
            )
            assert "{version}" in cfg.git_tag_format, (
                f"{name} tag format missing version"
            )

    def test_dvc_path_is_relative(self) -> None:
        for name, cfg in DVC_CONFIGS.items():
            assert not cfg.dvc_path.startswith("/"), (
                f"{name} dvc_path should be relative"
            )

    def test_get_dvc_tracked_datasets_returns_list(self) -> None:
        result = get_dvc_tracked_datasets()
        assert isinstance(result, list)
        assert len(result) >= 2
        assert "deepvess" in result
        assert "tubenet_2pm" in result
