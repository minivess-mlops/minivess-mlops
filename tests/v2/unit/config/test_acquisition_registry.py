"""Acquisition registry tests — prevents regression to manual download.

DeepVess was changed from download_method="manual" to "http_download" in the
11th pass session. A regression would break the acquisition flow.

Plan: run-debug-factorial-experiment-11th-pass-test-coverage-improvement.xml
Category 4 (P1): T4.3
"""

from __future__ import annotations


class TestAcquisitionRegistryMethods:
    """Verify download_method values for each dataset."""

    def test_deepvess_acquisition_method_is_http_download(self) -> None:
        from minivess.data.acquisition_registry import ACQUISITION_REGISTRY

        entry = ACQUISITION_REGISTRY["deepvess"]
        assert entry.download_method == "http_download"

    def test_vesselnn_acquisition_method_is_git_clone(self) -> None:
        from minivess.data.acquisition_registry import ACQUISITION_REGISTRY

        entry = ACQUISITION_REGISTRY["vesselnn"]
        assert entry.download_method == "git_clone"

    def test_minivess_acquisition_method_is_manual(self) -> None:
        from minivess.data.acquisition_registry import ACQUISITION_REGISTRY

        entry = ACQUISITION_REGISTRY["minivess"]
        assert entry.download_method == "manual"

    def test_all_datasets_have_download_method(self) -> None:
        from minivess.data.acquisition_registry import ACQUISITION_REGISTRY

        known_methods = {"http_download", "git_clone", "manual", "api"}
        for name, entry in ACQUISITION_REGISTRY.items():
            assert hasattr(entry, "download_method"), (
                f"Dataset '{name}' missing download_method"
            )
            assert entry.download_method in known_methods, (
                f"Dataset '{name}' has unknown download_method: {entry.download_method}"
            )
