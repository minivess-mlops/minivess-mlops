"""Tests for external dataset license verification and download helpers.

Covers Task 1.1 of data-engineering-improvement-plan.xml.
Closes #176 (partial — licenses portion).
"""

from __future__ import annotations

import pytest

from minivess.data.external_datasets import (
    EXTERNAL_DATASETS,
    download_external_dataset,
    validate_external_config,
)


class TestLicenseVerification:
    """License fields match verified status from architecture report."""

    def test_deepvess_license_is_not_tbd(self) -> None:
        cfg = EXTERNAL_DATASETS["deepvess"]
        assert cfg.license != "TBD", "DeepVess license should be updated from TBD"

    # tubenet_2pm license tests removed: olfactory bulb, different organ, only 1 2PM volume

    def test_deepvess_has_license_verified_flag(self) -> None:
        cfg = EXTERNAL_DATASETS["deepvess"]
        assert hasattr(cfg, "license_verified")
        # Verified 2026-03-22 from eCommons README: CC-BY 4.0
        assert cfg.license_verified is True, (
            "DeepVess license verified from eCommons README"
        )

    def test_all_configs_have_cite_ref(self) -> None:
        for name, cfg in EXTERNAL_DATASETS.items():
            assert cfg.cite_ref, f"{name} missing cite_ref"


class TestVesselNNConfig:
    """vesselNN dataset should be in the registry."""

    def test_vesselnn_config_exists(self) -> None:
        assert "vesselnn" in EXTERNAL_DATASETS

    def test_vesselnn_license_is_mit(self) -> None:
        cfg = EXTERNAL_DATASETS["vesselnn"]
        assert cfg.license == "MIT"

    def test_vesselnn_is_valid(self) -> None:
        cfg = EXTERNAL_DATASETS["vesselnn"]
        errors = validate_external_config(cfg)
        assert errors == []


class TestDownloadHelper:
    """download_external_dataset() stub."""

    def test_download_returns_path(self, tmp_path: object) -> None:
        from pathlib import Path

        target = Path(str(tmp_path)) / "deepvess"
        result = download_external_dataset("deepvess", target)
        assert isinstance(result, Path)

    def test_download_creates_directory_structure(self, tmp_path: object) -> None:
        from pathlib import Path

        target = Path(str(tmp_path)) / "deepvess"
        download_external_dataset("deepvess", target)
        assert (target / "images").is_dir()
        assert (target / "labels").is_dir()

    def test_download_unknown_dataset_raises(self, tmp_path: object) -> None:
        from pathlib import Path

        target = Path(str(tmp_path)) / "nonexistent"
        with pytest.raises(KeyError):
            download_external_dataset("nonexistent_dataset_xyz", target)
