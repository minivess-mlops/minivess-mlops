"""Tests for external test dataset configuration and discovery.

External datasets: DeepVess (multi-photon) and tUbeNet 2PM (two-photon).
Only multiphoton/two-photon microscopy datasets — no light-sheet, EM, etc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from minivess.data.external_datasets import (
    EXTERNAL_DATASETS,
    ExternalDatasetConfig,
    discover_external_test_pairs,
    validate_external_config,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# TestExternalDatasetConfig
# ---------------------------------------------------------------------------


class TestExternalDatasetConfig:
    """Tests for ExternalDatasetConfig dataclass."""

    def test_deepvess_config_exists(self) -> None:
        """DeepVess config is registered."""
        assert "deepvess" in EXTERNAL_DATASETS

    def test_tubenet_config_exists(self) -> None:
        """tUbeNet 2PM config is registered."""
        assert "tubenet_2pm" in EXTERNAL_DATASETS

    def test_only_multiphoton_datasets(self) -> None:
        """All registered datasets use multiphoton or two-photon modality."""
        valid_modalities = {"multi-photon microscopy", "two-photon microscopy"}
        for name, config in EXTERNAL_DATASETS.items():
            assert config.modality in valid_modalities, (
                f"Dataset '{name}' has non-multiphoton modality: {config.modality}"
            )

    def test_all_mouse_brain(self) -> None:
        """All registered datasets are mouse brain vasculature."""
        for name, config in EXTERNAL_DATASETS.items():
            assert config.species == "mouse", f"{name}: species={config.species}"
            assert config.organ == "brain", f"{name}: organ={config.organ}"

    def test_resolution_is_3tuple(self) -> None:
        """Resolution must be a 3-tuple of positive floats (x, y, z)."""
        for name, config in EXTERNAL_DATASETS.items():
            assert len(config.resolution_um) == 3, (
                f"{name}: len={len(config.resolution_um)}"
            )
            for val in config.resolution_um:
                assert val > 0, f"{name}: resolution has non-positive value {val}"

    def test_source_url_nonempty(self) -> None:
        """All datasets have a non-empty source URL."""
        for name, config in EXTERNAL_DATASETS.items():
            assert config.source_url, f"{name}: empty source_url"

    def test_cite_ref_nonempty(self) -> None:
        """All datasets have a non-empty citation reference."""
        for name, config in EXTERNAL_DATASETS.items():
            assert config.cite_ref, f"{name}: empty cite_ref"

    def test_frozen_dataclass(self) -> None:
        """ExternalDatasetConfig should be immutable."""
        config = EXTERNAL_DATASETS["deepvess"]
        with pytest.raises(AttributeError):
            config.name = "changed"  # type: ignore[misc]

    def test_no_light_sheet_datasets(self) -> None:
        """No light-sheet microscopy datasets should be registered."""
        for name, config in EXTERNAL_DATASETS.items():
            assert "light-sheet" not in config.modality.lower(), (
                f"Dataset '{name}' is light-sheet — excluded per requirements"
            )

    def test_no_electron_microscopy_datasets(self) -> None:
        """No electron microscopy datasets should be registered."""
        for name, config in EXTERNAL_DATASETS.items():
            assert "electron" not in config.modality.lower(), (
                f"Dataset '{name}' is electron microscopy — excluded"
            )


# ---------------------------------------------------------------------------
# TestValidateExternalConfig
# ---------------------------------------------------------------------------


class TestValidateExternalConfig:
    """Tests for validate_external_config()."""

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = ExternalDatasetConfig(
            name="test_dataset",
            source_url="https://example.com/data",
            modality="multi-photon microscopy",
            organ="brain",
            species="mouse",
            resolution_um=(1.0, 1.0, 1.7),
            n_volumes=1,
            license="CC-BY-4.0",
            cite_ref="test_2025",
        )
        errors = validate_external_config(config)
        assert len(errors) == 0

    def test_invalid_modality(self) -> None:
        """Non-multiphoton modality is rejected."""
        config = ExternalDatasetConfig(
            name="bad_modality",
            source_url="https://example.com",
            modality="light-sheet fluorescence",
            organ="brain",
            species="mouse",
            resolution_um=(1.0, 1.0, 1.0),
            n_volumes=1,
            license="CC-BY-4.0",
            cite_ref="test_2025",
        )
        errors = validate_external_config(config)
        assert len(errors) > 0
        assert any("modality" in e.lower() for e in errors)

    def test_invalid_species(self) -> None:
        """Non-mouse species is rejected."""
        config = ExternalDatasetConfig(
            name="human_data",
            source_url="https://example.com",
            modality="two-photon microscopy",
            organ="brain",
            species="human",
            resolution_um=(1.0, 1.0, 1.0),
            n_volumes=1,
            license="CC-BY-4.0",
            cite_ref="test_2025",
        )
        errors = validate_external_config(config)
        assert len(errors) > 0
        assert any("species" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# TestDiscoverExternalTestPairs
# ---------------------------------------------------------------------------


class TestDiscoverExternalTestPairs:
    """Tests for discover_external_test_pairs()."""

    def test_discovers_nifti_pairs(self, tmp_path: Path) -> None:
        """Discovers image/label pairs from NIfTI files."""
        data_dir = tmp_path / "deepvess"
        images = data_dir / "images"
        labels = data_dir / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)

        # Create dummy NIfTI files
        (images / "vol_001.nii.gz").write_bytes(b"fake")
        (labels / "vol_001.nii.gz").write_bytes(b"fake")
        (images / "vol_002.nii.gz").write_bytes(b"fake")
        (labels / "vol_002.nii.gz").write_bytes(b"fake")

        pairs = discover_external_test_pairs(data_dir, "deepvess")
        assert len(pairs) == 2
        for pair in pairs:
            assert "image" in pair
            assert "label" in pair

    def test_missing_label_excluded(self, tmp_path: Path) -> None:
        """Images without matching labels are excluded."""
        data_dir = tmp_path / "deepvess"
        images = data_dir / "images"
        labels = data_dir / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)

        (images / "vol_001.nii.gz").write_bytes(b"fake")
        (images / "vol_002.nii.gz").write_bytes(b"fake")
        (labels / "vol_001.nii.gz").write_bytes(b"fake")
        # vol_002 label is missing

        pairs = discover_external_test_pairs(data_dir, "deepvess")
        assert len(pairs) == 1

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Empty directory returns no pairs."""
        data_dir = tmp_path / "deepvess"
        data_dir.mkdir()

        pairs = discover_external_test_pairs(data_dir, "deepvess")
        assert len(pairs) == 0

    def test_missing_directory_returns_empty(self, tmp_path: Path) -> None:
        """Missing directory returns empty list (not error)."""
        data_dir = tmp_path / "nonexistent"
        pairs = discover_external_test_pairs(data_dir, "deepvess")
        assert len(pairs) == 0

    def test_tiff_pairs(self, tmp_path: Path) -> None:
        """Discovers TIFF image/label pairs."""
        data_dir = tmp_path / "tubenet_2pm"
        images = data_dir / "images"
        labels = data_dir / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)

        (images / "stack_001.tif").write_bytes(b"fake")
        (labels / "stack_001.tif").write_bytes(b"fake")

        pairs = discover_external_test_pairs(data_dir, "tubenet_2pm")
        assert len(pairs) == 1
