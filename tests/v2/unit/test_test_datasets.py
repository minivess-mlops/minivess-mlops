"""Tests for the hierarchical test dataset registry and DataLoader builder.

RED phase: These tests verify DatasetEntry, DatasetRegistry, and the
hierarchical DataLoader dictionary builder for multi-dataset generalization
evaluation.
"""

from __future__ import annotations

from pathlib import Path

import pytest


class TestDatasetEntry:
    """Verify DatasetEntry Pydantic model."""

    def test_dataset_entry_valid(self) -> None:
        """A DatasetEntry with required fields is valid."""
        from minivess.data.test_datasets import DatasetEntry

        entry = DatasetEntry(
            name="minivess_debug",
            data_dir="/tmp/debug_ds",
            description="Debug dataset for testing",
        )
        assert entry.name == "minivess_debug"
        assert entry.data_dir == "/tmp/debug_ds"
        assert entry.layout == "ebrains"  # default
        assert entry.subsets == []

    def test_dataset_entry_with_subsets(self) -> None:
        """A DatasetEntry can contain named subsets."""
        from minivess.data.test_datasets import DatasetEntry, DatasetSubset

        subsets = [
            DatasetSubset(
                name="thin_vessels",
                description="Volumes with thin vessels",
                volume_indices=[0, 1],
            ),
            DatasetSubset(
                name="thick_vessels",
                description="Volumes with thick vessels",
                volume_indices=[2, 3],
            ),
        ]
        entry = DatasetEntry(
            name="minivess_debug",
            data_dir="/tmp/debug_ds",
            subsets=subsets,
        )
        assert len(entry.subsets) == 2
        assert entry.subsets[0].name == "thin_vessels"
        assert entry.subsets[1].volume_indices == [2, 3]


class TestDatasetRegistry:
    """Verify DatasetRegistry register/list/get operations."""

    def test_dataset_registry_register_and_list(self) -> None:
        """Register datasets and list them alphabetically."""
        from minivess.data.test_datasets import DatasetEntry, DatasetRegistry

        registry = DatasetRegistry()

        registry.register(DatasetEntry(name="ds_beta", data_dir="/tmp/beta"))
        registry.register(DatasetEntry(name="ds_alpha", data_dir="/tmp/alpha"))

        names = registry.list_datasets()
        assert names == ["ds_alpha", "ds_beta"]

    def test_empty_registry_returns_empty_dict(self) -> None:
        """An empty registry produces an empty hierarchical dict."""
        from minivess.config.models import DataConfig
        from minivess.data.test_datasets import (
            DatasetRegistry,
            build_hierarchical_dataloaders,
        )

        registry = DatasetRegistry()
        config = DataConfig(
            dataset_name="test",
            data_dir=Path("/tmp/nonexistent"),
            patch_size=(16, 16, 8),
            num_workers=0,
        )
        result = build_hierarchical_dataloaders(registry, config)
        assert result == {}


class TestBuildHierarchicalDataloaders:
    """Verify hierarchical DataLoader dict structure with real NIfTI files."""

    @pytest.fixture()
    def debug_dataset_dir(self, tmp_path: Path) -> Path:
        """Create a debug dataset for loader tests."""
        from minivess.data.debug_dataset import create_debug_dataset

        return create_debug_dataset(
            tmp_path / "debug_ds",
            n_volumes=4,
            volume_shape=(32, 32, 8),
            seed=42,
        )

    def test_build_hierarchical_dataloaders_structure(
        self, tmp_path: Path, debug_dataset_dir: Path
    ) -> None:
        """Hierarchical dict has dataset name as top-level key."""
        from minivess.config.models import DataConfig
        from minivess.data.test_datasets import (
            DatasetEntry,
            DatasetRegistry,
            build_hierarchical_dataloaders,
        )

        registry = DatasetRegistry()
        registry.register(
            DatasetEntry(name="debug", data_dir=str(debug_dataset_dir))
        )

        config = DataConfig(
            dataset_name="test",
            data_dir=debug_dataset_dir,
            patch_size=(16, 16, 8),
            num_workers=0,
        )

        result = build_hierarchical_dataloaders(registry, config, cache_rate=1.0)

        assert "debug" in result
        assert isinstance(result["debug"], dict)

    def test_hierarchical_dict_has_all_key_per_dataset(
        self, tmp_path: Path, debug_dataset_dir: Path
    ) -> None:
        """Every dataset gets an 'all' key with a DataLoader for all volumes."""
        from minivess.config.models import DataConfig
        from minivess.data.test_datasets import (
            DatasetEntry,
            DatasetRegistry,
            build_hierarchical_dataloaders,
        )

        registry = DatasetRegistry()
        registry.register(
            DatasetEntry(name="debug", data_dir=str(debug_dataset_dir))
        )

        config = DataConfig(
            dataset_name="test",
            data_dir=debug_dataset_dir,
            patch_size=(16, 16, 8),
            num_workers=0,
        )

        result = build_hierarchical_dataloaders(registry, config, cache_rate=1.0)

        assert "all" in result["debug"]
        # The "all" loader should be a MONAI ThreadDataLoader
        loader = result["debug"]["all"]
        batch = next(iter(loader))
        assert "image" in batch
        assert "label" in batch

    def test_hierarchical_dict_subsets(
        self, tmp_path: Path, debug_dataset_dir: Path
    ) -> None:
        """Subsets produce additional keys in the dataset's dict."""
        from minivess.config.models import DataConfig
        from minivess.data.test_datasets import (
            DatasetEntry,
            DatasetRegistry,
            DatasetSubset,
            build_hierarchical_dataloaders,
        )

        registry = DatasetRegistry()
        registry.register(
            DatasetEntry(
                name="debug",
                data_dir=str(debug_dataset_dir),
                subsets=[
                    DatasetSubset(
                        name="first_two",
                        description="First two volumes",
                        volume_indices=[0, 1],
                    ),
                    DatasetSubset(
                        name="last_two",
                        description="Last two volumes",
                        volume_indices=[2, 3],
                    ),
                ],
            )
        )

        config = DataConfig(
            dataset_name="test",
            data_dir=debug_dataset_dir,
            patch_size=(16, 16, 8),
            num_workers=0,
        )

        result = build_hierarchical_dataloaders(registry, config, cache_rate=1.0)

        assert "all" in result["debug"]
        assert "first_two" in result["debug"]
        assert "last_two" in result["debug"]

        # Subset loaders should have fewer items than "all"
        all_count = len(result["debug"]["all"].dataset)
        first_count = len(result["debug"]["first_two"].dataset)
        last_count = len(result["debug"]["last_two"].dataset)

        assert all_count == 4
        assert first_count == 2
        assert last_count == 2


class TestValidationDataloadersFromFolds:
    """Verify fold-based hierarchical DataLoader construction."""

    def test_validation_dataloaders_from_folds(self, tmp_path: Path) -> None:
        """Fold splits produce fold_N keys with 'all' sub-keys."""
        from minivess.config.models import DataConfig
        from minivess.data.debug_dataset import create_debug_dataset
        from minivess.data.loader import discover_nifti_pairs
        from minivess.data.test_datasets import build_validation_dataloaders_from_folds

        ds_dir = create_debug_dataset(
            tmp_path / "fold_ds",
            n_volumes=4,
            volume_shape=(32, 32, 8),
            seed=42,
        )

        all_pairs = discover_nifti_pairs(ds_dir)

        # Simulate 2-fold split: each fold gets 2 validation volumes
        fold_splits: dict[int, list[dict[str, str]]] = {
            0: all_pairs[:2],
            1: all_pairs[2:],
        }

        config = DataConfig(
            dataset_name="test",
            data_dir=ds_dir,
            patch_size=(16, 16, 8),
            num_workers=0,
        )

        result = build_validation_dataloaders_from_folds(
            fold_splits, config, cache_rate=1.0
        )

        assert "fold_0" in result
        assert "fold_1" in result
        assert "all" in result["fold_0"]
        assert "all" in result["fold_1"]

        # Each fold's loader should have the correct number of items
        assert len(result["fold_0"]["all"].dataset) == 2
        assert len(result["fold_1"]["all"].dataset) == 2
