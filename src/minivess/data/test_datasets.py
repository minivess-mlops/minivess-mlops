"""Hierarchical test dataset registry and DataLoader builder.

Provides a two-level hierarchical DataLoader dictionary for multi-dataset
generalization evaluation:

    {dataset_name: {"all": DataLoader, "subset_name": DataLoader, ...}}

Used by the evaluation pipeline to run inference across multiple datasets
and their subsets (e.g., thin_vessels, thick_vessels) in a single pass.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from monai.data import CacheDataset, ThreadDataLoader
from pydantic import BaseModel, Field

from minivess.data.loader import discover_nifti_pairs
from minivess.data.transforms import build_val_transforms

if TYPE_CHECKING:
    from minivess.config.models import DataConfig

logger = logging.getLogger(__name__)


class DatasetSubset(BaseModel):
    """A named subset of a dataset (e.g., 'thin_vessels').

    Subsets are defined by explicit volume indices into the discovered
    data dicts list, or by metadata filter criteria.
    """

    name: str
    description: str = ""
    volume_indices: list[int] = Field(default_factory=list)
    filter_metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetEntry(BaseModel):
    """A registered evaluation dataset.

    Parameters
    ----------
    name:
        Unique identifier for the dataset.
    data_dir:
        Path to the dataset root directory (string for Pydantic serialization;
        converted to Path when building loaders).
    layout:
        Directory layout: ``"ebrains"`` (raw/seg) or ``"decathlon"``
        (imagesTr/labelsTr).
    description:
        Human-readable description of the dataset.
    subsets:
        Named subsets for fine-grained evaluation.
    """

    name: str
    data_dir: str
    layout: str = "ebrains"
    description: str = ""
    subsets: list[DatasetSubset] = Field(default_factory=list)


class DatasetRegistry:
    """Registry of evaluation datasets with hierarchical subset support.

    Maintains an in-memory mapping of dataset names to ``DatasetEntry``
    objects. Used to build the hierarchical DataLoader dictionary.
    """

    def __init__(self) -> None:
        self._datasets: dict[str, DatasetEntry] = {}

    def register(self, entry: DatasetEntry) -> None:
        """Register a dataset entry."""
        self._datasets[entry.name] = entry
        logger.info("Registered dataset '%s' at %s", entry.name, entry.data_dir)

    def list_datasets(self) -> list[str]:
        """Return sorted list of registered dataset names."""
        return sorted(self._datasets.keys())

    def get(self, name: str) -> DatasetEntry:
        """Get a dataset entry by name.

        Raises
        ------
        KeyError
            If the dataset is not registered.
        """
        return self._datasets[name]


# Type alias for the hierarchical DataLoader structure
HierarchicalDataLoaderDict = dict[str, dict[str, Any]]


def _build_loader_from_dicts(
    data_dicts: list[dict[str, str]],
    data_config: DataConfig,
    *,
    cache_rate: float = 1.0,
) -> ThreadDataLoader:
    """Build a validation ThreadDataLoader from data dicts.

    Uses the same pattern as ``build_val_loader`` in ``minivess.data.loader``
    but accepts pre-filtered data dicts instead of building from scratch.

    Parameters
    ----------
    data_dicts:
        List of {"image": path, "label": path} dictionaries.
    data_config:
        Data configuration with transform parameters.
    cache_rate:
        Fraction of data to cache in memory (0.0-1.0).
    """
    transforms = build_val_transforms(data_config)
    dataset = CacheDataset(
        data=data_dicts,
        transform=transforms,
        cache_rate=cache_rate,
        num_workers=data_config.num_workers,
        runtime_cache=True,
    )
    return ThreadDataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )


def build_hierarchical_dataloaders(
    registry: DatasetRegistry,
    data_config: DataConfig,
    *,
    cache_rate: float = 1.0,
) -> HierarchicalDataLoaderDict:
    """Build a hierarchical dict of DataLoaders from the dataset registry.

    For each registered dataset, discovers NIfTI pairs and creates:
    - An ``"all"`` DataLoader with every volume.
    - One DataLoader per named subset (filtered by ``volume_indices``).

    Parameters
    ----------
    registry:
        Dataset registry with registered ``DatasetEntry`` objects.
    data_config:
        Data configuration for building transforms.
    cache_rate:
        Fraction of data to cache in memory (0.0-1.0).

    Returns
    -------
    Nested dict: ``{dataset_name: {"all": DataLoader, "subset": DataLoader, ...}}``.
    Every dataset gets an ``"all"`` key with all volumes.
    """
    result: HierarchicalDataLoaderDict = {}

    for ds_name in registry.list_datasets():
        entry = registry.get(ds_name)
        data_dir = Path(entry.data_dir)

        all_pairs = discover_nifti_pairs(data_dir)
        logger.info(
            "Dataset '%s': discovered %d NIfTI pairs", ds_name, len(all_pairs)
        )

        ds_loaders: dict[str, Any] = {}

        # "all" loader: every volume in the dataset
        ds_loaders["all"] = _build_loader_from_dicts(
            all_pairs, data_config, cache_rate=cache_rate
        )

        # Subset loaders: filtered by volume indices
        for subset in entry.subsets:
            if subset.volume_indices:
                subset_pairs = [
                    all_pairs[i]
                    for i in subset.volume_indices
                    if i < len(all_pairs)
                ]
            else:
                # If no indices specified, use all pairs (metadata filtering
                # would go here in a future extension)
                subset_pairs = all_pairs

            if subset_pairs:
                ds_loaders[subset.name] = _build_loader_from_dicts(
                    subset_pairs, data_config, cache_rate=cache_rate
                )
                logger.info(
                    "  Subset '%s': %d volumes", subset.name, len(subset_pairs)
                )

        result[ds_name] = ds_loaders

    return result


def build_validation_dataloaders_from_folds(
    fold_splits: dict[int, list[dict[str, str]]],
    data_config: DataConfig,
    *,
    cache_rate: float = 1.0,
) -> HierarchicalDataLoaderDict:
    """Build hierarchical validation DataLoaders from k-fold splits.

    Creates one entry per fold in the same hierarchical structure as
    ``build_hierarchical_dataloaders``, enabling uniform evaluation
    code across both cross-validation and external test sets.

    Parameters
    ----------
    fold_splits:
        Mapping from fold index to validation data dicts for that fold.
        E.g., ``{0: [{"image": ..., "label": ...}, ...], 1: [...]}``.
    data_config:
        Data configuration for building transforms.
    cache_rate:
        Fraction of data to cache in memory (0.0-1.0).

    Returns
    -------
    Nested dict: ``{"fold_0": {"all": DataLoader}, "fold_1": {"all": DataLoader}, ...}``.
    """
    result: HierarchicalDataLoaderDict = {}

    for fold_idx in sorted(fold_splits.keys()):
        fold_key = f"fold_{fold_idx}"
        fold_pairs = fold_splits[fold_idx]

        loader = _build_loader_from_dicts(
            fold_pairs, data_config, cache_rate=cache_rate
        )
        result[fold_key] = {"all": loader}

        logger.info("Fold %d: %d validation volumes", fold_idx, len(fold_pairs))

    return result
