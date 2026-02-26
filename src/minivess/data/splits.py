from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from minivess.data.loader import discover_nifti_pairs

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FoldSplit:
    """A single fold's train/val split as lists of data dicts."""

    train: list[dict[str, str]]
    val: list[dict[str, str]]


def generate_kfold_splits(
    data_dicts: list[dict[str, str]],
    num_folds: int = 3,
    seed: int = 42,
) -> list[FoldSplit]:
    """Generate deterministic K-fold cross-validation splits.

    Parameters
    ----------
    data_dicts:
        List of {"image": path, "label": path} dictionaries.
    num_folds:
        Number of folds (must be >= 2).
    seed:
        Random seed for reproducible shuffling.

    Returns
    -------
    list[FoldSplit]
        One FoldSplit per fold.

    Raises
    ------
    ValueError
        If num_folds < 2 or more folds than samples.
    """
    if num_folds < 2:
        msg = f"num_folds must be >= 2, got {num_folds}"
        raise ValueError(msg)
    if num_folds > len(data_dicts):
        msg = f"num_folds ({num_folds}) exceeds sample count ({len(data_dicts)})"
        raise ValueError(msg)

    import random

    indices = list(range(len(data_dicts)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    # Assign each index to a fold (round-robin)
    fold_indices: list[list[int]] = [[] for _ in range(num_folds)]
    for i, idx in enumerate(indices):
        fold_indices[i % num_folds].append(idx)

    splits: list[FoldSplit] = []
    for fold_id in range(num_folds):
        val_indices = fold_indices[fold_id]
        train_indices = [
            idx
            for f_id, idxs in enumerate(fold_indices)
            if f_id != fold_id
            for idx in idxs
        ]
        splits.append(
            FoldSplit(
                train=[data_dicts[i] for i in sorted(train_indices)],
                val=[data_dicts[i] for i in sorted(val_indices)],
            )
        )
        logger.info(
            "Fold %d: %d train, %d val", fold_id, len(train_indices), len(val_indices)
        )

    return splits


def save_splits(splits: list[FoldSplit], path: Path) -> None:
    """Save splits to JSON file.

    Parameters
    ----------
    splits:
        List of FoldSplit to serialize.
    path:
        Output JSON file path.
    """
    data = [{"train": fold.train, "val": fold.val} for fold in splits]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Saved %d-fold splits to %s", len(splits), path)


def load_splits(path: Path) -> list[FoldSplit]:
    """Load splits from JSON file.

    Parameters
    ----------
    path:
        JSON file with serialized splits.

    Returns
    -------
    list[FoldSplit]
        Deserialized splits.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    return [FoldSplit(train=fold["train"], val=fold["val"]) for fold in data]


def generate_kfold_splits_from_dir(
    data_dir: Path,
    num_folds: int = 3,
    seed: int = 42,
) -> list[FoldSplit]:
    """Convenience wrapper: discover NIfTI pairs, then generate K-fold splits.

    Parameters
    ----------
    data_dir:
        Root directory containing NIfTI images/labels.
    num_folds:
        Number of folds.
    seed:
        Random seed for reproducibility.
    """
    data_dicts = discover_nifti_pairs(data_dir)
    return generate_kfold_splits(data_dicts, num_folds=num_folds, seed=seed)
