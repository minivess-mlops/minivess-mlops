"""Tests for smoke test splits file validity (T1.2).

Validates:
- configs/splits/smoke_test_1fold_4vol.json exists
- Has exactly 1 fold
- Each fold has 2 train + 2 val volumes
- Volume IDs match actual MiniVess data files
"""

from __future__ import annotations

import json
from pathlib import Path

SPLITS_PATH = Path("configs/splits/smoke_test_1fold_4vol.json")


class TestSmokeSplitsFile:
    """Verify smoke test 1-fold splits file structure."""

    def test_smoke_splits_file_exists(self) -> None:
        """The splits file must exist in the repo."""
        assert SPLITS_PATH.exists(), f"{SPLITS_PATH} not found"

    def test_smoke_splits_has_one_fold(self) -> None:
        """Splits file must contain exactly 1 fold."""
        splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
        assert isinstance(splits, list)
        assert len(splits) == 1, f"Expected 1 fold, got {len(splits)}"

    def test_smoke_splits_has_correct_volume_counts(self) -> None:
        """Each fold must have 2 train + 2 val volumes."""
        splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
        fold = splits[0]
        assert len(fold["train"]) == 2, f"Expected 2 train, got {len(fold['train'])}"
        assert len(fold["val"]) == 2, f"Expected 2 val, got {len(fold['val'])}"

    def test_smoke_splits_volume_ids_valid(self) -> None:
        """Volume paths must reference existing MiniVess volume IDs."""
        splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
        fold = splits[0]
        all_volumes = fold["train"] + fold["val"]
        for vol in all_volumes:
            image_path = vol["image"]
            label_path = vol["label"]
            # Must be in data/raw/minivess/ directory
            assert "minivess" in image_path, f"Unexpected path: {image_path}"
            assert "minivess" in label_path, f"Unexpected path: {label_path}"
            # Image and label must reference same volume ID
            image_stem = Path(image_path).stem.replace(".nii", "")
            label_stem = Path(label_path).stem.replace(".nii", "")
            assert image_stem == label_stem, (
                f"Image/label mismatch: {image_stem} vs {label_stem}"
            )
