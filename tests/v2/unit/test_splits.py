from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from minivess.data.splits import (
    FoldSplit,
    generate_kfold_splits,
    load_splits,
    save_splits,
)


def _make_data_dicts(n: int) -> list[dict[str, str]]:
    """Create n synthetic data dicts for testing."""
    return [{"image": f"/img/{i}.nii.gz", "label": f"/lbl/{i}.nii.gz"} for i in range(n)]


class TestGenerateKfoldSplits:
    """Tests for generate_kfold_splits()."""

    def test_deterministic_same_seed(self) -> None:
        data = _make_data_dicts(12)
        s1 = generate_kfold_splits(data, num_folds=3, seed=42)
        s2 = generate_kfold_splits(data, num_folds=3, seed=42)
        for f1, f2 in zip(s1, s2, strict=True):
            assert f1.train == f2.train
            assert f1.val == f2.val

    def test_different_seed_different_splits(self) -> None:
        data = _make_data_dicts(12)
        s1 = generate_kfold_splits(data, num_folds=3, seed=42)
        s2 = generate_kfold_splits(data, num_folds=3, seed=99)
        # At least one fold must differ
        any_different = any(
            f1.val != f2.val for f1, f2 in zip(s1, s2, strict=True)
        )
        assert any_different

    def test_no_overlap_between_train_and_val(self) -> None:
        data = _make_data_dicts(15)
        splits = generate_kfold_splits(data, num_folds=3, seed=42)
        for fold in splits:
            train_imgs = {d["image"] for d in fold.train}
            val_imgs = {d["image"] for d in fold.val}
            assert train_imgs.isdisjoint(val_imgs), "Train/val overlap detected"

    def test_full_coverage(self) -> None:
        data = _make_data_dicts(15)
        splits = generate_kfold_splits(data, num_folds=3, seed=42)
        all_images = {d["image"] for d in data}
        for fold in splits:
            fold_images = {d["image"] for d in fold.train} | {d["image"] for d in fold.val}
            assert fold_images == all_images

    def test_correct_number_of_folds(self) -> None:
        data = _make_data_dicts(10)
        splits = generate_kfold_splits(data, num_folds=5, seed=42)
        assert len(splits) == 5

    def test_val_sizes_balanced(self) -> None:
        data = _make_data_dicts(12)
        splits = generate_kfold_splits(data, num_folds=3, seed=42)
        val_sizes = [len(fold.val) for fold in splits]
        assert max(val_sizes) - min(val_sizes) <= 1

    def test_num_folds_less_than_2_raises(self) -> None:
        data = _make_data_dicts(5)
        with pytest.raises(ValueError, match="num_folds must be >= 2"):
            generate_kfold_splits(data, num_folds=1)

    def test_more_folds_than_samples_raises(self) -> None:
        data = _make_data_dicts(3)
        with pytest.raises(ValueError, match="exceeds sample count"):
            generate_kfold_splits(data, num_folds=5)


class TestSplitsIO:
    """Tests for save_splits() / load_splits() JSON roundtrip."""

    def test_roundtrip(self, tmp_path: Path) -> None:
        data = _make_data_dicts(12)
        original = generate_kfold_splits(data, num_folds=3, seed=42)
        path = tmp_path / "splits.json"
        save_splits(original, path)
        loaded = load_splits(path)

        assert len(loaded) == len(original)
        for orig, load in zip(original, loaded, strict=True):
            assert orig.train == load.train
            assert orig.val == load.val

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        data = _make_data_dicts(6)
        splits = generate_kfold_splits(data, num_folds=2, seed=42)
        path = tmp_path / "nested" / "dir" / "splits.json"
        save_splits(splits, path)
        assert path.exists()

    def test_loaded_type_is_foldsplit(self, tmp_path: Path) -> None:
        data = _make_data_dicts(6)
        splits = generate_kfold_splits(data, num_folds=2, seed=42)
        path = tmp_path / "splits.json"
        save_splits(splits, path)
        loaded = load_splits(path)
        assert all(isinstance(s, FoldSplit) for s in loaded)
