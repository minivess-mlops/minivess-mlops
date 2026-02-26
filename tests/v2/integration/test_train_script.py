from __future__ import annotations

import json
from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

_TEST_PATCH_SIZE = (16, 16, 16)  # Minimum viable for 4-level DynUNet


def _create_synthetic_nifti_dataset(data_dir: Path, n_volumes: int = 6) -> None:
    """Create a minimal synthetic NIfTI dataset in Medical Decathlon layout."""
    img_dir = data_dir / "imagesTr"
    lbl_dir = data_dir / "labelsTr"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    shape = (32, 32, 16)  # Must be >= 2x patch_size for cropping
    affine = np.eye(4)

    for i in range(n_volumes):
        img_data = rng.standard_normal(shape).astype(np.float32)
        # Ensure some foreground voxels for RandCropByPosNegLabeld
        lbl_data = np.zeros(shape, dtype=np.int16)
        lbl_data[8:24, 8:24, 4:12] = 1

        nib.save(nib.Nifti1Image(img_data, affine), img_dir / f"vol_{i:03d}.nii.gz")
        nib.save(nib.Nifti1Image(lbl_data, affine), lbl_dir / f"vol_{i:03d}.nii.gz")


class TestTrainScript:
    """Integration tests for scripts/train.py with synthetic data."""

    def test_smoke_debug_mode(self, tmp_path: Path) -> None:
        """1-epoch debug training should complete without errors or NaN."""
        from scripts.train import main

        data_dir = tmp_path / "data"
        _create_synthetic_nifti_dataset(data_dir, n_volumes=6)

        splits_file = tmp_path / "splits.json"

        patch = "x".join(str(x) for x in _TEST_PATCH_SIZE)
        main(
            [
                "--compute",
                "cpu",
                "--loss",
                "dice_ce",
                "--debug",
                "--data-dir",
                str(data_dir),
                "--splits-file",
                str(splits_file),
                "--num-folds",
                "2",
                "--seed",
                "42",
                "--experiment-name",
                "test_smoke",
                "--patch-size",
                patch,
            ]
        )

        # Verify splits file was created
        assert splits_file.exists()
        splits_data = json.loads(splits_file.read_text(encoding="utf-8"))
        assert len(splits_data) == 2

    def test_no_nan_loss(self, tmp_path: Path) -> None:
        """Training loss should not produce NaN."""
        from scripts.train import parse_args, run_experiment

        data_dir = tmp_path / "data"
        _create_synthetic_nifti_dataset(data_dir, n_volumes=6)

        patch = "x".join(str(x) for x in _TEST_PATCH_SIZE)
        args = parse_args(
            [
                "--compute",
                "cpu",
                "--loss",
                "dice_ce",
                "--debug",
                "--data-dir",
                str(data_dir),
                "--splits-file",
                str(tmp_path / "splits.json"),
                "--num-folds",
                "2",
                "--patch-size",
                patch,
            ]
        )

        results = run_experiment(args)

        for _loss_name, fold_results in results["training"].items():
            for fold_result in fold_results:
                assert (
                    fold_result["best_val_loss"] == fold_result["best_val_loss"]
                )  # not NaN
                assert fold_result["best_val_loss"] < float("inf")

    def test_compute_profile_applied(self, tmp_path: Path) -> None:
        """Verify that the compute profile is applied to configs."""
        from scripts.train import _build_configs, parse_args

        args = parse_args(
            [
                "--compute",
                "gpu_low",
                "--loss",
                "dice_ce",
                "--data-dir",
                str(tmp_path),
            ]
        )

        data_config, _, training_config = _build_configs(args)
        assert data_config.patch_size == (96, 96, 24)
        assert training_config.batch_size == 2
        assert training_config.mixed_precision is True

    def test_split_file_loaded(self, tmp_path: Path) -> None:
        """Verify that an existing split file is loaded."""
        from scripts.train import _build_configs, _load_or_generate_splits, parse_args

        from minivess.data.splits import FoldSplit, save_splits

        # Create fake split file
        splits = [
            FoldSplit(
                train=[{"image": "/a/1.nii.gz", "label": "/b/1.nii.gz"}],
                val=[{"image": "/a/2.nii.gz", "label": "/b/2.nii.gz"}],
            ),
            FoldSplit(
                train=[{"image": "/a/2.nii.gz", "label": "/b/2.nii.gz"}],
                val=[{"image": "/a/1.nii.gz", "label": "/b/1.nii.gz"}],
            ),
        ]
        splits_file = tmp_path / "splits.json"
        save_splits(splits, splits_file)

        args = parse_args(
            [
                "--compute",
                "cpu",
                "--loss",
                "dice_ce",
                "--data-dir",
                str(tmp_path),
                "--splits-file",
                str(splits_file),
                "--num-folds",
                "2",
            ]
        )

        _, _, training_config = _build_configs(args)
        loaded = _load_or_generate_splits(args, None, training_config)
        assert len(loaded) == 2
        assert loaded[0].train == splits[0].train
