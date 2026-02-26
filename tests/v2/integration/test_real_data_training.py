"""Real data training tests — require MiniVess dataset.

Skipped in CI. Run locally with:
    uv run pytest tests/v2/integration/test_real_data_training.py -m real_data -x -q
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Default data location (MiniVess downloaded by scripts/download_minivess.py)
_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"
_HAS_DATA = (_DATA_DIR / "imagesTr").exists() or (_DATA_DIR / "raw").exists()


pytestmark = pytest.mark.real_data


@pytest.mark.skipif(not _HAS_DATA, reason="MiniVess dataset not found")
class TestRealDataTraining:
    """Tests using the actual MiniVess dataset."""

    def test_one_epoch_dynunet_dice_ce(self) -> None:
        """1-epoch DynUNet training on real data with DiceCE loss."""
        from scripts.train import parse_args, run_experiment

        args = parse_args(
            [
                "--compute",
                "cpu",
                "--loss",
                "dice_ce",
                "--debug",
                "--data-dir",
                str(_DATA_DIR),
                "--num-folds",
                "2",
                "--max-epochs",
                "1",
            ]
        )

        results = run_experiment(args)

        for loss_name, fold_results in results.items():
            for fold_result in fold_results:
                best_loss = fold_result["best_val_loss"]
                assert best_loss == best_loss, f"NaN loss for {loss_name}"
                assert best_loss < float("inf"), f"Inf loss for {loss_name}"

    def test_metrics_reloaded_on_real_predictions(self) -> None:
        """MetricsReloaded metrics compute without error on real-shaped data."""
        from minivess.pipeline.evaluation import EvaluationRunner

        # Simulate real-sized volume
        rng = np.random.default_rng(42)
        shape = (128, 128, 64)
        pred = rng.integers(0, 2, size=shape)
        label = rng.integers(0, 2, size=shape)

        runner = EvaluationRunner()
        result = runner.evaluate_volume(pred, label)

        assert "dsc" in result
        assert not np.isnan(result["dsc"])

    def test_no_data_leakage_between_folds(self) -> None:
        """No volume appears in both train and val for any fold."""
        from minivess.data.loader import discover_nifti_pairs
        from minivess.data.splits import generate_kfold_splits

        data_dicts = discover_nifti_pairs(_DATA_DIR)
        splits = generate_kfold_splits(data_dicts, num_folds=3, seed=42)

        for fold_id, fold in enumerate(splits):
            train_imgs = {d["image"] for d in fold.train}
            val_imgs = {d["image"] for d in fold.val}
            overlap = train_imgs & val_imgs
            assert not overlap, (
                f"Fold {fold_id}: {len(overlap)} volumes in both train and val"
            )

    def test_augmentation_no_nan(self) -> None:
        """Data augmentation produces no NaN volumes."""
        from monai.data import CacheDataset

        from minivess.config.models import DataConfig
        from minivess.data.loader import discover_nifti_pairs
        from minivess.data.transforms import build_train_transforms

        data_dicts = discover_nifti_pairs(_DATA_DIR)[:3]  # Use 3 volumes
        config = DataConfig(
            dataset_name="minivess",
            data_dir=_DATA_DIR,
            patch_size=(64, 64, 16),
        )
        transforms = build_train_transforms(config)
        dataset = CacheDataset(data=data_dicts, transform=transforms, cache_rate=0.0)

        for i in range(min(len(dataset), 3)):
            sample = dataset[i]
            img = sample["image"]
            assert not np.isnan(img.numpy()).any(), f"NaN in augmented volume {i}"
