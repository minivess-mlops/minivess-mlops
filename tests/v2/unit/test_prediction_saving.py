from __future__ import annotations

import numpy as np
import pytest

from minivess.pipeline.prediction_store import load_volume_prediction, save_volume_prediction


@pytest.fixture()
def tmp_prediction_dir(tmp_path):
    """Create a temporary directory for predictions."""
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    return pred_dir


@pytest.fixture()
def sample_predictions():
    """Create sample prediction arrays."""
    rng = np.random.default_rng(42)
    hard_pred = rng.integers(0, 2, size=(16, 32, 32), dtype=np.uint8)
    soft_pred = rng.random((16, 32, 32)).astype(np.float32)
    return hard_pred, soft_pred


class TestPredictionStore:
    """Tests for compressed prediction saving and loading."""

    def test_save_creates_npz(self, tmp_prediction_dir, sample_predictions):
        """Saving predictions creates a .npz file."""
        hard_pred, soft_pred = sample_predictions
        path = save_volume_prediction(
            output_dir=tmp_prediction_dir,
            volume_name="vol_01",
            hard_pred=hard_pred,
            soft_pred=soft_pred,
        )
        assert path.exists()
        assert path.suffix == ".npz"

    def test_hard_pred_dtype_uint8(self, tmp_prediction_dir, sample_predictions):
        """Saved hard predictions must be uint8."""
        hard_pred, soft_pred = sample_predictions
        path = save_volume_prediction(
            output_dir=tmp_prediction_dir,
            volume_name="vol_01",
            hard_pred=hard_pred,
            soft_pred=soft_pred,
        )
        loaded_hard, _ = load_volume_prediction(path)
        assert loaded_hard.dtype == np.uint8

    def test_soft_pred_dtype_float16(self, tmp_prediction_dir, sample_predictions):
        """Saved soft predictions must be float16 (for compression)."""
        hard_pred, soft_pred = sample_predictions
        path = save_volume_prediction(
            output_dir=tmp_prediction_dir,
            volume_name="vol_01",
            hard_pred=hard_pred,
            soft_pred=soft_pred,
        )
        _, loaded_soft = load_volume_prediction(path)
        assert loaded_soft.dtype == np.float16

    def test_compressed_smaller_than_raw(self, tmp_prediction_dir, sample_predictions):
        """Compressed .npz must be smaller than raw arrays."""
        hard_pred, soft_pred = sample_predictions
        raw_size = hard_pred.nbytes + soft_pred.nbytes
        path = save_volume_prediction(
            output_dir=tmp_prediction_dir,
            volume_name="vol_01",
            hard_pred=hard_pred,
            soft_pred=soft_pred,
        )
        compressed_size = path.stat().st_size
        assert compressed_size < raw_size

    def test_roundtrip_hard_pred(self, tmp_prediction_dir, sample_predictions):
        """Saved hard predictions match after load."""
        hard_pred, soft_pred = sample_predictions
        path = save_volume_prediction(
            output_dir=tmp_prediction_dir,
            volume_name="vol_01",
            hard_pred=hard_pred,
            soft_pred=soft_pred,
        )
        loaded_hard, _ = load_volume_prediction(path)
        np.testing.assert_array_equal(loaded_hard, hard_pred)

    def test_roundtrip_soft_pred_approximate(self, tmp_prediction_dir, sample_predictions):
        """Soft predictions match approximately (float16 truncation)."""
        hard_pred, soft_pred = sample_predictions
        path = save_volume_prediction(
            output_dir=tmp_prediction_dir,
            volume_name="vol_01",
            hard_pred=hard_pred,
            soft_pred=soft_pred,
        )
        _, loaded_soft = load_volume_prediction(path)
        # float16 has ~3 decimal digits of precision
        np.testing.assert_allclose(loaded_soft.astype(np.float32), soft_pred, atol=1e-3)
