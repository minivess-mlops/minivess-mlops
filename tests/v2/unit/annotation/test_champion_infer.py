"""Tests for champion pre-segmentation bridge (T-ANN.2.1).

The ChampionInfer class wraps BentoML champion serving for MONAI Label.
When an annotator opens a volume, MONAI Label calls ChampionInfer to
get an initial mask from the deployed champion model.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from minivess.annotation.champion_infer import ChampionInfer


class TestChampionInfer:
    """Test the BentoML → MONAI Label pre-segmentation bridge."""

    def test_infer_returns_dict_with_mask(self, tmp_path: Path) -> None:
        """infer() must return a dict containing a 'pred' numpy array."""
        infer = ChampionInfer(bentoml_url="http://fake:3333")
        volume = np.random.default_rng(42).random((16, 16, 8), dtype=np.float32)

        with patch.object(infer, "_call_bentoml") as mock_call:
            mock_call.return_value = np.ones((16, 16, 8), dtype=np.int64)
            result = infer.run_inference(volume)

        assert "pred" in result
        assert isinstance(result["pred"], np.ndarray)

    def test_infer_calls_bentoml_client(self, tmp_path: Path) -> None:
        """BentoML predict should be called exactly once."""
        infer = ChampionInfer(bentoml_url="http://fake:3333")
        volume = np.random.default_rng(42).random((16, 16, 8), dtype=np.float32)

        with patch.object(infer, "_call_bentoml") as mock_call:
            mock_call.return_value = np.ones((16, 16, 8), dtype=np.int64)
            infer.run_inference(volume)
            mock_call.assert_called_once()

    def test_infer_handles_bentoml_unavailable(self, tmp_path: Path) -> None:
        """When BentoML is unreachable, return zero mask (graceful degradation)."""
        infer = ChampionInfer(bentoml_url="http://fake:3333")
        volume = np.random.default_rng(42).random((16, 16, 8), dtype=np.float32)

        with patch.object(infer, "_call_bentoml") as mock_call:
            mock_call.side_effect = ConnectionError("BentoML unreachable")
            result = infer.run_inference(volume)

        assert "pred" in result
        # Fallback is a zero mask
        assert np.all(result["pred"] == 0)

    def test_output_mask_shape_matches_input(self) -> None:
        """Output mask spatial shape must match input volume."""
        infer = ChampionInfer(bentoml_url="http://fake:3333")
        shape = (32, 24, 10)
        volume = np.random.default_rng(42).random(shape, dtype=np.float32)

        with patch.object(infer, "_call_bentoml") as mock_call:
            mock_call.return_value = np.ones(shape, dtype=np.int64)
            result = infer.run_inference(volume)

        assert result["pred"].shape == shape

    def test_output_is_binary_mask(self) -> None:
        """Output values must be 0 or 1 only."""
        infer = ChampionInfer(bentoml_url="http://fake:3333")
        volume = np.random.default_rng(42).random((16, 16, 8), dtype=np.float32)

        with patch.object(infer, "_call_bentoml") as mock_call:
            mock_mask = np.zeros((16, 16, 8), dtype=np.int64)
            mock_mask[4:12, 4:12, 2:6] = 1
            mock_call.return_value = mock_mask
            result = infer.run_inference(volume)

        unique_values = set(np.unique(result["pred"]))
        assert unique_values <= {0, 1}, f"Non-binary values: {unique_values}"
