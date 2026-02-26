from __future__ import annotations

import numpy as np
import torch

from minivess.pipeline.inference import SlidingWindowInferenceRunner


class TestSlidingWindowInferenceRunner:
    """Tests for sliding window inference on full volumes."""

    def test_init_defaults(self) -> None:
        runner = SlidingWindowInferenceRunner(
            roi_size=(64, 64, 16),
            num_classes=2,
        )
        assert runner.roi_size == (64, 64, 16)
        assert runner.num_classes == 2
        assert runner.overlap == 0.25

    def test_init_custom_overlap(self) -> None:
        runner = SlidingWindowInferenceRunner(
            roi_size=(96, 96, 24),
            num_classes=2,
            overlap=0.5,
        )
        assert runner.overlap == 0.5

    def test_predict_volume_returns_numpy(self) -> None:
        """predict_volume should return an integer numpy array."""

        class _FakeModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                b, c, d, h, w = x.shape
                return torch.randn(b, 2, d, h, w)

        model = _FakeModel()
        runner = SlidingWindowInferenceRunner(
            roi_size=(16, 16, 8),
            num_classes=2,
        )
        image_tensor = torch.randn(1, 1, 32, 32, 16)
        pred = runner.predict_volume(model, image_tensor, device="cpu")

        assert isinstance(pred, np.ndarray)
        assert pred.dtype in (np.int32, np.int64)
        assert pred.ndim == 3  # (D, H, W), no batch or channel dim

    def test_predict_volume_shape_matches_input(self) -> None:
        """Output spatial dims must match input spatial dims."""

        class _FakeModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                b, c, d, h, w = x.shape
                return torch.randn(b, 2, d, h, w)

        model = _FakeModel()
        runner = SlidingWindowInferenceRunner(
            roi_size=(16, 16, 8),
            num_classes=2,
        )
        image_tensor = torch.randn(1, 1, 32, 32, 16)
        pred = runner.predict_volume(model, image_tensor, device="cpu")

        assert pred.shape == (32, 32, 16)

    def test_predict_volume_binary_values(self) -> None:
        """For 2-class segmentation, output should only contain 0 and 1."""

        class _FakeModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                b, c, d, h, w = x.shape
                return torch.randn(b, 2, d, h, w)

        model = _FakeModel()
        runner = SlidingWindowInferenceRunner(
            roi_size=(16, 16, 8),
            num_classes=2,
        )
        image_tensor = torch.randn(1, 1, 32, 32, 16)
        pred = runner.predict_volume(model, image_tensor, device="cpu")

        unique_values = set(np.unique(pred))
        assert unique_values.issubset({0, 1})

    def test_infer_dataset_returns_predictions_and_labels(self) -> None:
        """infer_dataset should return parallel lists of preds and labels."""

        class _FakeModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                b, c, d, h, w = x.shape
                return torch.randn(b, 2, d, h, w)

        model = _FakeModel()
        runner = SlidingWindowInferenceRunner(
            roi_size=(16, 16, 8),
            num_classes=2,
        )

        # Simulate a validation loader that yields full volumes
        class _FakeLoader:
            def __iter__(self):
                for _ in range(3):
                    yield {
                        "image": torch.randn(1, 1, 32, 32, 16),
                        "label": torch.randint(0, 2, (1, 1, 32, 32, 16)),
                    }

        preds, labels = runner.infer_dataset(model, _FakeLoader(), device="cpu")

        assert len(preds) == 3
        assert len(labels) == 3
        for pred, label in zip(preds, labels, strict=True):
            assert isinstance(pred, np.ndarray)
            assert isinstance(label, np.ndarray)
            assert pred.shape == label.shape


class TestSlidingWindowWithAdapterModel:
    """Test that inference works with ModelAdapter (wraps SegmentationOutput)."""

    def test_predict_volume_with_adapter_model(self) -> None:
        """Should handle models that return SegmentationOutput."""
        from minivess.adapters.base import SegmentationOutput

        class _FakeAdapterModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> SegmentationOutput:
                b, c, d, h, w = x.shape
                logits = torch.randn(b, 2, d, h, w)
                return SegmentationOutput(
                    prediction=torch.softmax(logits, dim=1),
                    logits=logits,
                )

        model = _FakeAdapterModel()
        runner = SlidingWindowInferenceRunner(
            roi_size=(16, 16, 8),
            num_classes=2,
        )
        image_tensor = torch.randn(1, 1, 32, 32, 16)
        pred = runner.predict_volume(model, image_tensor, device="cpu")

        assert isinstance(pred, np.ndarray)
        assert pred.shape == (32, 32, 16)
