"""Tests for SegmentationTrainer OOM diagnostics path.

T4 from double-check plan: verify _log_oom_diagnostics() doesn't crash on CPU,
_is_oom_error() classifies correctly, and OOM is always re-raised.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from minivess.adapters.base import ModelAdapter, SegmentationOutput
from minivess.config.models import TrainingConfig
from minivess.pipeline.trainer import SegmentationTrainer


class _DummyAdapter(ModelAdapter):
    """Minimal adapter for OOM diagnostics testing."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(4, 2)

    def forward(self, images: Tensor, **kwargs: Any) -> SegmentationOutput:
        b = images.shape[0]
        logits = torch.randn(b, 2, 4, 4, 3)
        return self._build_output(logits, "dummy")

    def get_config(self) -> Any:
        return self._build_config(variant="dummy")

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class _OOMAdapter(ModelAdapter):
    """Adapter that raises OOM on forward."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(4, 2)

    def forward(self, images: Tensor, **kwargs: Any) -> SegmentationOutput:
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    def get_config(self) -> Any:
        return self._build_config(variant="oom_test")

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class _FiniteCriterion(nn.Module):
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return torch.tensor(0.5, device=pred.device)


def _make_trainer(
    model: ModelAdapter | None = None,
    *,
    device: str = "cpu",
) -> SegmentationTrainer:
    config = TrainingConfig(
        max_epochs=2,
        batch_size=1,
        learning_rate=1e-4,
        mixed_precision=False,
        warmup_epochs=0,
        val_interval=1,
    )
    return SegmentationTrainer(
        model=model or _DummyAdapter(),
        config=config,
        criterion=_FiniteCriterion(),
        device=device,
    )


def _make_train_batch() -> list[dict[str, Tensor]]:
    return [
        {
            "image": torch.randn(1, 1, 4, 4, 3),
            "label": torch.randint(0, 2, (1, 1, 4, 4, 3)).float(),
        }
    ]


class TestLogOomDiagnostics:
    """_log_oom_diagnostics must not crash on CPU device."""

    def test_log_oom_diagnostics_cpu_device_no_crash(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        trainer = _make_trainer(device="cpu")
        exc = RuntimeError("CUDA out of memory")
        with caplog.at_level(logging.ERROR, logger="minivess.pipeline.trainer"):
            # Must not crash — VRAM section should say "unavailable"
            trainer._log_oom_diagnostics(exc)  # noqa: SLF001

        assert any("oom" in r.message.lower() for r in caplog.records), (
            f"Expected OOM diagnostic log, got: {[r.message for r in caplog.records]}"
        )
        assert any("unavailable" in r.message.lower() for r in caplog.records), (
            "CPU device should report VRAM as unavailable"
        )


class TestIsOomError:
    """_is_oom_error must correctly classify OOM exceptions."""

    def test_detects_legacy_runtime_error(self) -> None:
        exc = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        assert SegmentationTrainer._is_oom_error(exc) is True  # noqa: SLF001

    def test_rejects_unrelated_runtime_error(self) -> None:
        exc = RuntimeError("something else went wrong")
        assert SegmentationTrainer._is_oom_error(exc) is False  # noqa: SLF001

    def test_detects_pytorch2_oom_error(self) -> None:
        # torch.cuda.OutOfMemoryError is a subclass of RuntimeError
        exc = torch.cuda.OutOfMemoryError("CUDA out of memory")
        assert SegmentationTrainer._is_oom_error(exc) is True  # noqa: SLF001


class TestTrainEpochReRaisesOOM:
    """train_epoch must re-raise OOM after diagnostics, never swallow."""

    def test_oom_is_reraised(self, caplog: pytest.LogCaptureFixture) -> None:
        trainer = _make_trainer(model=_OOMAdapter())
        loader = _make_train_batch()

        with (
            caplog.at_level(logging.ERROR, logger="minivess.pipeline.trainer"),
            pytest.raises(RuntimeError, match="CUDA out of memory"),
        ):
            trainer.train_epoch(loader)

        # Diagnostics should have been called
        assert any("oom" in r.message.lower() for r in caplog.records), (
            "OOM diagnostics should have been logged before re-raise"
        )

    def test_non_oom_runtime_error_reraised_without_diagnostics(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Non-OOM RuntimeError should re-raise without OOM diagnostics."""

        class _ErrorAdapter(ModelAdapter):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Linear(4, 2)

            def forward(self, images: Tensor, **kwargs: Any) -> SegmentationOutput:
                raise RuntimeError("some other error")

            def get_config(self) -> Any:
                return self._build_config(variant="error_test")

            def trainable_parameters(self) -> int:
                return 0

        trainer = _make_trainer(model=_ErrorAdapter())
        loader = _make_train_batch()

        with (
            caplog.at_level(logging.ERROR, logger="minivess.pipeline.trainer"),
            pytest.raises(RuntimeError, match="some other error"),
        ):
            trainer.train_epoch(loader)

        # No OOM diagnostics for non-OOM errors
        assert not any("oom" in r.message.lower() for r in caplog.records), (
            "Non-OOM error should not trigger OOM diagnostics"
        )
