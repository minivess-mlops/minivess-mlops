"""Tests for OOM detection and diagnostics in SegmentationTrainer.

Verifies that:
- torch.cuda.OutOfMemoryError during training produces structured diagnostics
- RuntimeError with "CUDA out of memory" (legacy) also triggers diagnostics
- OOM is always re-raised after logging (Rule #25: loud failures)

Issue: #940 (SAM3 batch_size=1 fix)
Plan: docs/planning/sam3-batch-size-1-and-robustification-plan.xml Task 2.1
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from minivess.config.models import CheckpointConfig, TrainingConfig

# Ensure the trainer logger propagates so caplog can capture records.
# Other tests may disable propagation (e.g., Prefect integration), causing
# caplog.records to be empty when this test runs in the full suite.
_trainer_logger = logging.getLogger("minivess.pipeline.trainer")


@pytest.fixture(autouse=True)
def _ensure_logger_propagation():
    """Ensure trainer logger propagates during these tests."""
    old = _trainer_logger.propagate
    _trainer_logger.propagate = True
    yield
    _trainer_logger.propagate = old


class _StubOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _OOMModel(nn.Module):
    """Model that raises OOM on forward pass."""

    def __init__(self, *, error_type: str = "pytorch2") -> None:
        super().__init__()
        self.conv = nn.Conv3d(1, 2, kernel_size=3, padding=1)
        self._error_type = error_type

    def forward(self, x: torch.Tensor) -> _StubOutput:
        if self._error_type == "pytorch2":
            raise torch.cuda.OutOfMemoryError(
                "CUDA out of memory. Tried to allocate 2.00 GiB"
            )
        # Legacy RuntimeError path
        raise RuntimeError(
            "CUDA out of memory. Tried to allocate 2.00 GiB "
            "(GPU 0; 23.65 GiB total capacity)"
        )

    def parameters(self, recurse: bool = True) -> Any:
        return super().parameters(recurse=recurse)

    def to(self, device: Any) -> _OOMModel:
        return super().to(device)  # type: ignore[return-value]

    def train(self, mode: bool = True) -> _OOMModel:
        return super().train(mode)  # type: ignore[return-value]

    def get_config(self) -> Any:
        """Stub model_info returning family."""
        info = MagicMock()
        info.family = "sam3_topolora"
        return info


def _make_batches(n: int) -> list[dict[str, torch.Tensor]]:
    """Create n synthetic training batches."""
    return [
        {
            "image": torch.randn(1, 1, 8, 8, 4),
            "label": torch.randint(0, 2, (1, 1, 8, 8, 4)).float(),
        }
        for _ in range(n)
    ]


def _make_trainer(
    *,
    error_type: str = "pytorch2",
    batch_size: int = 2,
    grad_accum_steps: int = 1,
) -> Any:
    """Create a SegmentationTrainer with an OOM-raising model."""
    from minivess.pipeline.trainer import SegmentationTrainer

    model = _OOMModel(error_type=error_type)
    config = TrainingConfig(
        max_epochs=2,
        batch_size=batch_size,
        learning_rate=0.01,
        warmup_epochs=0,
        mixed_precision=False,
        gradient_accumulation_steps=grad_accum_steps,
        checkpoint=CheckpointConfig(),
    )
    criterion = nn.MSELoss()
    return SegmentationTrainer(
        model=model,
        config=config,
        device="cpu",
        criterion=criterion,
    )


class TestOOMDetectionDiagnostics:
    """OOM errors must produce structured diagnostic messages."""

    def test_oom_detection_logs_diagnostics(self, caplog: pytest.LogCaptureFixture) -> None:
        """Mock OOM during training, verify diagnostic includes model info and batch_size."""
        trainer = _make_trainer(error_type="pytorch2", batch_size=2, grad_accum_steps=4)
        batches = _make_batches(4)

        with (
            caplog.at_level(logging.ERROR, logger="minivess.pipeline.trainer"),
            pytest.raises(torch.cuda.OutOfMemoryError),
        ):
            trainer.train_epoch(batches)

        # Verify diagnostic message content
        assert len(caplog.records) >= 1, "Expected at least one ERROR log on OOM"
        oom_log = caplog.records[-1]
        msg = oom_log.message

        # Must include model class (since model_family may not be available)
        assert "_OOMModel" in msg or "sam3_topolora" in msg, (
            f"OOM diagnostic must include model info, got: {msg}"
        )
        # Must include batch_size
        assert "batch_size" in msg, f"OOM diagnostic must include batch_size, got: {msg}"
        assert "2" in msg, f"OOM diagnostic must include batch_size value, got: {msg}"
        # Must include gradient_accumulation_steps
        assert "gradient_accumulation_steps" in msg or "grad_accum" in msg, (
            f"OOM diagnostic must include grad_accum info, got: {msg}"
        )

    def test_oom_detection_logs_diagnostics_legacy_runtime_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Legacy RuntimeError with 'CUDA out of memory' also triggers diagnostics."""
        trainer = _make_trainer(error_type="legacy", batch_size=1, grad_accum_steps=1)
        batches = _make_batches(2)

        with (
            caplog.at_level(logging.ERROR, logger="minivess.pipeline.trainer"),
            pytest.raises(RuntimeError, match="CUDA out of memory"),
        ):
            trainer.train_epoch(batches)

        assert len(caplog.records) >= 1, "Expected at least one ERROR log on legacy OOM"
        msg = caplog.records[-1].message
        assert "batch_size" in msg, f"Legacy OOM must also produce diagnostics, got: {msg}"


class TestOOMDetectionReraises:
    """OOM must always be re-raised after logging (Rule #25: loud failures)."""

    def test_oom_detection_reraises_pytorch2(self) -> None:
        """torch.cuda.OutOfMemoryError must be re-raised, not swallowed."""
        trainer = _make_trainer(error_type="pytorch2")
        batches = _make_batches(2)

        with pytest.raises(torch.cuda.OutOfMemoryError):
            trainer.train_epoch(batches)

    def test_oom_detection_reraises_legacy(self) -> None:
        """Legacy RuntimeError('CUDA out of memory') must be re-raised, not swallowed."""
        trainer = _make_trainer(error_type="legacy")
        batches = _make_batches(2)

        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            trainer.train_epoch(batches)

    def test_non_oom_runtime_error_not_caught(self) -> None:
        """Non-OOM RuntimeErrors must NOT be caught by the OOM handler."""

        class _NonOOMModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv3d(1, 2, kernel_size=3, padding=1)

            def forward(self, x: torch.Tensor) -> _StubOutput:
                raise RuntimeError("Some other CUDA error")

            def parameters(self, recurse: bool = True) -> Any:
                return super().parameters(recurse=recurse)

            def to(self, device: Any) -> _NonOOMModel:
                return super().to(device)  # type: ignore[return-value]

            def train(self, mode: bool = True) -> _NonOOMModel:
                return super().train(mode)  # type: ignore[return-value]

        from minivess.pipeline.trainer import SegmentationTrainer

        model = _NonOOMModel()
        config = TrainingConfig(
            max_epochs=2,
            batch_size=1,
            learning_rate=0.01,
            warmup_epochs=0,
            mixed_precision=False,
            checkpoint=CheckpointConfig(),
        )
        trainer = SegmentationTrainer(
            model=model,
            config=config,
            device="cpu",
            criterion=nn.MSELoss(),
        )
        batches = _make_batches(2)

        # Non-OOM RuntimeError should propagate without OOM diagnostics
        with pytest.raises(RuntimeError, match="Some other CUDA error"):
            trainer.train_epoch(batches)


class TestOOMDiagnosticContent:
    """Verify diagnostic message includes VRAM-relevant information."""

    def test_oom_diagnostic_suggests_batch_reduction(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """OOM diagnostic must suggest reducing batch_size."""
        trainer = _make_trainer(error_type="pytorch2", batch_size=2)
        batches = _make_batches(2)

        with (
            caplog.at_level(logging.ERROR, logger="minivess.pipeline.trainer"),
            pytest.raises(torch.cuda.OutOfMemoryError),
        ):
            trainer.train_epoch(batches)

        msg = caplog.records[-1].message
        assert "reduce" in msg.lower() or "decrease" in msg.lower() or "lower" in msg.lower(), (
            f"OOM diagnostic must suggest batch_size reduction, got: {msg}"
        )

    def test_oom_diagnostic_with_model_family_from_get_config(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When model has get_config(), diagnostic should include model family."""
        trainer = _make_trainer(error_type="pytorch2", batch_size=2)
        batches = _make_batches(2)

        with (
            caplog.at_level(logging.ERROR, logger="minivess.pipeline.trainer"),
            pytest.raises(torch.cuda.OutOfMemoryError),
        ):
            trainer.train_epoch(batches)

        msg = caplog.records[-1].message
        assert "sam3_topolora" in msg, (
            f"OOM diagnostic should include model family from get_config(), got: {msg}"
        )
