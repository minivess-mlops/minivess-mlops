from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch.amp import GradScaler, autocast
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from minivess.pipeline.loss_functions import build_loss_function

if TYPE_CHECKING:
    from pathlib import Path

    from minivess.adapters.base import ModelAdapter
    from minivess.config.models import TrainingConfig
    from minivess.observability.tracking import ExperimentTracker

logger = logging.getLogger(__name__)


@dataclass
class EpochResult:
    """Metrics from a single training or validation epoch."""

    loss: float
    metrics: dict[str, float] = field(default_factory=dict)


class SegmentationTrainer:
    """Model-agnostic training engine for 3D segmentation.

    Supports mixed precision, gradient clipping, early stopping,
    and warmup + cosine annealing schedule.
    """

    def __init__(
        self,
        model: ModelAdapter,
        config: TrainingConfig,
        *,
        loss_name: str = "dice_ce",
        device: str | torch.device = "cpu",
        tracker: ExperimentTracker | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(device)
        self.model.to(self.device)
        self.tracker = tracker

        self.criterion = build_loss_function(loss_name)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = GradScaler(enabled=config.mixed_precision)

        self._best_val_loss = float("inf")
        self._patience_counter = 0

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build the optimizer from training config."""
        if self.config.optimizer == "adamw":
            return AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        if self.config.optimizer == "sgd":
            return SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9,
            )
        msg = f"Unknown optimizer: {self.config.optimizer}"
        raise ValueError(msg)

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Build warmup + cosine annealing LR scheduler."""
        warmup = LinearLR(
            self.optimizer,
            start_factor=0.01,
            total_iters=self.config.warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.max_epochs - self.config.warmup_epochs,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[self.config.warmup_epochs],
        )

    def train_epoch(self, loader: Any) -> EpochResult:
        """Run one training epoch with mixed precision.

        Parameters
        ----------
        loader:
            Training DataLoader yielding batches with ``"image"`` and
            ``"label"`` keys.

        Returns
        -------
        EpochResult
            Average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        num_batches = 0

        for batch in loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            with autocast(
                device_type=self.device.type,
                enabled=self.config.mixed_precision,
            ):
                output = self.model(images)
                loss = self.criterion(output.logits, labels)

            self.scaler.scale(loss).backward()
            if self.config.gradient_clip_val > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_val,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / max(num_batches, 1)
        return EpochResult(loss=avg_loss)

    @torch.no_grad()
    def validate_epoch(self, loader: Any) -> EpochResult:
        """Run one validation epoch.

        Parameters
        ----------
        loader:
            Validation DataLoader yielding batches with ``"image"`` and
            ``"label"`` keys.

        Returns
        -------
        EpochResult
            Average validation loss for the epoch.
        """
        self.model.eval()
        running_loss = 0.0
        num_batches = 0

        for batch in loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            with autocast(
                device_type=self.device.type,
                enabled=self.config.mixed_precision,
            ):
                output = self.model(images)
                loss = self.criterion(output.logits, labels)

            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / max(num_batches, 1)
        return EpochResult(loss=avg_loss)

    def fit(
        self,
        train_loader: Any,
        val_loader: Any,
        *,
        checkpoint_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Full training loop with early stopping.

        Parameters
        ----------
        train_loader:
            Training DataLoader.
        val_loader:
            Validation DataLoader.
        checkpoint_dir:
            Directory for saving best model checkpoints. If ``None``,
            no checkpoints are saved.

        Returns
        -------
        dict[str, Any]
            Summary with ``best_val_loss``, ``final_epoch``, and ``history``.
        """
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        final_epoch = 0

        for epoch in range(self.config.max_epochs):
            train_result = self.train_epoch(train_loader)
            val_result = self.validate_epoch(val_loader)
            self.scheduler.step()

            history["train_loss"].append(train_result.loss)
            history["val_loss"].append(val_result.loss)
            final_epoch = epoch + 1

            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                "Epoch %d/%d — train_loss: %.4f, val_loss: %.4f, lr: %.2e",
                epoch + 1,
                self.config.max_epochs,
                train_result.loss,
                val_result.loss,
                current_lr,
            )

            # Log to MLflow if tracker is present
            if self.tracker is not None:
                self.tracker.log_epoch_metrics(
                    {
                        "train_loss": train_result.loss,
                        "val_loss": val_result.loss,
                        "learning_rate": current_lr,
                    },
                    step=epoch + 1,
                )

            # Early stopping + best checkpoint
            if val_result.loss < self._best_val_loss:
                self._best_val_loss = val_result.loss
                self._patience_counter = 0
                if checkpoint_dir is not None:
                    best_path = checkpoint_dir / "best_model.pth"
                    self.model.save_checkpoint(best_path)
                    logger.info("Saved best checkpoint to %s", best_path)
                    if self.tracker is not None:
                        self.tracker.log_artifact(
                            best_path, artifact_path="checkpoints"
                        )
            else:
                self._patience_counter += 1
                if self._patience_counter >= self.config.early_stopping_patience:
                    logger.info(
                        "Early stopping at epoch %d (patience=%d)",
                        epoch + 1,
                        self.config.early_stopping_patience,
                    )
                    break

        return {
            "best_val_loss": self._best_val_loss,
            "final_epoch": final_epoch,
            "history": history,
        }
