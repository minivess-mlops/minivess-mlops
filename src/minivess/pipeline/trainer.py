from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from torch.amp import GradScaler, autocast
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from minivess.pipeline.loss_functions import build_loss_function
from minivess.pipeline.multi_metric_tracker import (
    MetricCheckpoint,
    MetricDirection,
    MetricHistory,
    MetricTracker,
    MultiMetricTracker,
    save_metric_checkpoint,
)
from minivess.pipeline.validation_metrics import compute_compound_masd_cldice

if TYPE_CHECKING:
    from pathlib import Path

    from torch import nn

    from minivess.adapters.base import ModelAdapter
    from minivess.config.models import TrainingConfig
    from minivess.observability.tracking import ExperimentTracker
    from minivess.pipeline.metrics import SegmentationMetrics

logger = logging.getLogger(__name__)


@dataclass
class EpochResult:
    """Metrics from a single training or validation epoch."""

    loss: float
    metrics: dict[str, float] = field(default_factory=dict)


def _build_multi_tracker(config: TrainingConfig) -> MultiMetricTracker:
    """Build a :class:`MultiMetricTracker` from :class:`TrainingConfig`.

    Parameters
    ----------
    config:
        Training configuration containing ``checkpoint`` sub-config.

    Returns
    -------
    MultiMetricTracker
        Ready-to-use tracker built from ``config.checkpoint``.
    """
    ckpt_cfg = config.checkpoint
    trackers: list[MetricTracker] = []
    for m in ckpt_cfg.tracked_metrics:
        direction = (
            MetricDirection.MINIMIZE
            if m.direction == "minimize"
            else MetricDirection.MAXIMIZE
        )
        trackers.append(
            MetricTracker(
                name=m.name,
                direction=direction,
                patience=m.patience,
                min_delta=ckpt_cfg.min_delta,
            )
        )
    return MultiMetricTracker(
        trackers=trackers,
        primary_metric=ckpt_cfg.primary_metric,
        early_stopping_strategy=ckpt_cfg.early_stopping_strategy,
        min_epochs=ckpt_cfg.min_epochs,
    )


class SegmentationTrainer:
    """Model-agnostic training engine for 3D segmentation.

    Supports mixed precision, gradient clipping, early stopping,
    and warmup + cosine annealing schedule.

    Parameters
    ----------
    model:
        ModelAdapter to train.
    config:
        Training configuration.
    loss_name:
        Name of the loss function to build (ignored if ``criterion`` is provided).
    device:
        Device to train on.
    tracker:
        Optional experiment tracker (e.g., MLflow).
    metrics:
        Optional segmentation metrics tracker. If provided, metrics are
        computed each epoch and included in ``EpochResult.metrics``.
    criterion:
        Optional pre-built loss function. If provided, ``loss_name`` is ignored.
    optimizer:
        Optional pre-built optimizer. If provided, the internal optimizer
        builder is skipped.
    scheduler:
        Optional pre-built LR scheduler. If provided, the internal scheduler
        builder is skipped. Note: if you inject a scheduler, you should also
        inject its corresponding optimizer.
    """

    def __init__(
        self,
        model: ModelAdapter,
        config: TrainingConfig,
        *,
        loss_name: str = "dice_ce",
        device: str | torch.device = "cpu",
        tracker: ExperimentTracker | None = None,
        metrics: SegmentationMetrics | None = None,
        criterion: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        val_roi_size: tuple[int, int, int] | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(device)
        self.model.to(self.device)
        self.tracker = tracker
        self.metrics = metrics
        self.val_roi_size = val_roi_size

        self.criterion = (
            criterion if criterion is not None else build_loss_function(loss_name)
        )
        self.optimizer = optimizer if optimizer is not None else self._build_optimizer()
        self.scheduler = scheduler if scheduler is not None else self._build_scheduler()
        self.scaler = GradScaler(enabled=config.mixed_precision)

        self._multi_tracker: MultiMetricTracker = _build_multi_tracker(config)
        self._metric_history: MetricHistory = MetricHistory()

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

            if self.metrics is not None:
                with torch.no_grad():
                    self.metrics.update(output.logits, labels)

        avg_loss = running_loss / max(num_batches, 1)
        epoch_metrics: dict[str, float] = {}
        if self.metrics is not None:
            epoch_metrics = self.metrics.compute().to_dict()
            self.metrics.reset()
        return EpochResult(loss=avg_loss, metrics=epoch_metrics)

    @torch.no_grad()
    def validate_epoch(
        self, loader: Any, *, compute_extended: bool = False
    ) -> EpochResult:
        """Run one validation epoch.

        Parameters
        ----------
        loader:
            Validation DataLoader yielding batches with ``"image"`` and
            ``"label"`` keys.
        compute_extended:
            If True, compute MetricsReloaded metrics (clDice, MASD, compound)
            on CPU after sliding window inference. Adds ~30% overhead.

        Returns
        -------
        EpochResult
            Average validation loss for the epoch.
        """
        self.model.eval()
        running_loss = 0.0
        num_batches = 0

        # Collect full-volume predictions for MetricsReloaded (CPU numpy)
        collected_preds: list[np.ndarray] = []
        collected_labels: list[np.ndarray] = []

        for batch in loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            with autocast(
                device_type=self.device.type,
                enabled=self.config.mixed_precision,
            ):
                # Use sliding window inference when val_roi_size is set,
                # needed because full 512x512xZ volumes exceed GPU memory.
                if self.val_roi_size is not None:
                    def _model_fn(x):
                        return self.model(x).logits

                    logits = sliding_window_inference(
                        images,
                        roi_size=self.val_roi_size,
                        sw_batch_size=4,
                        predictor=_model_fn,
                        overlap=0.25,
                    )
                else:
                    output = self.model(images)
                    logits = output.logits
                loss = self.criterion(logits, labels)

            running_loss += loss.item()
            num_batches += 1

            if self.metrics is not None:
                self.metrics.update(logits, labels)

            if compute_extended:
                # Move to CPU and convert to binary predictions
                pred_probs = torch.softmax(logits, dim=1)
                pred_binary = pred_probs[:, 1:].argmax(dim=1) if logits.shape[1] > 2 else (pred_probs[:, 1] > 0.5).long()
                for b in range(images.shape[0]):
                    pred_np = pred_binary[b].cpu().numpy().astype(np.uint8)
                    label_np = labels[b, 0].cpu().numpy().astype(np.uint8)
                    collected_preds.append(pred_np)
                    collected_labels.append(label_np)

        avg_loss = running_loss / max(num_batches, 1)
        epoch_metrics: dict[str, float] = {}
        if self.metrics is not None:
            epoch_metrics = self.metrics.compute().to_dict()
            self.metrics.reset()

        # Compute MetricsReloaded extended metrics on CPU
        if compute_extended and collected_preds:
            extended = self._compute_extended_metrics(collected_preds, collected_labels)
            epoch_metrics.update(extended)

        return EpochResult(loss=avg_loss, metrics=epoch_metrics)

    def _compute_extended_metrics(
        self,
        predictions: list[np.ndarray],
        labels: list[np.ndarray],
    ) -> dict[str, float]:
        """Compute MetricsReloaded metrics (clDice, MASD) + compound.

        Returns metric keys WITHOUT the 'val_' prefix (added by fit()).
        """
        try:
            from minivess.pipeline.evaluation import EvaluationRunner
        except ImportError:
            logger.warning("MetricsReloaded not available, skipping extended metrics")
            return {}

        runner = EvaluationRunner(include_expensive=False)
        per_vol_cldice: list[float] = []
        per_vol_masd: list[float] = []
        per_vol_dsc: list[float] = []

        for pred, label in zip(predictions, labels, strict=True):
            try:
                vol_metrics = runner.evaluate_volume(pred, label)
                per_vol_cldice.append(vol_metrics.get("centreline_dsc", float("nan")))
                per_vol_masd.append(vol_metrics.get("measured_masd", float("nan")))
                per_vol_dsc.append(vol_metrics.get("dsc", float("nan")))
            except Exception:
                logger.exception("MetricsReloaded evaluation failed for a volume")
                per_vol_cldice.append(float("nan"))
                per_vol_masd.append(float("nan"))
                per_vol_dsc.append(float("nan"))

        mean_cldice = float(np.nanmean(per_vol_cldice)) if per_vol_cldice else float("nan")
        mean_masd = float(np.nanmean(per_vol_masd)) if per_vol_masd else float("nan")

        compound = compute_compound_masd_cldice(masd=mean_masd, cldice=mean_cldice)

        return {
            "cldice": mean_cldice,
            "masd": mean_masd,
            "compound_masd_cldice": compound,
        }

    def fit(
        self,
        train_loader: Any,
        val_loader: Any,
        *,
        checkpoint_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Full training loop with multi-metric early stopping.

        Uses :class:`MultiMetricTracker` built from ``config.checkpoint`` to
        determine when improvement has occurred and when to early-stop.

        Parameters
        ----------
        train_loader:
            Training DataLoader.
        val_loader:
            Validation DataLoader.
        checkpoint_dir:
            Directory for saving checkpoints. If ``None``, no checkpoints
            are saved (metric history will still be tracked in-memory).

        Returns
        -------
        dict[str, Any]
            Summary with ``best_val_loss`` (backward compat), ``final_epoch``,
            ``history``, and ``best_metrics``.
        """
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        final_epoch = 0
        ckpt_cfg = self.config.checkpoint
        epoch_start_time = time.perf_counter()

        # Determine if extended metrics (MetricsReloaded) are needed
        # by checking if any tracked metric requires them
        _extended_metric_names = {"val_cldice", "val_masd", "val_compound_masd_cldice"}
        _tracked_names = {m.name for m in ckpt_cfg.tracked_metrics}
        needs_extended = bool(_tracked_names & _extended_metric_names)

        for epoch in range(self.config.max_epochs):
            t0 = time.perf_counter()
            train_result = self.train_epoch(train_loader)
            val_result = self.validate_epoch(
                val_loader, compute_extended=needs_extended
            )
            self.scheduler.step()
            epoch_wall_time = time.perf_counter() - t0

            history["train_loss"].append(train_result.loss)
            history["val_loss"].append(val_result.loss)
            final_epoch = epoch + 1

            # Build full metric dict for this epoch
            all_metrics: dict[str, float] = {
                "train_loss": train_result.loss,
                "val_loss": val_result.loss,
            }
            for k, v in train_result.metrics.items():
                all_metrics[f"train_{k}"] = v
            for k, v in val_result.metrics.items():
                all_metrics[f"val_{k}"] = v

            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                "Epoch %d/%d — train_loss: %.4f, val_loss: %.4f, lr: %.2e",
                epoch + 1,
                self.config.max_epochs,
                train_result.loss,
                val_result.loss,
                current_lr,
            )

            # Log to MLflow / experiment tracker if present
            if self.tracker is not None:
                epoch_log: dict[str, float] = {
                    "train_loss": train_result.loss,
                    "val_loss": val_result.loss,
                    "learning_rate": current_lr,
                }
                for k, v in train_result.metrics.items():
                    epoch_log[f"train_{k}"] = v
                for k, v in val_result.metrics.items():
                    epoch_log[f"val_{k}"] = v
                self.tracker.log_epoch_metrics(epoch_log, step=epoch + 1)

            # Update multi-metric tracker and save per-metric best checkpoints
            improved_metrics = self._multi_tracker.update(all_metrics, epoch)
            cumulative_wall_time = time.perf_counter() - epoch_start_time

            if checkpoint_dir is not None and improved_metrics:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                for metric_name in improved_metrics:
                    tracker_obj = next(
                        t for t in self._multi_tracker.trackers if t.name == metric_name
                    )
                    ckpt_meta = MetricCheckpoint(
                        epoch=epoch,
                        metrics=all_metrics,
                        metric_name=metric_name,
                        metric_value=all_metrics.get(metric_name, float("nan")),
                        metric_direction=tracker_obj.direction.value,
                        train_loss=train_result.loss,
                        val_loss=val_result.loss,
                        wall_time_sec=cumulative_wall_time,
                        config_snapshot=self.config.model_dump(mode="json"),
                    )
                    safe_name = metric_name.replace("/", "_")
                    best_path = checkpoint_dir / f"best_{safe_name}.pth"
                    save_metric_checkpoint(
                        path=best_path,
                        model_state_dict=self.model.state_dict(),
                        optimizer_state_dict=self.optimizer.state_dict(),
                        scheduler_state_dict=self.scheduler.state_dict(),
                        checkpoint=ckpt_meta,
                        scaler_state_dict=self.scaler.state_dict(),
                    )
                    logger.info(
                        "Saved best checkpoint for '%s' to %s", metric_name, best_path
                    )
                    if self.tracker is not None:
                        self.tracker.log_artifact(
                            best_path, artifact_path="checkpoints"
                        )

            # Save last.pth if configured
            if checkpoint_dir is not None and ckpt_cfg.save_last:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                last_ckpt_meta = MetricCheckpoint(
                    epoch=epoch,
                    metrics=all_metrics,
                    metric_name="last",
                    metric_value=val_result.loss,
                    metric_direction="minimize",
                    train_loss=train_result.loss,
                    val_loss=val_result.loss,
                    wall_time_sec=cumulative_wall_time,
                    config_snapshot=self.config.model_dump(mode="json"),
                )
                last_path = checkpoint_dir / "last.pth"
                save_metric_checkpoint(
                    path=last_path,
                    model_state_dict=self.model.state_dict(),
                    optimizer_state_dict=self.optimizer.state_dict(),
                    scheduler_state_dict=self.scheduler.state_dict(),
                    checkpoint=last_ckpt_meta,
                    scaler_state_dict=self.scaler.state_dict(),
                )

            # Record epoch in history
            self._metric_history.record_epoch(
                epoch=epoch,
                metrics=all_metrics,
                wall_time_sec=epoch_wall_time,
                checkpoints_saved=improved_metrics,
            )

            # Save metric_history.json if configured
            if checkpoint_dir is not None and ckpt_cfg.save_history:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                self._metric_history.save_json(checkpoint_dir / "metric_history.json")

            # Early stopping decision via MultiMetricTracker
            if self._multi_tracker.should_stop(epoch):
                logger.info(
                    "Early stopping at epoch %d (strategy=%s)",
                    epoch + 1,
                    ckpt_cfg.early_stopping_strategy,
                )
                break

        # Upload last.pth and metric_history.json to MLflow
        if self.tracker is not None and checkpoint_dir is not None:
            last_path = checkpoint_dir / "last.pth"
            if last_path.exists():
                self.tracker.log_artifact(last_path, artifact_path="checkpoints")
            history_path = checkpoint_dir / "metric_history.json"
            if history_path.exists():
                self.tracker.log_artifact(history_path, artifact_path="history")

        # Backward-compatible best_val_loss: use primary tracker's best value
        # when primary metric is val_loss, otherwise fall back to best val_loss tracker
        primary_tracker = self._multi_tracker.get_primary_tracker()
        if primary_tracker.name == "val_loss":
            best_val_loss = primary_tracker.best_value
        else:
            # Try to find a val_loss tracker
            val_loss_trackers = [
                t for t in self._multi_tracker.trackers if t.name == "val_loss"
            ]
            best_val_loss = (
                val_loss_trackers[0].best_value
                if val_loss_trackers
                else primary_tracker.best_value
            )

        return {
            "best_val_loss": best_val_loss,
            "final_epoch": final_epoch,
            "history": history,
            "best_metrics": {
                tracker.name: tracker.best_value
                for tracker in self._multi_tracker.trackers
            },
        }
