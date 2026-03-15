from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path  # Used at runtime in fit() for fallback profiler output_dir
from typing import TYPE_CHECKING, Any

# system_monitor.py is in scripts/ (not in the package).
# We use Any to avoid a hard dependency — the monitor is duck-typed.
# Expected interface: epoch_summary() -> dict[str, float]
#                    get_latest_snapshot() -> ResourceSnapshot | None
#                    gpu_status(snap) -> str
import numpy as np
import torch
import yaml
from monai.inferers import sliding_window_inference  # type: ignore[attr-defined]
from torch.amp import GradScaler, autocast  # type: ignore[attr-defined]
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.profiler import record_function

from minivess.pipeline.checkpoint_utils import atomic_text_write, atomic_torch_save
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
    from torch import nn

    from minivess.adapters.base import ModelAdapter
    from minivess.config.models import ProfilingConfig, TrainingConfig
    from minivess.observability.tracking import ExperimentTracker
    from minivess.pipeline.metrics import SegmentationMetrics

logger = logging.getLogger(__name__)


def validate_checkpoint_path(checkpoint_dir: Path) -> None:
    """Validate that checkpoint_dir is a Docker volume or test path.

    Rejects repo-relative paths like ``checkpoints/`` or ``./checkpoints``.
    Accepts:
    - Docker paths (``/app/checkpoints/...``)
    - pytest tmp_path (contains ``tmp`` or ``pytest`` in path parts)
    - Any path when ``MINIVESS_ALLOW_HOST=1`` is set

    Parameters
    ----------
    checkpoint_dir:
        Path to validate.

    Raises
    ------
    ValueError
        When the path appears to be repo-relative (not volume-mounted).

    See Also
    --------
    CLAUDE.md Rule #18 (volume mounts), Rule #19 (STOP protocol)
    docs/planning/minivess-vision-enforcement-plan-execution.xml (T-02)
    """
    import os
    from pathlib import Path as _Path

    if os.environ.get("MINIVESS_ALLOW_HOST") == "1":
        return

    path = (
        _Path(checkpoint_dir).resolve()
        if not checkpoint_dir.is_absolute()
        else checkpoint_dir
    )
    path_str = str(path)
    parts = path.parts

    # Accept Docker volume paths
    if path_str.startswith("/app/"):
        return

    # Accept pytest tmp paths
    if any(
        p in ("tmp", "pytest") or p.startswith("pytest") or p.startswith("tmp")
        for p in parts
    ):
        return

    raise ValueError(
        f"Checkpoint path must be a volume-mounted Docker path, not repo-relative.\n"
        f"  Got: {checkpoint_dir}\n"
        f"  Expected: /app/checkpoints/... (Docker volume)\n"
        f"  Escape hatch (pytest ONLY): export MINIVESS_ALLOW_HOST=1\n"
        f"  See: CLAUDE.md Rule #18, docs/planning/minivess-vision-enforcement-plan.md"
    )


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
        sw_batch_size: int = 4,
        fold_label: str = "",
        system_monitor: Any | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self._fold_label = f"{fold_label}: " if fold_label else ""
        self._monitor = system_monitor
        self.device = torch.device(device)
        self.model.to(self.device)
        self.tracker = tracker
        self.metrics = metrics
        self.val_roi_size = val_roi_size
        self.sw_batch_size = sw_batch_size

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

    def _compute_loss(
        self,
        output: Any,
        batch: dict[str, Any],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss via isinstance dispatch.

        If criterion is MultiTaskLoss: call with (output, batch).
        Otherwise (standard criterion): call with (logits, labels).
        """
        from minivess.pipeline.multitask_loss import MultiTaskLoss

        if isinstance(self.criterion, MultiTaskLoss):
            result: torch.Tensor = self.criterion(output, batch)
            return result
        result2: torch.Tensor = self.criterion(output.logits, labels)
        return result2

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
            # Move all tensor values to device (supports multi-task GT keys)
            with record_function("data_to_device"):
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                images = batch["image"]
                labels = batch["label"]

            with record_function("zero_grad"):
                self.optimizer.zero_grad()

            with autocast(
                device_type=self.device.type,
                enabled=self.config.mixed_precision,
            ):
                with record_function("forward"):
                    output = self.model(images)
                with record_function("loss_compute"):
                    loss = self._compute_loss(output, batch, labels)

            if not torch.isfinite(loss):
                msg = (
                    f"Non-finite loss detected: {loss.item()}. "
                    "This prevents gradient poisoning from corrupting optimizer state."
                )
                raise ValueError(msg)

            with record_function("backward_optimizer"):
                self.scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
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
                with record_function("metrics_update"), torch.no_grad():
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
            with record_function("val_data_to_device"):
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                images = batch["image"]
                labels = batch["label"]

            with (
                torch.no_grad(),
                autocast(
                    device_type=self.device.type,
                    enabled=self.config.mixed_precision_val,
                ),
            ):
                with record_function("val_forward"):
                    # Use sliding window inference when val_roi_size is set,
                    # needed because full 512x512xZ volumes exceed GPU memory.
                    if self.val_roi_size is not None:

                        def _model_fn(x: torch.Tensor) -> torch.Tensor:
                            result: torch.Tensor = self.model(x).logits
                            return result

                        logits_raw = sliding_window_inference(
                            images,
                            roi_size=self.val_roi_size,
                            sw_batch_size=self.sw_batch_size,
                            predictor=_model_fn,
                            overlap=0.25,
                        )
                        assert isinstance(logits_raw, torch.Tensor)  # noqa: S101
                        logits = logits_raw
                    else:
                        output = self.model(images)
                        logits = output.logits

                with record_function("val_loss_compute"):
                    if self.val_roi_size is not None:
                        # Sliding window: standard loss only (no multi-task aux)
                        from minivess.pipeline.multitask_loss import MultiTaskLoss

                        if isinstance(self.criterion, MultiTaskLoss):
                            loss = self.criterion.seg_criterion(logits, labels)
                        else:
                            loss = self.criterion(logits, labels)
                    else:
                        loss = self._compute_loss(output, batch, labels)

            running_loss += loss.item()
            num_batches += 1

            if self.metrics is not None:
                self.metrics.update(logits, labels)

            if compute_extended:
                with record_function("val_extended_metrics"):
                    # Move to CPU and convert to binary predictions
                    pred_probs = torch.softmax(logits, dim=1)
                    pred_binary = (
                        pred_probs[:, 1:].argmax(dim=1)
                        if logits.shape[1] > 2
                        else (pred_probs[:, 1] > 0.5).long()
                    )
                    for b in range(images.shape[0]):
                        pred_np = pred_binary[b].cpu().numpy().astype(np.uint8)
                        label_np = labels[b, 0].cpu().numpy().astype(np.uint8)
                        collected_preds.append(pred_np)
                        collected_labels.append(label_np)

        avg_loss = running_loss / max(num_batches, 1)

        # NaN guard: FP16 encoder overflow or degenerate patches can produce
        # NaN validation loss. Log a warning but don't crash — training should
        # continue even if a single validation epoch produces NaN.
        # Training NaN still hard-fails (see train_epoch). Issue #715.
        if not math.isfinite(avg_loss):
            logger.warning(
                "Non-finite val_loss detected (%.4f over %d batches). "
                "Possible FP16 overflow in encoder — see issue #715. "
                "Logging NaN but continuing training.",
                avg_loss,
                num_batches,
            )

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
        except (ImportError, SyntaxError):
            # SyntaxError: MetricsReloaded has unescaped LaTeX `\d` in
            # docstrings that triggers SyntaxError on Python 3.12.12+.
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

        mean_cldice = (
            float(np.nanmean(per_vol_cldice)) if per_vol_cldice else float("nan")
        )
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
        fold_id: int = 0,
        checkpoint_dir: Path | None = None,
        profiling_config: ProfilingConfig | None = None,
        start_epoch: int = 0,
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
        start_epoch:
            Epoch to resume from (default 0 = fresh start). Used for
            spot preemption recovery with epoch_latest checkpoint.

        Returns
        -------
        dict[str, Any]
            Summary with ``best_val_loss`` (backward compat), ``final_epoch``,
            ``history``, and ``best_metrics``.
        """
        # Validate checkpoint path (CLAUDE.md Rule #18, #19)
        if checkpoint_dir is not None:
            validate_checkpoint_path(checkpoint_dir)

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        final_epoch = 0
        ckpt_cfg = self.config.checkpoint
        # Capture MLflow run ID from tracker (if active) for return dict
        _active_run_id: str | None = (
            self.tracker.run_id if self.tracker is not None else None
        )
        epoch_start_time = time.perf_counter()

        # Determine if extended metrics (MetricsReloaded) are needed
        # by checking if any tracked metric requires them
        _extended_metric_names = {"val_cldice", "val_masd", "val_compound_masd_cldice"}
        _tracked_names = {m.name for m in ckpt_cfg.tracked_metrics}
        needs_extended = bool(_tracked_names & _extended_metric_names)
        # MetricsReloaded (skeleton + surface distance) is expensive on full volumes.
        # Compute every N epochs to keep overhead manageable (~2x instead of 14x).
        extended_frequency = 5

        val_interval = self.config.val_interval
        _last_val_result: EpochResult | None = None

        # Build profiler context — nullcontext when disabled (zero overhead).
        # Use ExitStack to avoid re-indenting the entire epoch loop body.
        from contextlib import ExitStack

        from minivess.pipeline.profiler_integration import build_profiler_context

        _prof_config = profiling_config
        if _prof_config is None:
            from minivess.config.models import ProfilingConfig as _PC

            _prof_config = _PC(enabled=False)
        _prof_output_dir = checkpoint_dir if checkpoint_dir is not None else Path(".")
        _prof_ctx = build_profiler_context(_prof_config, output_dir=_prof_output_dir)
        _exit_stack = ExitStack()
        _prof = _exit_stack.enter_context(_prof_ctx)

        if start_epoch > 0:
            logger.info("Resuming training from epoch %d", start_epoch)

        for epoch in range(start_epoch, self.config.max_epochs):
            t0 = time.perf_counter()
            train_result = self.train_epoch(train_loader)

            # Validate every val_interval epochs + first + last epoch.
            # val_interval > max_epochs is a sentinel for "never validate"
            # (e.g. sam3_hybrid debug: val_interval = max_epochs + 1).
            is_last = epoch == self.config.max_epochs - 1
            _skip_all_val = val_interval > self.config.max_epochs
            run_val = not _skip_all_val and (
                epoch % val_interval == 0 or epoch == 0 or is_last
            )

            if run_val:
                # Compute extended metrics every N epochs + first + last
                compute_ext_this_epoch = needs_extended and (
                    epoch % extended_frequency == 0 or is_last
                )
                val_result = self.validate_epoch(
                    val_loader, compute_extended=compute_ext_this_epoch
                )
                _last_val_result = val_result
            else:
                # Reuse last validation result for logging (no new validation)
                val_result = _last_val_result or EpochResult(loss=float("nan"))

            self.scheduler.step()
            epoch_wall_time = time.perf_counter() - t0

            # Log first/steady epoch wall time for profiling (#683)
            if epoch == 0 and self.tracker is not None:
                self.tracker.log_epoch_metrics(
                    {"prof_first_epoch_seconds": epoch_wall_time}, step=0
                )
            elif epoch == 1 and self.tracker is not None:
                self.tracker.log_epoch_metrics(
                    {"prof_steady_epoch_seconds": epoch_wall_time}, step=1
                )

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
                "%sEpoch %d/%d — train_loss: %.4f, val_loss: %.4f, lr: %.2e",
                self._fold_label,
                epoch + 1,
                self.config.max_epochs,
                train_result.loss,
                val_result.loss,
                current_lr,
            )

            # T4/T5: GPU efficiency line + MLflow metrics (every epoch)
            gpu_metrics: dict[str, float] = {}
            if self._monitor is not None:
                snap = self._monitor.get_latest_snapshot()
                gpu_metrics = self._monitor.epoch_summary()
                if snap is not None and snap.gpu_memory_total_mb > 0:
                    status = self._monitor.gpu_status(snap)
                    logger.info(
                        "%s[GPU]  util=%d%%  bw=%d%%  temp=%dC  pwr=%.0fW"
                        " | cpu=%.0f%% | vram=%d/%dMB  clk=%dMHz  %s",
                        self._fold_label,
                        snap.gpu_utilization_percent,
                        snap.gpu_mem_bw_util_pct,
                        snap.gpu_temperature_c,
                        snap.gpu_power_w,
                        snap.cpu_percent,
                        snap.gpu_memory_used_mb,
                        snap.gpu_memory_total_mb,
                        snap.gpu_sm_clock_mhz,
                        status,
                    )

            # T9: [MEM] slow resource line every 10 epochs
            if self._monitor is not None and (epoch + 1) % 10 == 0:
                snap = self._monitor.get_latest_snapshot()
                if snap is not None:
                    logger.info(
                        "%s[MEM]  ram=%.1f/%.1fGB(%.0f%%)  swap=%.1f/%.1fGB"
                        "  vram=%d/%dMB  rss=%.1fGB",
                        self._fold_label,
                        snap.ram_used_gb,
                        snap.ram_total_gb,
                        snap.ram_percent,
                        snap.swap_used_gb,
                        snap.swap_total_gb,
                        snap.gpu_memory_used_mb,
                        snap.gpu_memory_total_mb,
                        snap.process_rss_gb,
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
                # T5: GPU epoch summary → MLflow (prefixed with sys_gpu_)
                for k, v in gpu_metrics.items():
                    epoch_log[f"sys_gpu_{k}"] = v
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
                        try:
                            self.tracker.log_artifact(
                                best_path, artifact_path="checkpoints"
                            )
                        except Exception:
                            logger.warning(
                                "Failed to upload checkpoint artifact to MLflow "
                                "(checkpoint saved locally at %s)",
                                best_path,
                                exc_info=True,
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

            # Write epoch_latest.yaml and epoch_latest.pth for spot-preemption recovery
            if checkpoint_dir is not None:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                _primary = (
                    self._multi_tracker.trackers[0]
                    if self._multi_tracker.trackers
                    else None
                )
                _best_val_loss = (
                    float(_primary.best_value) if _primary is not None else float("inf")
                )
                _run_id: str | None = None
                if self.tracker is not None:
                    _run_id = getattr(self.tracker, "run_id", None)
                _epoch_state: dict[str, object] = {
                    "epoch": int(epoch),
                    "fold": int(fold_id),
                    "mlflow_run_id": _run_id,
                    "best_val_loss": _best_val_loss,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
                # Atomic write order: .pth first, then .yaml
                # If preempted after .pth but before .yaml, resume detects stale yaml
                atomic_torch_save(
                    self.model.state_dict(),
                    checkpoint_dir / "epoch_latest.pth",
                )
                epoch_latest_path = checkpoint_dir / "epoch_latest.yaml"
                atomic_text_write(yaml.dump(_epoch_state), epoch_latest_path)

            # Step profiler (per-EPOCH, not per-batch — RC3)
            if _prof is not None:
                _prof.step()

            # Early stopping decision via MultiMetricTracker
            if self._multi_tracker.should_stop(epoch):
                logger.info(
                    "Early stopping at epoch %d (strategy=%s)",
                    epoch + 1,
                    ckpt_cfg.early_stopping_strategy,
                )
                break

        # Close profiler context (ExitStack cleanup)
        _exit_stack.close()

        # Upload last.pth and metric_history.json to MLflow
        if self.tracker is not None and checkpoint_dir is not None:
            last_path = checkpoint_dir / "last.pth"
            if last_path.exists():
                try:
                    self.tracker.log_artifact(last_path, artifact_path="checkpoints")
                except Exception:
                    logger.warning(
                        "Failed to upload last.pth to MLflow (saved locally at %s)",
                        last_path,
                        exc_info=True,
                    )
            history_path = checkpoint_dir / "metric_history.json"
            if history_path.exists():
                try:
                    self.tracker.log_artifact(history_path, artifact_path="history")
                except Exception:
                    logger.warning(
                        "Failed to upload metric_history.json to MLflow",
                        exc_info=True,
                    )

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
            "mlflow_run_id": _active_run_id,
        }
