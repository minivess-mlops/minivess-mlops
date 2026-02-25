from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from pathlib import Path


class MetricDirection(StrEnum):
    """Direction for metric improvement."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass
class MetricTracker:
    """Track a single metric across training epochs.

    Parameters
    ----------
    name:
        Metric identifier (e.g. ``"val_loss"``, ``"dsc"``).
    direction:
        Whether lower or higher values indicate improvement.
    patience:
        Number of epochs without improvement before considering this metric
        exhausted.
    min_delta:
        Minimum absolute change to count as a meaningful improvement.
    """

    name: str
    direction: MetricDirection
    patience: int = 10
    min_delta: float = 1e-4
    best_value: float = field(init=False)
    patience_counter: int = field(default=0, init=False)
    best_epoch: int = field(default=-1, init=False)

    def __post_init__(self) -> None:
        if self.direction == MetricDirection.MINIMIZE:
            self.best_value = math.inf
        else:
            self.best_value = -math.inf

    def has_improved(self, value: float) -> bool:
        """Return True if *value* is better than the current best by at least min_delta."""
        if self.direction == MetricDirection.MINIMIZE:
            return value < self.best_value - self.min_delta
        return value > self.best_value + self.min_delta

    def update(self, value: float, epoch: int) -> bool:
        """Update the tracker with a new observation.

        Parameters
        ----------
        value:
            The metric value for this epoch.
        epoch:
            Current epoch index (0-based).

        Returns
        -------
        bool
            True if the value represents an improvement.
        """
        if self.has_improved(value):
            self.best_value = value
            self.best_epoch = epoch
            self.patience_counter = 0
            return True
        self.patience_counter += 1
        return False

    def exhausted(self) -> bool:
        """Return True when patience has been fully consumed."""
        return self.patience_counter >= self.patience


class MultiMetricTracker:
    """Coordinate multiple :class:`MetricTracker` instances.

    Parameters
    ----------
    trackers:
        List of individual metric trackers.
    primary_metric:
        Name of the metric used by the ``"primary"`` early-stopping strategy.
    early_stopping_strategy:
        One of ``"all"``, ``"any"``, or ``"primary"``.

        * ``"all"``  — stop only when every tracker is exhausted.
        * ``"any"``  — stop as soon as any single tracker is exhausted.
        * ``"primary"`` — stop when the primary metric tracker is exhausted.
    min_epochs:
        Training will not be stopped before this many epochs have elapsed,
        regardless of patience state.
    """

    def __init__(
        self,
        trackers: list[MetricTracker],
        primary_metric: str,
        early_stopping_strategy: str = "all",
        min_epochs: int = 0,
    ) -> None:
        if early_stopping_strategy not in {"all", "any", "primary"}:
            raise ValueError(
                f"early_stopping_strategy must be 'all', 'any', or 'primary', "
                f"got {early_stopping_strategy!r}"
            )
        if not any(t.name == primary_metric for t in trackers):
            raise ValueError(
                f"primary_metric {primary_metric!r} not found in trackers "
                f"({[t.name for t in trackers]})"
            )
        self.trackers = trackers
        self.primary_metric = primary_metric
        self.early_stopping_strategy = early_stopping_strategy
        self.min_epochs = min_epochs

    def update(self, metrics: dict[str, float], epoch: int) -> list[str]:
        """Update all matching trackers for the current epoch.

        Parameters
        ----------
        metrics:
            Mapping from metric name to observed value.
        epoch:
            Current epoch index (0-based).

        Returns
        -------
        list[str]
            Names of the metrics that improved this epoch.
        """
        improved: list[str] = []
        for tracker in self.trackers:
            if tracker.name in metrics and tracker.update(metrics[tracker.name], epoch):
                improved.append(tracker.name)
        return improved

    def should_stop(self, current_epoch: int) -> bool:
        """Evaluate whether early stopping criteria are met.

        Parameters
        ----------
        current_epoch:
            The epoch index just completed (0-based).

        Returns
        -------
        bool
            True if training should be stopped.
        """
        if current_epoch < self.min_epochs:
            return False

        if self.early_stopping_strategy == "all":
            return all(t.exhausted() for t in self.trackers)
        if self.early_stopping_strategy == "any":
            return any(t.exhausted() for t in self.trackers)
        # "primary"
        return self.get_primary_tracker().exhausted()

    def get_primary_tracker(self) -> MetricTracker:
        """Return the tracker associated with the primary metric."""
        for tracker in self.trackers:
            if tracker.name == self.primary_metric:
                return tracker
        # Should be unreachable — validated in __init__
        raise RuntimeError(
            f"Primary tracker {self.primary_metric!r} not found"
        )  # pragma: no cover


@dataclass
class MetricCheckpoint:
    """Self-contained snapshot of training state at checkpoint time.

    Parameters
    ----------
    epoch:
        Epoch at which the checkpoint was taken (0-based).
    metrics:
        All tracked metric values recorded at this epoch.
    metric_name:
        The metric that triggered this checkpoint save.
    metric_value:
        Value of the triggering metric.
    metric_direction:
        Direction of the triggering metric (``"minimize"`` or ``"maximize"``).
    train_loss:
        Training loss at this epoch.
    val_loss:
        Validation loss at this epoch.
    wall_time_sec:
        Wall-clock time elapsed up to this epoch (seconds).
    config_snapshot:
        Frozen copy of the experiment configuration relevant at save time.
    """

    epoch: int
    metrics: dict[str, float]
    metric_name: str
    metric_value: float
    metric_direction: str
    train_loss: float
    val_loss: float
    wall_time_sec: float
    config_snapshot: dict[str, Any]


def save_metric_checkpoint(
    path: Path,
    model_state_dict: dict[str, Any],
    optimizer_state_dict: dict[str, Any],
    scheduler_state_dict: dict[str, Any],
    checkpoint: MetricCheckpoint,
) -> None:
    """Persist a self-contained checkpoint to disk.

    The checkpoint is structured so that ``torch.load`` with
    ``weights_only=True`` still works — the :class:`MetricCheckpoint` is
    serialised as a plain Python dict under the ``"checkpoint_metadata"``
    key, separately from the tensor-containing state dicts.

    Parameters
    ----------
    path:
        Destination file path (including filename / extension).
    model_state_dict:
        State dict returned by ``model.state_dict()``.
    optimizer_state_dict:
        State dict returned by ``optimizer.state_dict()``.
    scheduler_state_dict:
        State dict returned by ``scheduler.state_dict()``.
    checkpoint:
        Metadata snapshot to embed in the checkpoint.
    """
    payload: dict[str, Any] = {
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "scheduler_state_dict": scheduler_state_dict,
        "checkpoint_metadata": {
            "epoch": checkpoint.epoch,
            "metrics": checkpoint.metrics,
            "metric_name": checkpoint.metric_name,
            "metric_value": checkpoint.metric_value,
            "metric_direction": checkpoint.metric_direction,
            "train_loss": checkpoint.train_loss,
            "val_loss": checkpoint.val_loss,
            "wall_time_sec": checkpoint.wall_time_sec,
            "config_snapshot": checkpoint.config_snapshot,
        },
    }
    torch.save(payload, path)


def load_metric_checkpoint(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], MetricCheckpoint]:
    """Load a checkpoint saved by :func:`save_metric_checkpoint`.

    Parameters
    ----------
    path:
        File path to load from.

    Returns
    -------
    tuple
        ``(model_state_dict, optimizer_state_dict, scheduler_state_dict,
        MetricCheckpoint)``
    """
    payload: dict[str, Any] = torch.load(path, weights_only=True)
    meta: dict[str, Any] = payload["checkpoint_metadata"]
    ckpt = MetricCheckpoint(
        epoch=meta["epoch"],
        metrics=meta["metrics"],
        metric_name=meta["metric_name"],
        metric_value=meta["metric_value"],
        metric_direction=meta["metric_direction"],
        train_loss=meta["train_loss"],
        val_loss=meta["val_loss"],
        wall_time_sec=meta["wall_time_sec"],
        config_snapshot=meta["config_snapshot"],
    )
    return (
        payload["model_state_dict"],
        payload["optimizer_state_dict"],
        payload["scheduler_state_dict"],
        ckpt,
    )


class MetricHistory:
    """Accumulate per-epoch training history and serialise to JSON.

    Each recorded epoch is stored as a plain dict so that the JSON
    representation is human-readable and easily consumed by downstream
    analytics (e.g. DuckDB, pandas).
    """

    def __init__(self) -> None:
        self.epochs: list[dict[str, Any]] = []

    def record_epoch(
        self,
        epoch: int,
        metrics: dict[str, float],
        wall_time_sec: float,
        checkpoints_saved: list[str],
    ) -> None:
        """Append one epoch record to the history.

        Parameters
        ----------
        epoch:
            Epoch index (0-based).
        metrics:
            All metric values observed this epoch.
        wall_time_sec:
            Wall-clock time for this epoch (seconds).
        checkpoints_saved:
            Names of metrics whose checkpoints were saved this epoch.
        """
        self.epochs.append(
            {
                "epoch": epoch,
                "metrics": metrics,
                "wall_time_sec": wall_time_sec,
                "checkpoints_saved": checkpoints_saved,
            }
        )

    def save_json(self, path: Path) -> None:
        """Write history to a JSON file.

        Parameters
        ----------
        path:
            Destination file path.
        """
        path.write_text(json.dumps({"epochs": self.epochs}, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path) -> MetricHistory:
        """Reconstruct a :class:`MetricHistory` from a JSON file.

        Parameters
        ----------
        path:
            Source file path.

        Returns
        -------
        MetricHistory
            Populated history instance.
        """
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        instance = cls()
        instance.epochs = data["epochs"]
        return instance
