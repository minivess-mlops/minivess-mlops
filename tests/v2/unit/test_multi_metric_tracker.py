from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from minivess.pipeline.multi_metric_tracker import (
    MetricCheckpoint,
    MetricDirection,
    MetricHistory,
    MetricTracker,
    MultiMetricTracker,
    load_metric_checkpoint,
    save_metric_checkpoint,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# MetricTracker — direction initialisation
# ---------------------------------------------------------------------------


def test_tracker_best_value_initialization() -> None:
    """MINIMIZE tracker starts at +inf, MAXIMIZE starts at -inf."""
    minimize_tracker = MetricTracker(
        name="val_loss", direction=MetricDirection.MINIMIZE
    )
    maximize_tracker = MetricTracker(name="dsc", direction=MetricDirection.MAXIMIZE)

    assert minimize_tracker.best_value == math.inf
    assert maximize_tracker.best_value == -math.inf


# ---------------------------------------------------------------------------
# MetricTracker — direction correctness
# ---------------------------------------------------------------------------


def test_minimize_direction_correct() -> None:
    """MetricTracker with MINIMIZE direction correctly identifies improvement as lower value."""
    tracker = MetricTracker(
        name="val_loss", direction=MetricDirection.MINIMIZE, min_delta=1e-4
    )
    # Lower value should be an improvement
    assert tracker.has_improved(0.5) is True
    tracker.update(0.5, epoch=0)
    # Same value: not an improvement (no delta)
    assert tracker.has_improved(0.5) is False
    # Slightly higher: not an improvement
    assert tracker.has_improved(0.5001) is False
    # Lower by more than min_delta: improvement
    assert tracker.has_improved(0.4998) is True


def test_maximize_direction_correct() -> None:
    """MetricTracker with MAXIMIZE direction correctly identifies improvement as higher value."""
    tracker = MetricTracker(
        name="dsc", direction=MetricDirection.MAXIMIZE, min_delta=1e-4
    )
    # Higher value should be an improvement
    assert tracker.has_improved(0.8) is True
    tracker.update(0.8, epoch=0)
    # Same value: not an improvement
    assert tracker.has_improved(0.8) is False
    # Slightly lower: not an improvement
    assert tracker.has_improved(0.7999) is False
    # Higher by more than min_delta: improvement
    assert tracker.has_improved(0.8002) is True


# ---------------------------------------------------------------------------
# MetricTracker — update behaviour
# ---------------------------------------------------------------------------


def test_single_metric_improvement_detected() -> None:
    """MetricTracker.update returns True when val_loss improves by > min_delta."""
    tracker = MetricTracker(
        name="val_loss",
        direction=MetricDirection.MINIMIZE,
        patience=5,
        min_delta=1e-4,
    )
    # First update always improves (starts at +inf)
    improved = tracker.update(0.5, epoch=0)
    assert improved is True
    assert tracker.best_value == 0.5
    assert tracker.best_epoch == 0
    assert tracker.patience_counter == 0

    # Second update with worse value
    improved = tracker.update(0.6, epoch=1)
    assert improved is False
    assert tracker.patience_counter == 1

    # Third update with clear improvement
    improved = tracker.update(0.3, epoch=2)
    assert improved is True
    assert tracker.best_value == 0.3
    assert tracker.best_epoch == 2
    assert tracker.patience_counter == 0


def test_min_delta_filters_trivial_improvement() -> None:
    """Improvement smaller than min_delta is not counted as improvement."""
    tracker = MetricTracker(
        name="val_loss",
        direction=MetricDirection.MINIMIZE,
        patience=5,
        min_delta=0.01,
    )
    tracker.update(0.5, epoch=0)

    # Trivial improvement (less than min_delta=0.01)
    improved = tracker.update(0.4995, epoch=1)
    assert improved is False
    assert tracker.patience_counter == 1

    # Significant improvement (> min_delta)
    improved = tracker.update(0.48, epoch=2)
    assert improved is True
    assert tracker.patience_counter == 0


# ---------------------------------------------------------------------------
# MetricTracker — patience / exhaustion
# ---------------------------------------------------------------------------


def test_per_metric_patience_independent() -> None:
    """Each MetricTracker has its own patience counter that increments independently."""
    tracker_loss = MetricTracker(
        name="val_loss", direction=MetricDirection.MINIMIZE, patience=3
    )
    tracker_dsc = MetricTracker(
        name="dsc", direction=MetricDirection.MAXIMIZE, patience=3
    )

    # Initialise both trackers
    tracker_loss.update(0.5, epoch=0)
    tracker_dsc.update(0.7, epoch=0)

    # Only loss degrades; DSC improves every epoch (distinct increasing values)
    dsc_values = [0.80, 0.85, 0.90]
    for ep, dsc_val in zip(range(1, 4), dsc_values, strict=True):
        tracker_loss.update(0.6, epoch=ep)  # no improvement
        tracker_dsc.update(dsc_val, epoch=ep)  # improvement every epoch

    assert tracker_loss.patience_counter == 3
    assert tracker_dsc.patience_counter == 0


# ---------------------------------------------------------------------------
# MultiMetricTracker — update returns improved metric names
# ---------------------------------------------------------------------------


def test_multi_metric_tracker_returns_improved_names() -> None:
    """update() returns exactly the list of metric names that improved this epoch."""
    trackers = [
        MetricTracker(name="val_loss", direction=MetricDirection.MINIMIZE),
        MetricTracker(name="dsc", direction=MetricDirection.MAXIMIZE),
    ]
    mmt = MultiMetricTracker(trackers=trackers, primary_metric="val_loss")

    # First epoch: all improve from sentinel values
    improved = mmt.update({"val_loss": 0.5, "dsc": 0.7}, epoch=0)
    assert set(improved) == {"val_loss", "dsc"}

    # Second epoch: only dsc improves
    improved = mmt.update({"val_loss": 0.6, "dsc": 0.8}, epoch=1)
    assert improved == ["dsc"]

    # Third epoch: only val_loss improves
    improved = mmt.update({"val_loss": 0.3, "dsc": 0.75}, epoch=2)
    assert improved == ["val_loss"]


def test_multi_metric_independent_tracking() -> None:
    """MultiMetricTracker tracks multiple metrics independently — improving val_loss
    but not DSC should only return val_loss in the improved list."""
    trackers = [
        MetricTracker(name="val_loss", direction=MetricDirection.MINIMIZE, patience=5),
        MetricTracker(name="dsc", direction=MetricDirection.MAXIMIZE, patience=5),
    ]
    mmt = MultiMetricTracker(trackers=trackers, primary_metric="val_loss")

    # Initialise
    mmt.update({"val_loss": 0.5, "dsc": 0.8}, epoch=0)

    # val_loss improves, dsc does not
    improved = mmt.update({"val_loss": 0.3, "dsc": 0.75}, epoch=1)
    assert "val_loss" in improved
    assert "dsc" not in improved

    # Verify independent counters
    val_loss_tracker = mmt.get_primary_tracker()
    dsc_tracker = next(t for t in mmt.trackers if t.name == "dsc")
    assert val_loss_tracker.patience_counter == 0
    assert dsc_tracker.patience_counter == 1


# ---------------------------------------------------------------------------
# MultiMetricTracker — early stopping strategies
# ---------------------------------------------------------------------------


def test_early_stop_strategy_all() -> None:
    """With strategy='all', training stops only when ALL trackers exhaust patience."""
    trackers = [
        MetricTracker(name="val_loss", direction=MetricDirection.MINIMIZE, patience=2),
        MetricTracker(name="dsc", direction=MetricDirection.MAXIMIZE, patience=2),
    ]
    mmt = MultiMetricTracker(
        trackers=trackers,
        primary_metric="val_loss",
        early_stopping_strategy="all",
    )

    mmt.update({"val_loss": 0.5, "dsc": 0.7}, epoch=0)
    # val_loss doesn't improve for 2 epochs but dsc does
    mmt.update({"val_loss": 0.6, "dsc": 0.8}, epoch=1)
    mmt.update({"val_loss": 0.7, "dsc": 0.9}, epoch=2)

    # val_loss patience exhausted (counter=2), but dsc improved → should NOT stop
    assert mmt.should_stop(current_epoch=2) is False

    # Now dsc also stops improving
    mmt.update({"val_loss": 0.8, "dsc": 0.85}, epoch=3)
    mmt.update({"val_loss": 0.9, "dsc": 0.80}, epoch=4)

    # Both exhausted
    assert mmt.should_stop(current_epoch=4) is True


def test_early_stop_strategy_any() -> None:
    """With strategy='any', training stops when ANY single tracker exhausts patience."""
    trackers = [
        MetricTracker(name="val_loss", direction=MetricDirection.MINIMIZE, patience=2),
        MetricTracker(name="dsc", direction=MetricDirection.MAXIMIZE, patience=5),
    ]
    mmt = MultiMetricTracker(
        trackers=trackers,
        primary_metric="val_loss",
        early_stopping_strategy="any",
    )

    mmt.update({"val_loss": 0.5, "dsc": 0.7}, epoch=0)
    mmt.update({"val_loss": 0.6, "dsc": 0.8}, epoch=1)
    mmt.update({"val_loss": 0.7, "dsc": 0.9}, epoch=2)

    # val_loss patience exhausted (counter=2) → should stop even though dsc is fine
    assert mmt.should_stop(current_epoch=2) is True


def test_early_stop_strategy_primary() -> None:
    """With strategy='primary', training stops when the primary metric exhausts patience."""
    trackers = [
        MetricTracker(name="val_loss", direction=MetricDirection.MINIMIZE, patience=2),
        MetricTracker(name="dsc", direction=MetricDirection.MAXIMIZE, patience=2),
    ]
    mmt = MultiMetricTracker(
        trackers=trackers,
        primary_metric="val_loss",
        early_stopping_strategy="primary",
    )

    mmt.update({"val_loss": 0.5, "dsc": 0.7}, epoch=0)
    mmt.update({"val_loss": 0.6, "dsc": 0.8}, epoch=1)
    mmt.update({"val_loss": 0.7, "dsc": 0.9}, epoch=2)

    # val_loss patience exhausted → stop
    assert mmt.should_stop(current_epoch=2) is True

    # If we re-init and let only dsc exhaust, primary (val_loss) keeps improving
    trackers2 = [
        MetricTracker(name="val_loss", direction=MetricDirection.MINIMIZE, patience=2),
        MetricTracker(name="dsc", direction=MetricDirection.MAXIMIZE, patience=2),
    ]
    mmt2 = MultiMetricTracker(
        trackers=trackers2,
        primary_metric="val_loss",
        early_stopping_strategy="primary",
    )
    mmt2.update({"val_loss": 0.5, "dsc": 0.7}, epoch=0)
    mmt2.update({"val_loss": 0.4, "dsc": 0.6}, epoch=1)  # val_loss improves
    mmt2.update({"val_loss": 0.3, "dsc": 0.5}, epoch=2)  # val_loss improves

    # dsc exhausted but primary (val_loss) fine → do NOT stop
    assert mmt2.should_stop(current_epoch=2) is False


def test_min_epochs_prevents_early_stop() -> None:
    """should_stop returns False when current_epoch < min_epochs even if patience exhausted."""
    trackers = [
        MetricTracker(name="val_loss", direction=MetricDirection.MINIMIZE, patience=1),
    ]
    mmt = MultiMetricTracker(
        trackers=trackers,
        primary_metric="val_loss",
        early_stopping_strategy="primary",
        min_epochs=10,
    )

    mmt.update({"val_loss": 0.5}, epoch=0)
    mmt.update(
        {"val_loss": 0.6}, epoch=1
    )  # no improvement → patience_counter = 1 = patience

    # Patience would be exhausted but we haven't reached min_epochs yet
    assert mmt.should_stop(current_epoch=1) is False

    # At min_epochs or beyond, stopping is allowed
    assert mmt.should_stop(current_epoch=10) is True


# ---------------------------------------------------------------------------
# MultiMetricTracker — get_primary_tracker
# ---------------------------------------------------------------------------


def test_get_primary_tracker() -> None:
    """get_primary_tracker returns the tracker matching primary_metric name."""
    trackers = [
        MetricTracker(name="val_loss", direction=MetricDirection.MINIMIZE),
        MetricTracker(name="dsc", direction=MetricDirection.MAXIMIZE),
        MetricTracker(name="hd95", direction=MetricDirection.MINIMIZE),
    ]
    mmt = MultiMetricTracker(trackers=trackers, primary_metric="dsc")
    primary = mmt.get_primary_tracker()
    assert primary.name == "dsc"
    assert primary.direction == MetricDirection.MAXIMIZE


# ---------------------------------------------------------------------------
# MetricCheckpoint — save/load round-trip
# ---------------------------------------------------------------------------


def test_checkpoint_save_load_round_trip(tmp_path: Path) -> None:
    """save_metric_checkpoint then load_metric_checkpoint recovers all fields."""
    checkpoint_path = tmp_path / "checkpoint.pt"

    model_state = {"weight": torch.tensor([1.0, 2.0])}
    optimizer_state = {"lr": 1e-3, "step": 5}
    scheduler_state = {"last_epoch": 5}

    all_metrics = {"val_loss": 0.42, "dsc": 0.85, "hd95": 12.3}
    checkpoint = MetricCheckpoint(
        epoch=5,
        metrics=all_metrics,
        metric_name="val_loss",
        metric_value=0.42,
        metric_direction="minimize",
        train_loss=0.55,
        val_loss=0.42,
        wall_time_sec=123.4,
        config_snapshot={"lr": 1e-3, "batch_size": 4},
    )

    save_metric_checkpoint(
        path=checkpoint_path,
        model_state_dict=model_state,
        optimizer_state_dict=optimizer_state,
        scheduler_state_dict=scheduler_state,
        checkpoint=checkpoint,
    )

    assert checkpoint_path.exists()

    loaded_model, loaded_opt, loaded_sched, loaded_ckpt, loaded_scaler = (
        load_metric_checkpoint(checkpoint_path)
    )

    # Model state round-trip
    assert torch.allclose(loaded_model["weight"], model_state["weight"])

    # Optimizer / scheduler state
    assert loaded_opt == optimizer_state
    assert loaded_sched == scheduler_state

    # No scaler saved in this test (not passed)
    assert loaded_scaler is None

    # MetricCheckpoint fields
    assert loaded_ckpt.epoch == 5
    assert loaded_ckpt.metrics == all_metrics
    assert loaded_ckpt.metric_name == "val_loss"
    assert loaded_ckpt.metric_value == 0.42
    assert loaded_ckpt.metric_direction == "minimize"
    assert loaded_ckpt.train_loss == 0.55
    assert loaded_ckpt.val_loss == 0.42
    assert abs(loaded_ckpt.wall_time_sec - 123.4) < 1e-6
    assert loaded_ckpt.config_snapshot == {"lr": 1e-3, "batch_size": 4}


def test_checkpoint_contains_all_metrics(tmp_path: Path) -> None:
    """MetricCheckpoint stores all tracked metrics, not just the triggering one."""
    checkpoint_path = tmp_path / "ckpt.pt"

    all_metrics = {"val_loss": 0.3, "dsc": 0.9, "hd95": 5.0, "nsd": 0.88}
    checkpoint = MetricCheckpoint(
        epoch=10,
        metrics=all_metrics,
        metric_name="dsc",  # triggered by dsc
        metric_value=0.9,
        metric_direction="maximize",
        train_loss=0.4,
        val_loss=0.3,
        wall_time_sec=200.0,
        config_snapshot={},
    )

    save_metric_checkpoint(
        path=checkpoint_path,
        model_state_dict={"w": torch.tensor([0.5])},
        optimizer_state_dict={},
        scheduler_state_dict={},
        checkpoint=checkpoint,
    )

    _, _, _, loaded_ckpt, _ = load_metric_checkpoint(checkpoint_path)

    # All 4 metrics must be present
    assert set(loaded_ckpt.metrics.keys()) == {"val_loss", "dsc", "hd95", "nsd"}
    assert loaded_ckpt.metrics["val_loss"] == 0.3
    assert loaded_ckpt.metrics["dsc"] == 0.9
    assert loaded_ckpt.metric_name == "dsc"


# ---------------------------------------------------------------------------
# MetricHistory — JSON round-trip
# ---------------------------------------------------------------------------


def test_metric_history_json_round_trip(tmp_path: Path) -> None:
    """MetricHistory save_json then load_json recovers all epoch records."""
    history = MetricHistory()
    history.record_epoch(
        epoch=0,
        metrics={"val_loss": 0.5, "dsc": 0.7},
        wall_time_sec=10.0,
        checkpoints_saved=["val_loss"],
    )
    history.record_epoch(
        epoch=1,
        metrics={"val_loss": 0.4, "dsc": 0.8},
        wall_time_sec=11.0,
        checkpoints_saved=["val_loss", "dsc"],
    )

    json_path = tmp_path / "history.json"
    history.save_json(json_path)
    assert json_path.exists()

    loaded = MetricHistory.load_json(json_path)
    assert len(loaded.epochs) == 2

    ep0 = loaded.epochs[0]
    assert ep0["epoch"] == 0
    assert ep0["metrics"] == {"val_loss": 0.5, "dsc": 0.7}
    assert ep0["wall_time_sec"] == 10.0
    assert ep0["checkpoints_saved"] == ["val_loss"]

    ep1 = loaded.epochs[1]
    assert ep1["epoch"] == 1
    assert ep1["checkpoints_saved"] == ["val_loss", "dsc"]
