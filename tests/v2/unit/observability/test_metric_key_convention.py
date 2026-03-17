"""Tests for metric key naming convention (slash-prefix).

Verifies that metric_keys.py defines all keys with slash separators
and that the MIGRATION_MAP covers all legacy underscore-prefix keys.

Issue: #790
"""

from __future__ import annotations


def test_all_metric_keys_use_slash_separator() -> None:
    """Every key constant in MetricKeys must use slash separator."""
    from minivess.observability.metric_keys import MetricKeys

    for attr_name in dir(MetricKeys):
        if attr_name.startswith("_"):
            continue
        value = getattr(MetricKeys, attr_name)
        if not isinstance(value, str):
            continue
        assert "/" in value, (
            f"MetricKeys.{attr_name} = {value!r} does not use slash separator"
        )


def test_migration_map_values_use_slash() -> None:
    """Every value in MIGRATION_MAP must use slash separator."""
    from minivess.observability.metric_keys import MIGRATION_MAP

    for old_key, new_key in MIGRATION_MAP.items():
        assert "/" in new_key, (
            f"MIGRATION_MAP[{old_key!r}] = {new_key!r} does not use slash"
        )


def test_migration_map_keys_do_not_use_slash() -> None:
    """Every key in MIGRATION_MAP should be the old underscore convention."""
    from minivess.observability.metric_keys import MIGRATION_MAP

    for old_key in MIGRATION_MAP:
        assert "/" not in old_key, (
            f"MIGRATION_MAP key {old_key!r} should be old underscore convention"
        )


def test_migration_map_covers_core_training_keys() -> None:
    """MIGRATION_MAP must cover the core training/validation metric keys."""
    from minivess.observability.metric_keys import MIGRATION_MAP

    required_old_keys = [
        "train_loss",
        "val_loss",
        "learning_rate",
        "train_dice",
        "val_dice",
        "train_f1_foreground",
        "val_f1_foreground",
        "val_cldice",
        "val_masd",
        "val_compound_masd_cldice",
    ]
    for key in required_old_keys:
        assert key in MIGRATION_MAP, f"MIGRATION_MAP missing required old key: {key!r}"


def test_migration_map_covers_system_keys() -> None:
    """MIGRATION_MAP must cover sys_ prefix keys."""
    from minivess.observability.metric_keys import MIGRATION_MAP

    required_old_keys = [
        "sys_python_version",
        "sys_gpu_model",
        "sys_torch_version",
    ]
    for key in required_old_keys:
        assert key in MIGRATION_MAP, f"MIGRATION_MAP missing required old key: {key!r}"


def test_migration_map_covers_cost_keys() -> None:
    """MIGRATION_MAP must cover cost_ prefix keys."""
    from minivess.observability.metric_keys import MIGRATION_MAP

    required_old_keys = [
        "cost_total_usd",
        "cost_total_wall_seconds",
        "cost_setup_fraction",
    ]
    for key in required_old_keys:
        assert key in MIGRATION_MAP, f"MIGRATION_MAP missing required old key: {key!r}"


def test_migration_map_covers_fold_keys() -> None:
    """MIGRATION_MAP must cover fold-level metric keys (template form)."""
    from minivess.observability.metric_keys import MIGRATION_MAP

    # Fold keys use {id} placeholder in map
    required_old_keys = [
        "vram_peak_mb",
        "n_folds_completed",
    ]
    for key in required_old_keys:
        assert key in MIGRATION_MAP, f"MIGRATION_MAP missing required old key: {key!r}"


def test_normalize_metric_key_maps_old_to_new() -> None:
    """normalize_metric_key() should map old keys to new slash-prefix keys."""
    from minivess.observability.metric_keys import normalize_metric_key

    assert normalize_metric_key("train_loss") == "train/loss"
    assert normalize_metric_key("val_dice") == "val/dice"
    assert normalize_metric_key("learning_rate") == "optim/lr"


def test_normalize_metric_key_passes_through_new_keys() -> None:
    """normalize_metric_key() should pass through already-migrated keys."""
    from minivess.observability.metric_keys import normalize_metric_key

    assert normalize_metric_key("train/loss") == "train/loss"
    assert normalize_metric_key("val/dice") == "val/dice"


def test_normalize_metric_key_passes_through_unknown_keys() -> None:
    """normalize_metric_key() should pass through unknown keys unchanged."""
    from minivess.observability.metric_keys import normalize_metric_key

    assert normalize_metric_key("some_custom_metric") == "some_custom_metric"


def test_metric_keys_has_training_keys() -> None:
    """MetricKeys must define train/loss, val/loss, etc."""
    from minivess.observability.metric_keys import MetricKeys

    assert MetricKeys.TRAIN_LOSS == "train/loss"
    assert MetricKeys.VAL_LOSS == "val/loss"
    assert MetricKeys.TRAIN_DICE == "train/dice"
    assert MetricKeys.VAL_DICE == "val/dice"
    assert MetricKeys.OPTIM_LR == "optim/lr"


def test_metric_keys_has_profiling_keys() -> None:
    """MetricKeys must define prof/ prefix keys."""
    from minivess.observability.metric_keys import MetricKeys

    assert MetricKeys.PROF_FIRST_EPOCH_SECONDS == "prof/first_epoch_seconds"
    assert MetricKeys.PROF_STEADY_EPOCH_SECONDS == "prof/steady_epoch_seconds"
    assert MetricKeys.PROF_OVERHEAD_PCT == "prof/overhead_pct"


def test_metric_keys_has_cost_keys() -> None:
    """MetricKeys must define cost/ prefix keys."""
    from minivess.observability.metric_keys import MetricKeys

    assert MetricKeys.COST_TOTAL_USD == "cost/total_usd"
    assert MetricKeys.COST_TOTAL_WALL_SECONDS == "cost/total_wall_seconds"


def test_metric_keys_has_fold_keys() -> None:
    """MetricKeys must define fold/ prefix keys."""
    from minivess.observability.metric_keys import MetricKeys

    assert MetricKeys.FOLD_N_COMPLETED == "fold/n_completed"
    assert MetricKeys.VRAM_PEAK_MB == "vram/peak_mb"


def test_metric_keys_has_gradient_keys() -> None:
    """MetricKeys must define grad/ prefix keys (T3)."""
    from minivess.observability.metric_keys import MetricKeys

    assert MetricKeys.GRAD_NORM_MEAN == "grad/norm_mean"
    assert MetricKeys.GRAD_CLIP_COUNT == "grad/clip_count"


def test_metric_keys_has_inference_keys() -> None:
    """MetricKeys must define infer/ prefix keys (T4)."""
    from minivess.observability.metric_keys import MetricKeys

    assert MetricKeys.INFER_LATENCY_MS_PER_VOLUME == "infer/latency_ms_per_volume"


def test_metric_keys_has_checkpoint_keys() -> None:
    """MetricKeys must define checkpoint/ prefix keys (T7)."""
    from minivess.observability.metric_keys import MetricKeys

    assert MetricKeys.CHECKPOINT_SIZE_MB == "checkpoint/size_mb"
