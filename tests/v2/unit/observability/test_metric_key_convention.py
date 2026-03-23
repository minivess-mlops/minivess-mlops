"""Tests for metric key naming convention (slash-prefix).

Verifies that metric_keys.py defines all keys with slash separators.
Greenfield: MIGRATION_MAP and normalize functions have been deleted (#790).

Issue: #790
"""

from __future__ import annotations


def test_all_metric_keys_use_slash_separator() -> None:
    """Every key constant in MetricKeys must use slash separator."""
    from minivess.observability.metric_keys import MetricKeys

    # PREFIX constants (e.g., EVAL_PREFIX="eval") are short prefixes that
    # don't themselves contain "/" — they're used to BUILD slash-separated keys.
    _PREFIX_ATTRS = {"GPU_PREFIX", "EVAL_PREFIX", "EVAL_TEST_PREFIX"}

    for attr_name in dir(MetricKeys):
        if attr_name.startswith("_"):
            continue
        value = getattr(MetricKeys, attr_name)
        if not isinstance(value, str):
            continue
        if attr_name in _PREFIX_ATTRS:
            continue  # Prefixes are building blocks, not full keys
        assert "/" in value, (
            f"MetricKeys.{attr_name} = {value!r} does not use slash separator"
        )


def test_no_migration_map_exists() -> None:
    """Greenfield: MIGRATION_MAP must not exist in metric_keys module."""
    import minivess.observability.metric_keys as mk

    assert not hasattr(mk, "MIGRATION_MAP"), (
        "MIGRATION_MAP must be deleted (greenfield -- no legacy runs)"
    )


def test_no_normalize_functions_exist() -> None:
    """Greenfield: normalize_metric_key/dict must not exist."""
    import minivess.observability.metric_keys as mk

    assert not hasattr(mk, "normalize_metric_key")
    assert not hasattr(mk, "normalize_metric_dict")


def test_metric_keys_has_training_keys() -> None:
    from minivess.observability.metric_keys import MetricKeys

    assert MetricKeys.TRAIN_LOSS == "train/loss"
    assert MetricKeys.VAL_LOSS == "val/loss"
    assert MetricKeys.TRAIN_DICE == "train/dice"
    assert MetricKeys.VAL_DICE == "val/dice"
    assert MetricKeys.OPTIM_LR == "optim/lr"


def test_metric_keys_has_profiling_keys() -> None:
    from minivess.observability.metric_keys import MetricKeys

    assert MetricKeys.PROF_FIRST_EPOCH_SECONDS == "prof/first_epoch_seconds"
    assert MetricKeys.PROF_STEADY_EPOCH_SECONDS == "prof/steady_epoch_seconds"
    assert MetricKeys.PROF_OVERHEAD_PCT == "prof/overhead_pct"


def test_metric_keys_has_cost_keys() -> None:
    from minivess.observability.metric_keys import MetricKeys

    assert MetricKeys.COST_TOTAL_USD == "cost/total_usd"
    assert MetricKeys.COST_TOTAL_WALL_SECONDS == "cost/total_wall_seconds"


def test_metric_keys_has_fold_keys() -> None:
    from minivess.observability.metric_keys import MetricKeys

    assert MetricKeys.FOLD_N_COMPLETED == "fold/n_completed"
    assert MetricKeys.VRAM_PEAK_MB == "vram/peak_mb"


def test_metric_keys_has_gradient_keys() -> None:
    from minivess.observability.metric_keys import MetricKeys

    assert MetricKeys.GRAD_NORM_MEAN == "grad/norm_mean"
    assert MetricKeys.GRAD_CLIP_COUNT == "grad/clip_count"


def test_metric_keys_has_inference_keys() -> None:
    from minivess.observability.metric_keys import MetricKeys

    assert MetricKeys.INFER_LATENCY_MS_PER_VOLUME == "infer/latency_ms_per_volume"


def test_metric_keys_has_checkpoint_keys() -> None:
    from minivess.observability.metric_keys import MetricKeys

    assert MetricKeys.CHECKPOINT_SIZE_MB == "checkpoint/size_mb"
