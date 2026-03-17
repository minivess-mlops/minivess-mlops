"""Tests for backward compatibility of metric key normalization (T8).

Verifies that normalize_metric_key() and normalize_metric_dict() handle
legacy underscore-convention keys when reading old MLflow runs.

Issue: #790
"""

from __future__ import annotations


def test_normalize_metric_dict_converts_old_keys() -> None:
    """normalize_metric_dict() should convert all old keys in a dict."""
    from minivess.observability.metric_keys import normalize_metric_dict

    old_dict = {
        "train_loss": 0.5,
        "val_dice": 0.8,
        "learning_rate": 0.001,
        "some_unknown_key": 42,
    }
    result = normalize_metric_dict(old_dict)
    assert "train/loss" in result
    assert "val/dice" in result
    assert "optim/lr" in result
    assert "some_unknown_key" in result
    assert result["train/loss"] == 0.5
    assert result["val/dice"] == 0.8


def test_normalize_metric_dict_preserves_new_keys() -> None:
    """normalize_metric_dict() should not modify already-migrated keys."""
    from minivess.observability.metric_keys import normalize_metric_dict

    new_dict = {
        "train/loss": 0.5,
        "val/dice": 0.8,
    }
    result = normalize_metric_dict(new_dict)
    assert result == new_dict


def test_normalize_metric_dict_empty() -> None:
    """normalize_metric_dict() should handle empty dict."""
    from minivess.observability.metric_keys import normalize_metric_dict

    assert normalize_metric_dict({}) == {}


def test_normalize_metric_dict_mixed_old_and_new() -> None:
    """normalize_metric_dict() should handle a mix of old and new keys."""
    from minivess.observability.metric_keys import normalize_metric_dict

    mixed = {
        "train_loss": 0.5,
        "val/dice": 0.8,
        "cost_total_usd": 1.23,
    }
    result = normalize_metric_dict(mixed)
    assert "train/loss" in result
    assert "val/dice" in result
    assert "cost/total_usd" in result
