"""Unit tests for the YAML-driven metric registry.

Tests cover:
- MetricDefinition dataclass (immutability, field types)
- MetricRegistry (lookup, helpers, dunder methods)
- load_metric_registry() (YAML loading, validation, error paths)

All tests are self-contained and do NOT require a running MLflow server.
The real configs/metric_registry.yaml is loaded in the integration-style
``TestLoadFromDefaultYaml`` class; all other tests use a synthetic YAML
written to tmp_path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from minivess.pipeline.metric_registry import (
    MetricDefinition,
    MetricRegistry,
    load_metric_registry,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_YAML = _REPO_ROOT / "configs" / "metric_registry.yaml"

_MINIMAL_ENTRY = {
    "name": "dsc",
    "display_name": "Dice Score",
    "mlflow_name": "eval_fold{fold_id}_dsc",
    "direction": "maximize",
    "unit": "",
    "bounds": [0.0, 1.0],
    "description": "Voxel overlap",
}

_MASD_ENTRY = {
    "name": "measured_masd",
    "display_name": "MASD",
    "mlflow_name": "eval_fold{fold_id}_measured_masd",
    "direction": "minimize",
    "unit": "voxels",
    "bounds": [0.0, 50.0],
    "description": "Surface distance",
}

_VAL_LOSS_ENTRY = {
    "name": "val_loss",
    "display_name": "Validation Loss",
    "mlflow_name": "val_loss",
    "direction": "minimize",
    "unit": "",
    "bounds": [0.0, 10.0],
    "description": "Val loss",
}


def _write_yaml(tmp_path: Path, entries: list[dict[str, Any]]) -> Path:
    """Write a metric registry YAML to tmp_path and return its path."""
    yaml_path = tmp_path / "metric_registry.yaml"
    yaml_path.write_text(yaml.dump({"metrics": entries}), encoding="utf-8")
    return yaml_path


@pytest.fixture()  # type: ignore[misc]
def single_metric_yaml(tmp_path: Path) -> Path:
    """YAML file with a single maximize metric."""
    return _write_yaml(tmp_path, [_MINIMAL_ENTRY])


@pytest.fixture()  # type: ignore[misc]
def multi_metric_yaml(tmp_path: Path) -> Path:
    """YAML file with three metrics (maximize, minimize, minimize)."""
    return _write_yaml(tmp_path, [_MINIMAL_ENTRY, _MASD_ENTRY, _VAL_LOSS_ENTRY])


@pytest.fixture()  # type: ignore[misc]
def single_registry(single_metric_yaml: Path) -> MetricRegistry:
    return load_metric_registry(single_metric_yaml)


@pytest.fixture()  # type: ignore[misc]
def multi_registry(multi_metric_yaml: Path) -> MetricRegistry:
    return load_metric_registry(multi_metric_yaml)


# ---------------------------------------------------------------------------
# Tests: MetricDefinition
# ---------------------------------------------------------------------------


class TestMetricDefinition:
    """Tests for the MetricDefinition frozen dataclass."""

    def test_dataclass_fields_present(self) -> None:
        defn = MetricDefinition(
            name="dsc",
            display_name="Dice Score",
            mlflow_name="eval_fold{fold_id}_dsc",
            direction="maximize",
            unit="",
            bounds=(0.0, 1.0),
            description="Overlap",
        )
        assert defn.name == "dsc"
        assert defn.display_name == "Dice Score"
        assert defn.mlflow_name == "eval_fold{fold_id}_dsc"
        assert defn.direction == "maximize"
        assert defn.unit == ""
        assert defn.bounds == (0.0, 1.0)
        assert defn.description == "Overlap"

    def test_frozen_raises_on_mutation(self) -> None:
        """Frozen dataclass must reject attribute assignment."""
        defn = MetricDefinition(
            name="dsc",
            display_name="Dice Score",
            mlflow_name="eval_fold{fold_id}_dsc",
            direction="maximize",
            unit="",
            bounds=(0.0, 1.0),
            description="Overlap",
        )
        with pytest.raises((AttributeError, TypeError)):
            defn.name = "new_name"  # type: ignore[misc]

    def test_bounds_is_tuple(self) -> None:
        defn = MetricDefinition(
            name="masd",
            display_name="MASD",
            mlflow_name="masd",
            direction="minimize",
            unit="voxels",
            bounds=(0.0, 50.0),
            description="",
        )
        assert isinstance(defn.bounds, tuple)
        assert len(defn.bounds) == 2

    def test_two_instances_with_same_fields_are_equal(self) -> None:
        a = MetricDefinition(
            name="dsc",
            display_name="Dice Score",
            mlflow_name="eval_fold{fold_id}_dsc",
            direction="maximize",
            unit="",
            bounds=(0.0, 1.0),
            description="Overlap",
        )
        b = MetricDefinition(
            name="dsc",
            display_name="Dice Score",
            mlflow_name="eval_fold{fold_id}_dsc",
            direction="maximize",
            unit="",
            bounds=(0.0, 1.0),
            description="Overlap",
        )
        assert a == b

    def test_hashable(self) -> None:
        """Frozen dataclasses must be hashable."""
        defn = MetricDefinition(
            name="dsc",
            display_name="Dice Score",
            mlflow_name="eval_fold{fold_id}_dsc",
            direction="maximize",
            unit="",
            bounds=(0.0, 1.0),
            description="Overlap",
        )
        # Should not raise
        _ = {defn}


# ---------------------------------------------------------------------------
# Tests: MetricRegistry
# ---------------------------------------------------------------------------


class TestMetricRegistry:
    """Tests for MetricRegistry behaviour."""

    def test_load_from_yaml(self, single_metric_yaml: Path) -> None:
        """load_metric_registry returns a MetricRegistry instance."""
        registry = load_metric_registry(single_metric_yaml)
        assert isinstance(registry, MetricRegistry)

    def test_get_known_metric(self, single_registry: MetricRegistry) -> None:
        defn = single_registry.get("dsc")
        assert defn.name == "dsc"

    def test_get_unknown_raises_keyerror(self, single_registry: MetricRegistry) -> None:
        with pytest.raises(KeyError, match="Unknown metric"):
            single_registry.get("nonexistent_metric_xyz")

    def test_keyerror_message_contains_available_names(
        self, multi_registry: MetricRegistry
    ) -> None:
        with pytest.raises(KeyError) as exc_info:
            multi_registry.get("nonexistent")
        assert "dsc" in str(exc_info.value)

    def test_display_name(self, single_registry: MetricRegistry) -> None:
        assert single_registry.display_name("dsc") == "Dice Score"

    def test_direction_maximize(self, single_registry: MetricRegistry) -> None:
        assert single_registry.direction("dsc") == "maximize"

    def test_direction_minimize(self, multi_registry: MetricRegistry) -> None:
        assert multi_registry.direction("measured_masd") == "minimize"

    def test_is_higher_better_true_for_maximize(
        self, single_registry: MetricRegistry
    ) -> None:
        assert single_registry.is_higher_better("dsc") is True

    def test_is_higher_better_false_for_minimize(
        self, multi_registry: MetricRegistry
    ) -> None:
        assert multi_registry.is_higher_better("measured_masd") is False

    def test_all_names_returns_sorted_list(
        self, multi_registry: MetricRegistry
    ) -> None:
        names = multi_registry.all_names()
        assert names == sorted(names)

    def test_all_names_contains_all_metrics(
        self, multi_registry: MetricRegistry
    ) -> None:
        names = multi_registry.all_names()
        assert "dsc" in names
        assert "measured_masd" in names
        assert "val_loss" in names

    def test_contains_known_metric(self, single_registry: MetricRegistry) -> None:
        assert "dsc" in single_registry

    def test_contains_unknown_metric(self, single_registry: MetricRegistry) -> None:
        assert "bogus_metric" not in single_registry

    def test_len(self, multi_registry: MetricRegistry) -> None:
        assert len(multi_registry) == 3

    def test_len_single(self, single_registry: MetricRegistry) -> None:
        assert len(single_registry) == 1

    def test_bounds_tuple(self, multi_registry: MetricRegistry) -> None:
        defn = multi_registry.get("measured_masd")
        assert defn.bounds == (0.0, 50.0)

    def test_mlflow_name_preserved(self, single_registry: MetricRegistry) -> None:
        defn = single_registry.get("dsc")
        assert defn.mlflow_name == "eval_fold{fold_id}_dsc"

    def test_unit_preserved(self, multi_registry: MetricRegistry) -> None:
        defn = multi_registry.get("measured_masd")
        assert defn.unit == "voxels"


# ---------------------------------------------------------------------------
# Tests: load_metric_registry — error paths
# ---------------------------------------------------------------------------


class TestLoadMetricRegistryErrors:
    """Error-path tests for load_metric_registry."""

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_metric_registry(tmp_path / "does_not_exist.yaml")

    def test_missing_metrics_key_raises_value_error(self, tmp_path: Path) -> None:
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text(yaml.dump({"other_key": []}), encoding="utf-8")
        with pytest.raises(ValueError, match="top-level 'metrics' key"):
            load_metric_registry(bad_yaml)

    def test_missing_name_field_raises_value_error(self, tmp_path: Path) -> None:
        bad_entry = {"display_name": "No name", "direction": "maximize"}
        bad_yaml = _write_yaml(tmp_path, [bad_entry])
        with pytest.raises(ValueError, match="name"):
            load_metric_registry(bad_yaml)

    def test_missing_display_name_raises_value_error(self, tmp_path: Path) -> None:
        bad_entry = {"name": "no_display", "direction": "maximize"}
        bad_yaml = _write_yaml(tmp_path, [bad_entry])
        with pytest.raises(ValueError, match="display_name"):
            load_metric_registry(bad_yaml)

    def test_invalid_direction_raises_value_error(self, tmp_path: Path) -> None:
        bad_entry = {
            "name": "bad_dir",
            "display_name": "Bad Direction",
            "direction": "sideways",
        }
        bad_yaml = _write_yaml(tmp_path, [bad_entry])
        with pytest.raises(ValueError, match="direction"):
            load_metric_registry(bad_yaml)

    def test_empty_metrics_list_gives_empty_registry(self, tmp_path: Path) -> None:
        empty_yaml = _write_yaml(tmp_path, [])
        registry = load_metric_registry(empty_yaml)
        assert len(registry) == 0

    def test_none_yaml_content_raises(self, tmp_path: Path) -> None:
        """A YAML that deserialises to None (empty file) raises ValueError."""
        null_yaml = tmp_path / "null.yaml"
        null_yaml.write_text("", encoding="utf-8")
        with pytest.raises(ValueError):
            load_metric_registry(null_yaml)


# ---------------------------------------------------------------------------
# Tests: defaults and optional fields
# ---------------------------------------------------------------------------


class TestLoadMetricRegistryDefaults:
    """Verify default values when optional YAML fields are omitted."""

    def test_direction_defaults_to_maximize(self, tmp_path: Path) -> None:
        entry = {"name": "foo", "display_name": "Foo Metric"}
        yaml_path = _write_yaml(tmp_path, [entry])
        registry = load_metric_registry(yaml_path)
        assert registry.direction("foo") == "maximize"

    def test_mlflow_name_defaults_to_name(self, tmp_path: Path) -> None:
        entry = {"name": "bar", "display_name": "Bar"}
        yaml_path = _write_yaml(tmp_path, [entry])
        registry = load_metric_registry(yaml_path)
        defn = registry.get("bar")
        assert defn.mlflow_name == "bar"

    def test_unit_defaults_to_empty_string(self, tmp_path: Path) -> None:
        entry = {"name": "baz", "display_name": "Baz"}
        yaml_path = _write_yaml(tmp_path, [entry])
        registry = load_metric_registry(yaml_path)
        assert registry.get("baz").unit == ""

    def test_description_defaults_to_empty_string(self, tmp_path: Path) -> None:
        entry = {"name": "qux", "display_name": "Qux"}
        yaml_path = _write_yaml(tmp_path, [entry])
        registry = load_metric_registry(yaml_path)
        assert registry.get("qux").description == ""

    def test_bounds_default_to_zero_one(self, tmp_path: Path) -> None:
        entry = {"name": "norm", "display_name": "Normalised"}
        yaml_path = _write_yaml(tmp_path, [entry])
        registry = load_metric_registry(yaml_path)
        assert registry.get("norm").bounds == (0.0, 1.0)


# ---------------------------------------------------------------------------
# Integration: load from real configs/metric_registry.yaml
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _DEFAULT_YAML.exists(),
    reason=f"Default metric_registry.yaml not found at {_DEFAULT_YAML}",
)
class TestLoadFromDefaultYaml:
    """Integration tests that load the real configs/metric_registry.yaml."""

    def test_loads_without_error(self) -> None:
        registry = load_metric_registry()
        assert isinstance(registry, MetricRegistry)

    def test_has_dsc(self) -> None:
        registry = load_metric_registry()
        assert "dsc" in registry

    def test_has_centreline_dsc(self) -> None:
        registry = load_metric_registry()
        assert "centreline_dsc" in registry

    def test_has_measured_masd(self) -> None:
        registry = load_metric_registry()
        assert "measured_masd" in registry

    def test_has_val_compound_masd_cldice(self) -> None:
        registry = load_metric_registry()
        assert "val_compound_masd_cldice" in registry

    def test_has_val_loss(self) -> None:
        registry = load_metric_registry()
        assert "val_loss" in registry

    def test_dsc_is_maximize(self) -> None:
        registry = load_metric_registry()
        assert registry.direction("dsc") == "maximize"

    def test_measured_masd_is_minimize(self) -> None:
        registry = load_metric_registry()
        assert registry.direction("measured_masd") == "minimize"

    def test_val_loss_is_minimize(self) -> None:
        registry = load_metric_registry()
        assert registry.direction("val_loss") == "minimize"

    def test_measured_masd_unit_is_voxels(self) -> None:
        registry = load_metric_registry()
        assert registry.get("measured_masd").unit == "voxels"

    def test_all_names_sorted(self) -> None:
        registry = load_metric_registry()
        names = registry.all_names()
        assert names == sorted(names)

    def test_minimum_metric_count(self) -> None:
        """Real YAML should have at least 6 entries."""
        registry = load_metric_registry()
        assert len(registry) >= 6

    def test_all_directions_valid(self) -> None:
        registry = load_metric_registry()
        valid = {"maximize", "minimize"}
        for name in registry.all_names():
            d = registry.direction(name)
            assert d in valid, f"Metric {name!r} has invalid direction {d!r}"
