"""YAML-driven metric registry for consistent metric naming and display.

Eliminates hardcoded metric strings throughout the codebase by providing
a single source of truth for metric definitions loaded from
``configs/metric_registry.yaml``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Default YAML path relative to this file:
#   src/minivess/pipeline/metric_registry.py  (3 parents → repo root)
#   repo_root/configs/metric_registry.yaml
_DEFAULT_YAML: Path = (
    Path(__file__).resolve().parents[3] / "configs" / "metric_registry.yaml"
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricDefinition:
    """Immutable definition of a single tracked metric.

    Attributes
    ----------
    name:
        Internal snake_case identifier (e.g. ``"dsc"``).
    display_name:
        Human-readable label for plots and reports
        (e.g. ``"Dice Score (DSC)"``).
    mlflow_name:
        MLflow metric key template.  May contain ``{fold_id}`` for
        per-fold eval metrics (e.g. ``"eval_fold{fold_id}_dsc"``).
    direction:
        ``"maximize"`` or ``"minimize"`` — which direction is better.
    unit:
        Unit string for axis labels (e.g. ``"mm"``, ``"%"``, ``""``).
    bounds:
        ``(lower, upper)`` valid range for sanity checks.
    description:
        Brief plain-text description of what the metric measures.
    """

    name: str
    display_name: str
    mlflow_name: str
    direction: str
    unit: str
    bounds: tuple[float, float]
    description: str


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class MetricRegistry:
    """Registry of :class:`MetricDefinition` instances loaded from YAML.

    Provides O(1) lookup by internal name and helpers used by report
    generators and plot utilities.

    Parameters
    ----------
    definitions:
        Ordered list of :class:`MetricDefinition` objects.
    """

    def __init__(self, definitions: list[MetricDefinition]) -> None:
        self._by_name: dict[str, MetricDefinition] = {d.name: d for d in definitions}

    # ------------------------------------------------------------------
    # Core lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> MetricDefinition:
        """Return the :class:`MetricDefinition` for *name*.

        Parameters
        ----------
        name:
            Internal metric name (e.g. ``"dsc"``).

        Raises
        ------
        KeyError
            When *name* is not registered.
        """
        if name not in self._by_name:
            available = sorted(self._by_name.keys())
            msg = f"Unknown metric: {name!r}. Available: {available}"
            raise KeyError(msg)
        return self._by_name[name]

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def all_names(self) -> list[str]:
        """Return sorted list of all registered metric names."""
        return sorted(self._by_name.keys())

    def display_name(self, name: str) -> str:
        """Return the human-readable display name for *name*."""
        return self.get(name).display_name

    def direction(self, name: str) -> str:
        """Return ``"maximize"`` or ``"minimize"`` for *name*."""
        return self.get(name).direction

    def is_higher_better(self, name: str) -> bool:
        """Return ``True`` when higher values of *name* are better."""
        return self.get(name).direction == "maximize"

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._by_name)

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def __repr__(self) -> str:  # pragma: no cover
        return f"MetricRegistry({len(self)} metrics)"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_metric_registry(yaml_path: Path | None = None) -> MetricRegistry:
    """Load a :class:`MetricRegistry` from a YAML file.

    Parameters
    ----------
    yaml_path:
        Path to the registry YAML.  When ``None``, the default
        ``configs/metric_registry.yaml`` (relative to the repository root)
        is used.

    Returns
    -------
    MetricRegistry
        Populated registry ready for use.

    Raises
    ------
    FileNotFoundError
        When *yaml_path* does not exist.
    ValueError
        When the YAML is missing the top-level ``metrics`` key or a metric
        entry is missing the required ``name`` or ``display_name`` fields.
    """
    resolved = yaml_path if yaml_path is not None else _DEFAULT_YAML

    if not resolved.exists():
        msg = f"Metric registry YAML not found: {resolved}"
        raise FileNotFoundError(msg)

    with resolved.open(encoding="utf-8") as fh:
        data: Any = yaml.safe_load(fh)

    if not isinstance(data, dict) or "metrics" not in data:
        msg = (
            f"Metric registry YAML at {resolved} must contain a top-level 'metrics' key"
        )
        raise ValueError(msg)

    definitions: list[MetricDefinition] = []
    for entry in data["metrics"]:
        _validate_entry(entry, resolved)
        raw_bounds = entry.get("bounds", [0.0, 1.0])
        bounds: tuple[float, float] = (float(raw_bounds[0]), float(raw_bounds[1]))
        definitions.append(
            MetricDefinition(
                name=str(entry["name"]),
                display_name=str(entry["display_name"]),
                mlflow_name=str(entry.get("mlflow_name", entry["name"])),
                direction=str(entry.get("direction", "maximize")),
                unit=str(entry.get("unit", "")),
                bounds=bounds,
                description=str(entry.get("description", "")),
            )
        )

    logger.debug("Loaded %d metric definitions from %s", len(definitions), resolved)
    return MetricRegistry(definitions)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_entry(entry: Any, source: Path) -> None:
    """Raise :exc:`ValueError` when a required field is absent.

    Parameters
    ----------
    entry:
        Raw dict from the YAML ``metrics`` list.
    source:
        YAML file path — used in error messages only.
    """
    for required_field in ("name", "display_name"):
        if required_field not in entry:
            msg = (
                f"Metric entry in {source} is missing required field "
                f"{required_field!r}: {entry}"
            )
            raise ValueError(msg)

    direction = entry.get("direction", "maximize")
    if direction not in ("maximize", "minimize"):
        msg = (
            f"Metric {entry.get('name', '?')!r} in {source} has invalid "
            f"direction {direction!r}. Must be 'maximize' or 'minimize'."
        )
        raise ValueError(msg)
