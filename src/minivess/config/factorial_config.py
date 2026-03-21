"""Composable factorial experiment configuration.

Parses factorial YAML files from ``configs/factorial/*.yaml`` and auto-derives
ALL factor names, levels, and the Cartesian product. Zero hardcoded factor names.

The factorial YAML is sectioned by layer:
  - ``factors.training``: GPU factors (model_family, loss_name, aux_calibration)
  - ``factors.post_training``: CPU factors (method, recalibration)
  - ``factors.analysis``: CPU/GPU factors (ensemble_strategy)

Users create their own factorial YAMLs — ``paper_full.yaml`` and ``debug.yaml``
are just example named configs. A different lab can create ``my_lab.yaml`` with
any subset of factors.

Source of truth: ``docs/planning/pre-gcp-master-plan.xml`` line 16
Synthesis: ``docs/planning/intermedia-plan-synthesis-pre-debug-run.md`` Part 1.3-1.4
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from pathlib import (
    Path,  # noqa: TC003 — Pydantic/dataclass needs runtime access
)
from typing import Any

logger = logging.getLogger(__name__)

# Canonical mapping: YAML factor name → MLflow tag / DuckDB column name.
# YAML uses short names (e.g., "method"); MLflow/DuckDB use qualified names
# (e.g., "post_training_method") to avoid ambiguity across layers.
FACTOR_NAME_MAPPING: dict[str, str] = {
    "method": "post_training_method",
    "aux_calibration": "with_aux_calib",
}


@dataclass(frozen=True)
class FactorialFactor:
    """One factor in the factorial design."""

    name: str
    layer: str  # "training", "post_training", "analysis"
    levels: list[str]

    @property
    def n_levels(self) -> int:
        return len(self.levels)


@dataclass
class FactorialDesign:
    """Complete factorial design parsed from YAML.

    Auto-derived from whichever YAML is used — zero hardcoded factor names.
    """

    experiment_name: str
    description: str
    factors: list[FactorialFactor]
    zero_shot_baselines: list[dict[str, Any]]
    fixed: dict[str, Any]
    mlflow: dict[str, str]
    biostatistics: dict[str, Any]
    debug: bool = False

    @property
    def n_conditions(self) -> int:
        """Total Cartesian product across all layers."""
        if not self.factors:
            return 0
        product = 1
        for f in self.factors:
            product *= f.n_levels
        return product

    @property
    def n_training_conditions(self) -> int:
        """Cartesian product of training factors ONLY (Layer A — costs money)."""
        training = [f for f in self.factors if f.layer == "training"]
        if not training:
            return 0
        product = 1
        for f in training:
            product *= f.n_levels
        return product

    @property
    def n_folds(self) -> int:
        return int(self.fixed.get("num_folds", 3))

    @property
    def n_training_runs(self) -> int:
        """Training runs = training conditions × folds (GPU cost)."""
        return self.n_training_conditions * self.n_folds

    def factor_names(self, layer: str | None = None) -> list[str]:
        """Get factor names, optionally filtered by layer."""
        if layer is None:
            return [f.name for f in self.factors]
        return [f.name for f in self.factors if f.layer == layer]

    def factor_levels(self) -> dict[str, list[str]]:
        """Get all factor names → levels mapping."""
        return {f.name: f.levels for f in self.factors}

    def training_conditions(self) -> list[dict[str, str]]:
        """Generate all training-layer Cartesian product conditions."""
        training = [f for f in self.factors if f.layer == "training"]
        if not training:
            return []
        names = [f.name for f in training]
        level_lists = [f.levels for f in training]
        return [
            dict(zip(names, combo, strict=True))
            for combo in itertools.product(*level_lists)
        ]

    def all_conditions(self) -> list[dict[str, str]]:
        """Generate ALL Cartesian product conditions across ALL layers."""
        if not self.factors:
            return []
        names = [f.name for f in self.factors]
        level_lists = [f.levels for f in self.factors]
        return [
            dict(zip(names, combo, strict=True))
            for combo in itertools.product(*level_lists)
        ]


def parse_factorial_yaml(yaml_path: Path) -> FactorialDesign:
    """Parse a factorial YAML file into a FactorialDesign.

    Accepts any YAML with a ``factors`` dict sectioned by layer.
    Zero hardcoded factor names — auto-derives from YAML keys.

    Parameters
    ----------
    yaml_path:
        Path to the factorial YAML (e.g., ``configs/factorial/debug.yaml``).

    Returns
    -------
    FactorialDesign with all factors, levels, and metadata.

    Raises
    ------
    FileNotFoundError
        If the YAML file doesn't exist.
    ValueError
        If the YAML is missing required fields.
    """
    import yaml

    if not yaml_path.exists():
        msg = f"Factorial YAML not found: {yaml_path}"
        raise FileNotFoundError(msg)

    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if raw is None:
        msg = f"Empty factorial YAML: {yaml_path}"
        raise ValueError(msg)

    factors_raw = raw.get("factors", {})
    if not factors_raw:
        msg = f"No 'factors' section in {yaml_path}"
        raise ValueError(msg)

    # Parse factors from each layer section
    factors: list[FactorialFactor] = []
    for layer_name, layer_factors in factors_raw.items():
        if not isinstance(layer_factors, dict):
            continue
        for factor_name, levels in layer_factors.items():
            if not isinstance(levels, list):
                continue
            factors.append(
                FactorialFactor(
                    name=factor_name,
                    layer=layer_name,
                    levels=[str(level) for level in levels],
                )
            )

    # Parse other sections
    experiment_name = raw.get("experiment_name", yaml_path.stem)
    description = raw.get("description", "")
    zero_shot = raw.get("zero_shot_baselines", [])
    fixed = raw.get("fixed", {})
    mlflow = raw.get("mlflow", {})
    biostatistics = raw.get("biostatistics", {})
    debug = raw.get("debug", False)

    design = FactorialDesign(
        experiment_name=experiment_name,
        description=description,
        factors=factors,
        zero_shot_baselines=zero_shot if zero_shot else [],
        fixed=fixed,
        mlflow=mlflow,
        biostatistics=biostatistics,
        debug=debug,
    )

    logger.info(
        "Parsed factorial design '%s': %d factors, %d total conditions "
        "(%d training × %d folds = %d GPU runs)",
        experiment_name,
        len(factors),
        design.n_conditions,
        design.n_training_conditions,
        design.n_folds,
        design.n_training_runs,
    )

    return design
