"""Prefect Data Flow (Flow 1) — Data Engineering.

Discovers, validates, profiles, and splits datasets for the training pipeline.
Uses ``_prefect_compat`` decorators for graceful degradation without Prefect.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from minivess.orchestration import flow, task

if TYPE_CHECKING:
    from pathlib import Path

    from minivess.data.splits import FoldSplit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataValidationReport:
    """Result of data pair validation.

    Attributes
    ----------
    n_pairs:
        Total number of discovered pairs.
    n_valid:
        Number of pairs that pass validation.
    errors:
        List of critical error messages.
    warnings:
        List of non-critical warning messages.
    """

    n_pairs: int
    n_valid: int
    errors: list[str]
    warnings: list[str]


@dataclass
class DataFlowResult:
    """Result of the complete data engineering flow.

    Attributes
    ----------
    pairs:
        Discovered image/label pairs.
    validation_report:
        Validation report (None if skipped).
    quality_passed:
        Whether the quality gate passed.
    n_folds:
        Number of cross-validation folds generated.
    splits:
        K-fold splits (None if quality gate failed).
    external_datasets:
        Dict of external dataset name → discovered pairs.
    provenance:
        Data provenance metadata logged to MLflow.
    """

    pairs: list[dict[str, str]]
    validation_report: DataValidationReport | None
    quality_passed: bool
    n_folds: int
    splits: list[FoldSplit] | None
    external_datasets: dict[str, list[dict[str, str]]]
    provenance: dict[str, Any]


# ---------------------------------------------------------------------------
# @task functions
# ---------------------------------------------------------------------------


@task(name="discover-data")
def discover_data_task(data_dir: Path) -> list[dict[str, str]]:
    """Discover image/label pairs from a data directory.

    Parameters
    ----------
    data_dir:
        Root data directory with images/labels subdirectories.

    Returns
    -------
    List of ``{"image": ..., "label": ...}`` dicts.
    """
    from minivess.data.external_datasets import discover_external_test_pairs

    pairs: list[dict[str, str]] = discover_external_test_pairs(
        data_dir=data_dir, dataset_name="primary"
    )
    logger.info("Discovered %d image/label pairs in %s", len(pairs), data_dir)
    return pairs


@task(name="validate-data")
def validate_data_task(
    pairs: list[dict[str, str]],
) -> DataValidationReport:
    """Validate discovered data pairs.

    Parameters
    ----------
    pairs:
        Image/label pairs to validate.

    Returns
    -------
    DataValidationReport with counts and error/warning lists.
    """
    errors: list[str] = []
    warnings: list[str] = []
    n_valid = 0

    for pair in pairs:
        if not pair.get("image") or not pair.get("label"):
            errors.append(f"Missing image or label in pair: {pair}")
            continue
        n_valid += 1

    if not pairs:
        errors.append("No data pairs found")

    report = DataValidationReport(
        n_pairs=len(pairs),
        n_valid=n_valid,
        errors=errors,
        warnings=warnings,
    )
    logger.info(
        "Validation: %d/%d valid, %d errors, %d warnings",
        report.n_valid,
        report.n_pairs,
        len(report.errors),
        len(report.warnings),
    )
    return report


@task(name="data-quality-gate")
def data_quality_gate(report: DataValidationReport) -> bool:
    """Check whether data passes quality gate.

    Parameters
    ----------
    report:
        Validation report to check.

    Returns
    -------
    True if data is acceptable for training.
    """
    if report.n_pairs == 0:
        logger.warning("Quality gate FAILED: no data pairs found")
        return False
    if report.errors:
        logger.warning("Quality gate FAILED: %d errors", len(report.errors))
        return False
    logger.info("Quality gate PASSED: %d valid pairs", report.n_valid)
    return True


@task(name="validate-external-data")
def validate_external_data_task(
    dataset_name: str,
    data_dir: Path,
) -> list[dict[str, str]]:
    """Validate and discover pairs for an external dataset.

    Parameters
    ----------
    dataset_name:
        Name of the external dataset.
    data_dir:
        Root directory for the external dataset.

    Returns
    -------
    List of discovered pairs.
    """
    from minivess.data.external_datasets import discover_external_test_pairs

    pairs: list[dict[str, str]] = discover_external_test_pairs(
        data_dir=data_dir, dataset_name=dataset_name
    )
    logger.info(
        "External dataset '%s': %d pairs in %s",
        dataset_name,
        len(pairs),
        data_dir,
    )
    return pairs


@task(name="split-data")
def split_data_task(
    pairs: list[dict[str, str]],
    n_folds: int,
    seed: int,
) -> list[FoldSplit]:
    """Generate K-fold splits from data pairs.

    Parameters
    ----------
    pairs:
        Image/label pairs to split.
    n_folds:
        Number of cross-validation folds.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    List of FoldSplit objects.
    """
    from minivess.data.splits import generate_kfold_splits

    result: list[FoldSplit] = generate_kfold_splits(
        data_dicts=pairs, num_folds=n_folds, seed=seed
    )
    logger.info("Generated %d-fold splits from %d pairs", n_folds, len(pairs))
    return result


@task(name="log-data-provenance")
def log_data_provenance_task(
    n_volumes: int,
    n_folds: int,
    dataset_hash: str,
) -> dict[str, Any]:
    """Compute data provenance metadata for MLflow logging.

    Parameters
    ----------
    n_volumes:
        Number of volumes in the dataset.
    n_folds:
        Number of cross-validation folds.
    dataset_hash:
        Hash of the dataset for reproducibility.

    Returns
    -------
    Dict of MLflow-compatible params with ``data_`` prefix.
    """
    provenance: dict[str, Any] = {
        "data_n_volumes": n_volumes,
        "data_n_folds": n_folds,
        "data_hash": dataset_hash,
    }
    logger.info("Data provenance: %s", provenance)
    return provenance


# ---------------------------------------------------------------------------
# @flow orchestrator
# ---------------------------------------------------------------------------


def _compute_dataset_hash(pairs: list[dict[str, str]]) -> str:
    """Compute a deterministic hash of dataset pairs for provenance."""
    content = "|".join(
        f"{p.get('image', '')}:{p.get('label', '')}"
        for p in sorted(pairs, key=lambda p: p.get("image", ""))
    )
    return hashlib.sha256(content.encode()).hexdigest()[:16]


@flow(name="minivess-data")
def run_data_flow(
    data_dir: Path,
    n_folds: int = 3,
    seed: int = 42,
    external_dirs: dict[str, Path] | None = None,
) -> DataFlowResult:
    """Data Engineering Flow (Flow 1).

    Discovers, validates, profiles, and splits datasets.

    Parameters
    ----------
    data_dir:
        Root directory for primary dataset.
    n_folds:
        Number of cross-validation folds.
    seed:
        Random seed for reproducibility.
    external_dirs:
        Optional dict of dataset_name → data_dir for external datasets.

    Returns
    -------
    DataFlowResult with all flow outputs.
    """
    logger.info("Starting data flow → %s", data_dir)

    # Step 1: Discover
    pairs = discover_data_task(data_dir=data_dir)

    # Step 2: Validate
    report = validate_data_task(pairs=pairs)

    # Step 3: Quality gate
    quality_passed = data_quality_gate(report=report)

    # Step 4: Split (only if quality gate passed)
    splits: list[FoldSplit] | None = None
    if quality_passed and len(pairs) >= n_folds:
        splits = split_data_task(pairs=pairs, n_folds=n_folds, seed=seed)

    # Step 5: External datasets
    external_datasets: dict[str, list[dict[str, str]]] = {}
    if external_dirs:
        for name, ext_dir in external_dirs.items():
            external_datasets[name] = validate_external_data_task(
                dataset_name=name, data_dir=ext_dir
            )

    # Step 6: Provenance
    dataset_hash = _compute_dataset_hash(pairs)
    provenance = log_data_provenance_task(
        n_volumes=len(pairs),
        n_folds=n_folds,
        dataset_hash=dataset_hash,
    )

    logger.info("Data flow complete: %d pairs, quality=%s", len(pairs), quality_passed)
    return DataFlowResult(
        pairs=pairs,
        validation_report=report,
        quality_passed=quality_passed,
        n_folds=n_folds,
        splits=splits,
        external_datasets=external_datasets,
        provenance=provenance,
    )
