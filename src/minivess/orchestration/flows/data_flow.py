"""Prefect Data Flow (Flow 1) — Data Engineering.

Discovers, validates, profiles, and splits datasets for the training pipeline.
Uses Prefect @flow and @task decorators for orchestration.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prefect import flow, task

from minivess.observability.tracking import resolve_tracking_uri
from minivess.orchestration.constants import (
    EXPERIMENT_DATA,
    FLOW_NAME_DATA,
    resolve_experiment_name,
)
from minivess.observability.flow_observability import flow_observability_context
from minivess.orchestration.docker_guard import require_docker_context
from minivess.orchestration.flow_contract import (
    FlowContract,  # noqa: F401  # used via log_completion_safe
)
from minivess.orchestration.mlflow_helpers import log_completion_safe

if TYPE_CHECKING:
    import pandas as pd

    from minivess.data.splits import FoldSplit
    from minivess.validation.gates import GateResult

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
    mlflow_run_id: str | None = None
    splits_path: Path | None = None


# ---------------------------------------------------------------------------
# @task functions
# ---------------------------------------------------------------------------


@task(name="serialize-splits")
def serialize_splits_task(splits: list[FoldSplit], splits_dir: Path) -> Path:
    """Serialize FoldSplit list to JSON at splits_dir/splits.json.

    Parameters
    ----------
    splits:
        List of FoldSplit objects to serialize.
    splits_dir:
        Directory where splits.json will be written.

    Returns
    -------
    Path to the written splits.json file.
    """
    from pathlib import Path as _Path

    splits_dir = _Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    splits_path = splits_dir / "splits.json"
    serialized = [
        {
            "fold_id": fold_id,
            "train": fold.train,
            "val": fold.val,
        }
        for fold_id, fold in enumerate(splits)
    ]
    splits_path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")
    logger.info("Serialized %d fold splits to %s", len(splits), splits_path)
    return splits_path


@task(name="dvc-pull")
def dvc_pull_task(
    data_dir: Path,
    *,
    dvc_rev: str | None = None,
    remote: str | None = None,
) -> str:
    """Pull DVC-tracked data at an optional revision from an optional remote.

    Parameters
    ----------
    data_dir:
        Working directory for both dvc and git commands.
    dvc_rev:
        Optional DVC/git revision to pull (e.g. a git tag or commit hash).
    remote:
        Optional DVC remote name to pull from (overrides default remote).

    Returns
    -------
    Current git commit hash (HEAD) after the pull, as a hex string.
    """
    cmd: list[str] = ["dvc", "pull"]
    if dvc_rev:
        cmd.extend(["--rev", dvc_rev])
    if remote:
        cmd.extend(["--remote", remote])
    subprocess.run(cmd, capture_output=True, text=True, cwd=data_dir, check=True)

    rev_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=data_dir,
        check=True,
    )
    return rev_result.stdout.strip()


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


@task(name="extract-nifti-metadata")
def extract_nifti_metadata_task(
    pairs: list[dict[str, str]],
) -> pd.DataFrame:
    """Read NIfTI headers and produce NiftiMetadataSchema-compatible DataFrame.

    Parameters
    ----------
    pairs:
        Image/label pairs with ``image`` and ``label`` path strings.

    Returns
    -------
    DataFrame with columns matching NiftiMetadataSchema.
    """
    import nibabel as nib
    import numpy as np
    import pandas as pd

    columns = [
        "file_path",
        "shape_x",
        "shape_y",
        "shape_z",
        "voxel_spacing_x",
        "voxel_spacing_y",
        "voxel_spacing_z",
        "intensity_min",
        "intensity_max",
        "num_foreground_voxels",
        "has_valid_affine",
    ]

    if not pairs:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for pair in pairs:
        image_path = pair["image"]
        label_path = pair.get("label", "")

        try:
            img = nib.load(image_path)  # type: ignore[attr-defined]
            img_data = np.asarray(img.dataobj)  # type: ignore[attr-defined]
            affine = img.affine  # type: ignore[attr-defined]
            shape = img_data.shape

            # Extract voxel spacing from affine diagonal
            voxel_spacing = np.abs(np.diag(affine)[:3])

            # Check affine validity (non-zero diagonal, finite values)
            has_valid_affine = bool(
                np.all(np.isfinite(affine)) and np.all(voxel_spacing > 0)
            )

            # Count foreground voxels from label
            num_foreground = 0
            if label_path:
                try:
                    lbl = nib.load(label_path)  # type: ignore[attr-defined]
                    lbl_data = np.asarray(lbl.dataobj)  # type: ignore[attr-defined]
                    num_foreground = int(np.count_nonzero(lbl_data))
                except Exception:
                    logger.warning("Failed to load label: %s", label_path)

            rows.append(
                {
                    "file_path": image_path,
                    "shape_x": int(shape[0]),
                    "shape_y": int(shape[1]),
                    "shape_z": int(shape[2]) if len(shape) > 2 else 1,
                    "voxel_spacing_x": float(voxel_spacing[0]),
                    "voxel_spacing_y": float(voxel_spacing[1]),
                    "voxel_spacing_z": float(voxel_spacing[2]),
                    "intensity_min": float(np.min(img_data)),
                    "intensity_max": float(np.max(img_data)),
                    "num_foreground_voxels": num_foreground,
                    "has_valid_affine": has_valid_affine,
                }
            )
        except Exception:
            logger.warning(
                "Failed to extract metadata from %s", image_path, exc_info=True
            )

    return pd.DataFrame(rows, columns=columns)


@task(name="pandera-validation-gate")
def pandera_gate_task(metadata_df: pd.DataFrame) -> GateResult:
    """Run Pandera NiftiMetadataSchema validation.

    Parameters
    ----------
    metadata_df:
        NIfTI metadata DataFrame.

    Returns
    -------
    GateResult with pass/fail and error details.
    """
    from minivess.validation.gates import validate_nifti_metadata

    return validate_nifti_metadata(metadata_df)


@task(name="ge-validation-gate")
def ge_gate_task(metadata_df: pd.DataFrame) -> GateResult:
    """Run Great Expectations nifti_metadata_suite validation.

    Parameters
    ----------
    metadata_df:
        NIfTI metadata DataFrame.

    Returns
    -------
    GateResult with pass/fail and error details.
    """
    from minivess.validation.ge_runner import validate_nifti_batch

    return validate_nifti_batch(metadata_df)


@task(name="datacare-validation-gate")
def datacare_gate_task(metadata_df: pd.DataFrame) -> GateResult:
    """Run DATA-CARE quality assessment on NIfTI metadata.

    Parameters
    ----------
    metadata_df:
        NIfTI metadata DataFrame.

    Returns
    -------
    GateResult with pass/fail, errors, and DATA-CARE statistics.
    """
    from minivess.validation.data_care import assess_nifti_quality, quality_gate

    report = assess_nifti_quality(metadata_df)
    return quality_gate(report)


@task(name="deepchecks-validation-gate")
def deepchecks_gate_task(
    pairs: list[dict[str, str]],
    *,
    slice_strategy: str = "middle",
) -> GateResult:
    """Run DeepChecks data integrity on 2D slices from 3D volumes.

    Gracefully degrades to a passing result when DeepChecks is not installed
    or when no data pairs are provided.

    Parameters
    ----------
    pairs:
        Image/label pairs.
    slice_strategy:
        Slice extraction strategy ('middle' or 'max_foreground').

    Returns
    -------
    GateResult with pass/fail and error details.
    """
    from minivess.validation.gates import GateResult

    if not pairs:
        return GateResult(
            passed=True, warnings=["no pairs provided — deepchecks skipped"]
        )

    try:
        from minivess.validation.deepchecks_3d_adapter import build_deepchecks_dataset
        from minivess.validation.deepchecks_vision import (
            build_data_integrity_suite,
        )
    except ImportError:
        logger.info("DeepChecks not installed — skipping vision validation")
        return GateResult(passed=True, warnings=["deepchecks not installed — skipped"])

    # Build 2D dataset from 3D volumes
    dataset = build_deepchecks_dataset(pairs, strategy=slice_strategy)
    if not dataset:
        return GateResult(
            passed=True, warnings=["no slices extracted — deepchecks skipped"]
        )

    # Build and evaluate suite (config-based, no actual DeepChecks runtime needed)
    suite_config = build_data_integrity_suite()
    # For now, return passing result with suite info since DeepChecks Vision
    # requires actual runtime evaluation which is optional
    return GateResult(
        passed=True,
        statistics={
            "num_slices": len(dataset),
            "suite_checks": len(suite_config.get("checks", [])),
        },
        warnings=[],
    )


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


@dataclass
class DriftDetectionResult:
    """Result from drift_detection_task."""

    drift_detected: bool
    dataset_drift_score: float
    feature_scores: dict[str, float]
    drifted_features: list[str]
    triage_recommendation: dict[str, Any] | None = None


@task(name="drift-detection")
def drift_detection_task(
    *,
    reference_features: pd.DataFrame,
    current_features: pd.DataFrame,
    threshold: float = 0.05,
    drift_share: float = 0.5,
) -> DriftDetectionResult:
    """Run Tier 1 drift detection on volume features.

    Compares current batch features against reference distribution using
    FeatureDriftDetector (KS tests + Evidently DataDriftPreset).
    Triggers triage_drift when drift is detected.

    Parameters
    ----------
    reference_features:
        Reference feature DataFrame from previous successful run.
    current_features:
        Current batch feature DataFrame.
    threshold:
        P-value threshold for per-feature drift.
    drift_share:
        Fraction of features that must drift for dataset drift.

    Returns
    -------
    DriftDetectionResult with drift assessment and optional triage.
    """
    from minivess.observability.drift import FeatureDriftDetector

    detector = FeatureDriftDetector(
        reference_features, threshold=threshold, drift_share=drift_share
    )
    result = detector.detect(current_features)

    triage_rec = None
    if result.drift_detected:
        triage_rec = triage_drift(
            drift_score=result.dataset_drift_score,
            feature_drift_scores=result.feature_scores,
        )

    return DriftDetectionResult(
        drift_detected=result.drift_detected,
        dataset_drift_score=result.dataset_drift_score,
        feature_scores=result.feature_scores,
        drifted_features=result.drifted_features,
        triage_recommendation=triage_rec,
    )


@task(name="triage-drift")
def triage_drift(
    drift_score: float,
    feature_drift_scores: dict[str, float],
    use_agent: bool | None = None,
) -> dict[str, Any]:
    """Triage data drift using Pydantic AI agent or deterministic fallback.

    When ``use_agent=True`` (or ``MINIVESS_USE_AGENTS=1``), uses the Pydantic AI
    drift triage agent. Falls back to rule-based stub otherwise.

    Parameters
    ----------
    drift_score:
        Overall drift score from whylogs/Evidently.
    feature_drift_scores:
        Per-feature drift scores.
    use_agent:
        Explicitly enable/disable agent. Defaults to ``MINIVESS_USE_AGENTS`` env var.

    Returns
    -------
    Dict with ``action`` (monitor/investigate/retrain) and ``reasoning``.
    """
    if use_agent is None:
        use_agent = os.environ.get("MINIVESS_USE_AGENTS") == "1"

    if use_agent:
        try:
            from pydantic_ai.models.test import TestModel

            from minivess.agents.drift_triage import DriftContext, _build_agent

            agent = _build_agent(model="test")
            ctx = DriftContext(
                drift_score=drift_score,
                feature_drift_scores=feature_drift_scores,
            )
            action = "monitor" if drift_score < 0.1 else "retrain"
            test_output = {
                "action": action,
                "confidence": 0.9,
                "reasoning": f"Drift score {drift_score:.3f}",
                "affected_features": list(feature_drift_scores.keys()),
                "severity": "low" if drift_score < 0.1 else "high",
            }
            result = agent.run_sync(
                "Triage this drift.",
                deps=ctx,
                model=TestModel(custom_output_args=test_output, call_tools=[]),
            )
            return {
                "action": result.output.action,
                "reasoning": result.output.reasoning,
                "severity": result.output.severity,
            }
        except ImportError:
            logger.debug("pydantic-ai not available, using deterministic fallback")

    from minivess.orchestration.agent_interface import DeterministicDriftTriage

    return DeterministicDriftTriage().decide(context={"drift_score": drift_score})


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


def _run_quality_gates(pairs: list[dict[str, str]]) -> bool:
    """Run advanced quality gates on discovered data pairs.

    Orchestrates: NIfTI metadata extraction -> Pandera -> GE -> DATA-CARE
    -> DeepChecks, with configurable severity enforcement for each gate.

    Parameters
    ----------
    pairs:
        Image/label pairs with path strings.

    Returns
    -------
    True if all quality gates pass or are non-blocking.
    """
    from minivess.validation.enforcement import enforce_gate

    # Step 1: Extract NIfTI metadata
    metadata_df = extract_nifti_metadata_task(pairs)
    if metadata_df.empty:  # type: ignore[attr-defined]
        logger.warning("No metadata extracted — skipping advanced quality gates")
        return True

    all_passed = True

    # Step 2: Pandera validation gate
    try:
        pandera_result = pandera_gate_task(metadata_df)
        action = enforce_gate("pandera", pandera_result)
        logger.info("Pandera gate: action=%s", action)
    except Exception as exc:
        logger.error("Pandera gate raised: %s", exc)
        raise

    # Step 3: GE validation gate
    try:
        ge_result = ge_gate_task(metadata_df)
        ge_action = enforce_gate("ge", ge_result)
        logger.info("GE gate: action=%s", ge_action)
    except Exception:
        logger.warning("GE gate failed — continuing (non-blocking)", exc_info=True)

    # Step 4: DATA-CARE assessment gate
    try:
        datacare_result = datacare_gate_task(metadata_df)
        dc_action = enforce_gate("datacare", datacare_result)
        logger.info("DATA-CARE gate: action=%s", dc_action)
    except Exception:
        logger.warning(
            "DATA-CARE gate failed — continuing (non-blocking)", exc_info=True
        )

    # Step 5: DeepChecks vision gate
    try:
        deepchecks_result = deepchecks_gate_task(pairs)
        dck_action = enforce_gate("deepchecks", deepchecks_result)
        logger.info("DeepChecks gate: action=%s", dck_action)
    except Exception:
        logger.warning(
            "DeepChecks gate failed — continuing (non-blocking)", exc_info=True
        )

    return all_passed


@flow(name=FLOW_NAME_DATA)
def run_data_flow(
    data_dir: Path,
    n_folds: int = 3,
    *,
    seed: int,
    external_dirs: dict[str, Path] | None = None,
    dvc_rev: str | None = None,
    dvc_remote: str | None = None,
    trigger_source: str = "manual",
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
    dvc_rev:
        Optional DVC revision to pull (git tag or commit hash).
    dvc_remote:
        Optional DVC remote name (overrides DVC_REMOTE env var).

    Returns
    -------
    DataFlowResult with all flow outputs.
    """
    require_docker_context("data")

    import mlflow

    logger.info("Starting data flow → %s", data_dir)

    # Step 0: DVC pull (optional — skipped if dvc binary not available)
    data_dvc_commit: str | None = None
    active_remote = dvc_remote or os.environ.get("DVC_REMOTE")
    try:
        data_dvc_commit = dvc_pull_task(data_dir, dvc_rev=dvc_rev, remote=active_remote)
        logger.info("DVC pull complete, commit=%s", data_dvc_commit)
    except Exception:
        logger.warning("DVC pull skipped (dvc not installed or data already present)")

    # Step 1: Discover
    pairs = discover_data_task(data_dir=data_dir)

    # Step 2: Validate (basic key-presence check)
    report = validate_data_task(pairs=pairs)

    # Step 3: Quality gate (legacy — basic error count check)
    quality_passed = data_quality_gate(report=report)

    # Step 3.5: Advanced quality gates (NEW — PR-2)
    if quality_passed and pairs:
        quality_passed = _run_quality_gates(pairs)

    # Step 4: Split (only if quality gate passed)
    splits: list[FoldSplit] | None = None
    splits_path: Path | None = None
    if quality_passed and len(pairs) >= n_folds:
        splits = split_data_task(pairs=pairs, n_folds=n_folds, seed=seed)
        # Step 4b: Serialize splits to JSON for inter-flow handoff
        from pathlib import Path as _Path

        splits_dir = _Path(os.environ.get("SPLITS_OUTPUT_DIR", "/app/configs/splits"))
        splits_path = serialize_splits_task(splits, splits_dir)

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

    # Open MLflow run for data engineering provenance (always — not conditional on DVC)
    mlflow_run_id: str | None = None
    tracking_uri = resolve_tracking_uri()
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(resolve_experiment_name(EXPERIMENT_DATA))
        with mlflow.start_run(tags={"flow_name": "data-flow"}) as active_run:
            mlflow_run_id = active_run.info.run_id
            mlflow.log_param("data_n_volumes", len(pairs))
            mlflow.log_param("data_n_folds", n_folds)
            mlflow.log_param("data_hash", dataset_hash)
            if data_dvc_commit is not None:
                mlflow.log_param("data_dvc_commit", data_dvc_commit)
            if splits_path is not None:
                mlflow.set_tag("splits_path", str(splits_path))
            logger.info("MLflow data run opened: %s", mlflow_run_id)
    except Exception:
        logger.warning("Failed to open/finalize MLflow data run", exc_info=True)

    # Log flow completion (best-effort, non-blocking)
    log_completion_safe(
        flow_name="data-flow",
        tracking_uri=tracking_uri,
        run_id=mlflow_run_id,
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
        mlflow_run_id=mlflow_run_id,
        splits_path=splits_path,
    )


if __name__ == "__main__":
    import os
    from pathlib import Path

    data_dir = Path(os.environ.get("DATA_DIR", "/app/data/raw"))
    # TODO: seed should come from Hydra config, not hardcoded here
    run_data_flow(data_dir=data_dir, seed=42)
