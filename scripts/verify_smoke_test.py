"""Verify smoke test / dev run results on cloud MLflow (plan v3.1 T1.2).

Queries the UpCloud MLflow server for smoke test or dev experiment runs and
verifies they completed successfully with expected artifacts/params/metrics.

Handles both naming conventions:
- smoke_test_{uuid}_{model}: from smoke_test_gpu.yaml (Lambda/staging)
- dev_{uuid}_{model}: from dev_runpod.yaml (RunPod/dev)

GitHub: #697

Usage:
    uv run python scripts/verify_smoke_test.py
    uv run python scripts/verify_smoke_test.py sam3_hybrid
    # or: make verify-smoke-test MODEL=sam3_hybrid
"""

from __future__ import annotations

import os
import sys

# Minimum expected artifact sizes per model (bytes).
# Used to catch truncated uploads (a 0-byte checkpoint passes "exists" check).
# Values are conservative minimums (~1% of expected size).
_MIN_ARTIFACT_SIZES: dict[str, int] = {
    "dynunet": 1 * 1024 * 1024,  # ~4 MB expected, 1 MB min
    "sam3_vanilla": 1 * 1024 * 1024,  # ~4 MB expected, 1 MB min
    "sam3_hybrid": 100 * 1024 * 1024,  # ~2.5 GB expected, 100 MB min
    "sam3_topolora": 100 * 1024 * 1024,  # similar to hybrid
    "vesselfm": 50 * 1024 * 1024,  # ~120 MB expected, 50 MB min
}
_DEFAULT_MIN_SIZE = 1 * 1024 * 1024  # 1 MB fallback for unknown models

# Sanity range for train_loss (catches NaN, 0.0, runaway loss)
_TRAIN_LOSS_MIN = 0.01
_TRAIN_LOSS_MAX = 5.0


def _get_client():  # noqa: ANN202
    """Get authenticated MLflow client for cloud server."""
    import mlflow

    uri = os.environ.get("MLFLOW_CLOUD_URI")
    if not uri:
        print("ERROR: MLFLOW_CLOUD_URI not set")
        sys.exit(1)

    os.environ.setdefault("MLFLOW_TRACKING_USERNAME", "admin")
    password = os.environ.get("MLFLOW_CLOUD_PASSWORD")
    if not password:
        print("ERROR: MLFLOW_CLOUD_PASSWORD not set")
        sys.exit(1)

    return mlflow.MlflowClient(tracking_uri=uri)


def _find_experiments(client, model_family: str):  # noqa: ANN001, ANN202
    """Find experiments for a model, searching both dev_* and smoke_test_* prefixes.

    Returns list of matching experiments, most recent first.
    """
    experiments = client.search_experiments()

    # T1.2 fix: dev_runpod.yaml creates experiments named "dev_{uuid}_{model}"
    # smoke_test_gpu.yaml creates "smoke_test_{uuid}_{model}"
    # Must search both prefixes.
    matching = [
        e
        for e in experiments
        if model_family in e.name
        and (
            e.name.startswith("dev_")
            or e.name.startswith("smoke_test_")
            or e.name.startswith("smoke_")
        )
    ]
    # Sort by creation time descending (most recent first)
    matching.sort(key=lambda e: e.creation_time or 0, reverse=True)
    return matching


def _check_artifact_size(client, run_id: str, model_family: str) -> tuple[bool, str]:  # noqa: ANN001
    """Check that artifacts exist and have size above minimum threshold.

    Returns:
        (ok, message): ok=True if artifacts pass size check.
    """
    try:
        artifacts = client.list_artifacts(run_id)
    except Exception as e:  # noqa: BLE001
        return False, f"Failed to list artifacts: {e}"

    if not artifacts:
        return False, (
            "No artifacts found. "
            "This indicates MLFLOW_TRACKING_URI was not resolved correctly on the pod. "
            "Check pod logs for the 'MLflow URI resolved' diagnostic line."
        )

    # Sum total artifact size
    total_size = sum(a.file_size or 0 for a in artifacts if not a.is_dir)
    min_size = _MIN_ARTIFACT_SIZES.get(model_family, _DEFAULT_MIN_SIZE)
    min_size_mb = min_size / 1024 / 1024

    if total_size < min_size:
        actual_mb = total_size / 1024 / 1024
        return False, (
            f"Artifact total size {actual_mb:.1f} MB < minimum {min_size_mb:.0f} MB. "
            f"Checkpoint appears truncated (upload failed mid-transfer). "
            f"Artifacts: {[a.path for a in artifacts[:5]]}"
        )

    actual_mb = total_size / 1024 / 1024
    return (
        True,
        f"{len(artifacts)} artifacts, total {actual_mb:.1f} MB (min {min_size_mb:.0f} MB)",
    )


def _check_train_loss(run, model_family: str) -> tuple[bool, str]:  # noqa: ANN001
    """Check train_loss is present and in sanity range.

    Returns:
        (ok, message): ok=True if train_loss is sensible.
    """
    metrics = run.data.metrics
    if not metrics:
        return False, "No metrics logged at all"

    # Find train_loss (may be logged as "train_loss" or "train/loss")
    loss_key = None
    for key in metrics:
        if "train" in key.lower() and "loss" in key.lower():
            loss_key = key
            break

    if loss_key is None:
        return (
            False,
            f"No train_loss metric found. Available metrics: {list(metrics.keys())[:10]}",
        )

    loss_val = metrics[loss_key]
    import math

    if math.isnan(loss_val) or math.isinf(loss_val):
        return False, f"{loss_key} = {loss_val} (NaN/Inf — training diverged)"

    if loss_val < _TRAIN_LOSS_MIN:
        return (
            False,
            f"{loss_key} = {loss_val:.4f} < {_TRAIN_LOSS_MIN} (suspiciously low — no gradient?)",
        )

    if loss_val > _TRAIN_LOSS_MAX:
        return False, (
            f"{loss_key} = {loss_val:.4f} > {_TRAIN_LOSS_MAX} "
            f"(loss did not decrease — check model init / lr)"
        )

    return (
        True,
        f"{loss_key} = {loss_val:.4f} (in range [{_TRAIN_LOSS_MIN}, {_TRAIN_LOSS_MAX}])",
    )


def verify_smoke_test(model_family: str = "sam3_vanilla") -> bool:
    """Verify a smoke test or dev run exists for the given model.

    Searches both dev_* and smoke_test_* experiment naming conventions.

    Args:
        model_family: Model name (dynunet, sam3_vanilla, sam3_hybrid, vesselfm, ...)

    Returns:
        True if verification passes, False otherwise.
    """
    client = _get_client()

    # Find experiments (both dev_ and smoke_test_ prefixes)
    matching_exps = _find_experiments(client, model_family)

    if not matching_exps:
        print(f"FAIL: No experiment found for model '{model_family}'")
        print(
            f"  Searched for experiments containing '{model_family}' with prefix dev_* or smoke_test_*"
        )
        print(f"  Run: make dev-gpu MODEL={model_family}")
        return False

    exp = matching_exps[0]
    print(f"Found experiment: {exp.name} (id={exp.experiment_id})")

    # Search for FINISHED runs
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        max_results=1,
    )

    if not runs:
        print(f"FAIL: No FINISHED runs in experiment {exp.name}")
        # Check if there are FAILED runs
        failed_runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="attributes.status = 'FAILED'",
            max_results=1,
        )
        if failed_runs:
            print(f"  Found FAILED run: {failed_runs[0].info.run_id}")
            print(f"  Tags: {failed_runs[0].data.tags}")
        return False

    run = runs[0]
    print(f"Found run: {run.info.run_id}")
    print(f"  Status: {run.info.status}")

    ok = True

    # Check params
    params = run.data.params
    print(f"  Params ({len(params)} total): {dict(list(params.items())[:5])}...")
    if "model_family" not in params:
        print("  WARN: model_family param not logged")
    elif params["model_family"] != model_family:
        print(
            f"  WARN: model_family mismatch: expected {model_family}, got {params['model_family']}"
        )

    # T1.2: Sanity-check train_loss range
    loss_ok, loss_msg = _check_train_loss(run, model_family)
    if loss_ok:
        print(f"  PASS metrics: {loss_msg}")
    else:
        print(f"  FAIL metrics: {loss_msg}")
        ok = False

    # T1.2: Check artifact existence AND size
    art_ok, art_msg = _check_artifact_size(client, run.info.run_id, model_family)
    if art_ok:
        print(f"  PASS artifacts: {art_msg}")
    else:
        print(f"  FAIL artifacts: {art_msg}")
        ok = False

    if ok:
        print(f"PASS: {model_family} verified successfully on UpCloud MLflow.")
    else:
        print(f"FAIL: {model_family} verification failed — see details above.")
    return ok


def main() -> int:
    """Verify smoke test / dev run results. Returns 0 on success, 1 on failure."""
    model = sys.argv[1] if len(sys.argv) > 1 else "sam3_vanilla"
    return 0 if verify_smoke_test(model) else 1


if __name__ == "__main__":
    sys.exit(main())
