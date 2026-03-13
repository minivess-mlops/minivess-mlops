"""Verify smoke test results on cloud MLflow (#637, T3.1).

Queries the UpCloud MLflow server for smoke test experiment runs and
verifies they completed successfully with expected artifacts/params.

Usage:
    uv run python scripts/verify_smoke_test.py
    # or: make verify-smoke-test
"""

from __future__ import annotations

import os
import sys


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


def verify_smoke_test(model_family: str = "sam3_vanilla") -> bool:
    """Verify a smoke test run exists for the given model."""
    client = _get_client()

    # Search for smoke test experiments
    experiments = client.search_experiments()
    smoke_exps = [
        e for e in experiments if "smoke_test" in e.name and model_family in e.name
    ]

    if not smoke_exps:
        print(f"FAIL: No smoke test experiment found for {model_family}")
        return False

    exp = smoke_exps[0]
    print(f"Found experiment: {exp.name} (id={exp.experiment_id})")

    # Search for FINISHED runs
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        max_results=1,
    )

    if not runs:
        print(f"FAIL: No FINISHED runs in experiment {exp.name}")
        return False

    run = runs[0]
    print(f"Found run: {run.info.run_id}")
    print(f"  Status: {run.info.status}")
    print(f"  Params: {dict(run.data.params)}")
    print(f"  Metrics: {dict(run.data.metrics)}")

    # Verify expected data
    ok = True
    if "model_family" not in run.data.params:
        print("  WARN: model_family param not logged")
    if not run.data.metrics:
        print("  FAIL: No metrics logged")
        ok = False

    if ok:
        print(f"PASS: Smoke test for {model_family} verified successfully.")
    return ok


def main() -> int:
    """Verify smoke test results. Returns 0 on success, 1 on failure."""
    model = sys.argv[1] if len(sys.argv) > 1 else "sam3_vanilla"
    return 0 if verify_smoke_test(model) else 1


if __name__ == "__main__":
    sys.exit(main())
