"""Cloud MLflow test fixtures (#625).

Provides provider-agnostic fixtures for testing against live cloud
MLflow deployments. All credentials come from MLFLOW_CLOUD_* env vars.

Secret fields use ``field(repr=False)`` to prevent credential leaks
in pytest failure output (reviewer finding: all 3 flagged this).
"""

from __future__ import annotations

import contextlib
import os
import uuid
from dataclasses import dataclass, field

import pytest


@dataclass(frozen=True)
class CloudMLflowConnection:
    """Provider-agnostic cloud MLflow connection details.

    Secret fields use field(repr=False) to prevent credential leaks
    in pytest failure output.
    """

    tracking_uri: str
    username: str
    password: str = field(repr=False)
    s3_endpoint: str = ""
    s3_access_key: str = field(default="", repr=False)
    s3_secret_key: str = field(default="", repr=False)
    s3_bucket: str = "mlflow-artifacts"
    provider_name: str = "unknown"


_TEST_SESSION_ID = uuid.uuid4().hex[:8]


@pytest.fixture(scope="session")
def test_run_id() -> str:
    """Unique prefix for this test session's experiments.

    UUID-based to prevent interference between concurrent test runs.
    """
    return f"_test_{_TEST_SESSION_ID}"


@pytest.fixture(scope="session")
def cloud_mlflow_connection() -> CloudMLflowConnection:
    """Read cloud MLflow connection from env vars.

    Skips entire session if credentials not available.
    """
    uri = os.environ.get("MLFLOW_CLOUD_URI")
    if not uri:
        pytest.skip("MLFLOW_CLOUD_URI not set — skipping cloud tests")
    password = os.environ.get("MLFLOW_CLOUD_PASSWORD")
    if not password:
        pytest.skip("MLFLOW_CLOUD_PASSWORD not set — skipping cloud tests")
    return CloudMLflowConnection(
        tracking_uri=uri,
        username=os.environ.get("MLFLOW_CLOUD_USERNAME", "admin"),
        password=password,
        s3_endpoint=os.environ.get("MLFLOW_CLOUD_S3_ENDPOINT", ""),
        s3_access_key=os.environ.get("MLFLOW_CLOUD_S3_ACCESS_KEY", ""),
        s3_secret_key=os.environ.get("MLFLOW_CLOUD_S3_SECRET_KEY", ""),
        s3_bucket=os.environ.get("MLFLOW_CLOUD_S3_BUCKET", "mlflow-artifacts"),
        provider_name=os.environ.get("MLFLOW_CLOUD_PROVIDER", "unknown"),
    )


@pytest.fixture(scope="session")
def cloud_mlflow_client(cloud_mlflow_connection: CloudMLflowConnection):
    """Authenticated MlflowClient for cloud server.

    Uses save/restore for env vars instead of bare os.environ mutation
    (reviewer finding: session-scoped mutation leaks into other tests).
    """
    from mlflow import MlflowClient

    # Save existing values
    saved_user = os.environ.get("MLFLOW_TRACKING_USERNAME")
    saved_pass = os.environ.get("MLFLOW_TRACKING_PASSWORD")

    os.environ["MLFLOW_TRACKING_USERNAME"] = cloud_mlflow_connection.username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = cloud_mlflow_connection.password

    client = MlflowClient(tracking_uri=cloud_mlflow_connection.tracking_uri)
    # Verify connectivity
    client.search_experiments(max_results=1)
    yield client

    # Restore original values
    if saved_user is not None:
        os.environ["MLFLOW_TRACKING_USERNAME"] = saved_user
    else:
        os.environ.pop("MLFLOW_TRACKING_USERNAME", None)
    if saved_pass is not None:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = saved_pass
    else:
        os.environ.pop("MLFLOW_TRACKING_PASSWORD", None)


@pytest.fixture(scope="session")
def cloud_s3_client(cloud_mlflow_connection: CloudMLflowConnection):
    """boto3 S3 client configured for the cloud provider's endpoint."""
    import boto3

    if not cloud_mlflow_connection.s3_endpoint:
        pytest.skip("MLFLOW_CLOUD_S3_ENDPOINT not set — skipping S3 tests")
    return boto3.client(
        "s3",
        endpoint_url=cloud_mlflow_connection.s3_endpoint,
        aws_access_key_id=cloud_mlflow_connection.s3_access_key,
        aws_secret_access_key=cloud_mlflow_connection.s3_secret_key,
    )


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_experiments(request: pytest.FixtureRequest):
    """Bidirectional cleanup: delete _test_* experiments before AND after.

    Handles crash recovery (reviewer finding: yield-only misses SIGKILL).
    Only runs when cloud credentials are available (otherwise no-op).
    """
    uri = os.environ.get("MLFLOW_CLOUD_URI")
    if not uri:
        yield
        return

    from mlflow.entities import ViewType

    # Get cloud client from the fixture manager (avoid circular dep)
    try:
        client = request.getfixturevalue("cloud_mlflow_client")
    except pytest.FixtureLookupError:
        yield
        return

    def _cleanup() -> None:
        for exp in client.search_experiments(view_type=ViewType.ALL):
            if exp.name.startswith("_test_"):
                with contextlib.suppress(Exception):
                    client.delete_experiment(exp.experiment_id)

    _cleanup()  # Startup: clean stale experiments from crashed runs
    yield
    _cleanup()  # Teardown: clean this session's experiments
