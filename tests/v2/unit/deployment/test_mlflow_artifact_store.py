"""Tests for MLflow artifact GCS store configuration (Task 2.E — #952).

Validates that the MLflow artifact store uses GCS, not HTTP upload, and that
retry/timeout env vars are configured for cloud resilience.

These tests operate WITHOUT modifying yaml_contract.yaml (Rule #31 STOP POINT).

Rule #16: yaml.safe_load() — no regex for structured data.
"""

from __future__ import annotations

from pathlib import Path

import yaml

SKYPILOT_DIR = Path("deployment/skypilot")
FACTORIAL_YAML = SKYPILOT_DIR / "train_factorial.yaml"


def _load_yaml(path: Path) -> dict:
    """Load a YAML file using safe_load (Rule #16: no regex)."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# MLflow HTTP retry resilience
# ---------------------------------------------------------------------------


class TestMlflowHttpRetry:
    """MLflow HTTP retry env vars protect against transient cloud failures.

    Without retries, a single network blip during a multi-hour training run
    causes silent metric loss or a crash. MLFLOW_HTTP_REQUEST_MAX_RETRIES
    ensures the client retries before giving up.
    """

    def test_train_factorial_has_mlflow_http_retry(self) -> None:
        """MLFLOW_HTTP_REQUEST_MAX_RETRIES must be in train_factorial.yaml envs.

        This env var controls how many times the MLflow client retries a
        failed HTTP request. Without it, the default is low (typically 3)
        which is insufficient for spot VMs with intermittent connectivity.
        """
        config = _load_yaml(FACTORIAL_YAML)
        envs = config.get("envs", {})
        assert "MLFLOW_HTTP_REQUEST_MAX_RETRIES" in envs, (
            "train_factorial.yaml envs MUST include MLFLOW_HTTP_REQUEST_MAX_RETRIES. "
            "Without it, transient network failures during training will crash or "
            "silently lose metrics. Spot VMs need high retry counts."
        )
        # Value should be a reasonable retry count (not "0" or "1")
        retries = int(envs["MLFLOW_HTTP_REQUEST_MAX_RETRIES"])
        assert retries >= 5, (
            f"MLFLOW_HTTP_REQUEST_MAX_RETRIES={retries} is too low for spot VMs. "
            f"Spot preemption recovery can cause multiple transient failures. "
            f"Minimum recommended: 5, current best practice: 10."
        )


class TestMlflowHttpTimeout:
    """MLflow HTTP timeout prevents indefinite hangs on unresponsive servers.

    Without a timeout, the MLflow client can hang forever waiting for a
    response from Cloud Run, blocking training progress.
    """

    def test_train_factorial_has_mlflow_timeout(self) -> None:
        """MLFLOW_HTTP_REQUEST_TIMEOUT must be in train_factorial.yaml envs.

        This env var sets the maximum time (in seconds) the MLflow client
        waits for an HTTP response. Without it, the client may hang
        indefinitely if Cloud Run becomes unresponsive.
        """
        config = _load_yaml(FACTORIAL_YAML)
        envs = config.get("envs", {})
        assert "MLFLOW_HTTP_REQUEST_TIMEOUT" in envs, (
            "train_factorial.yaml envs MUST include MLFLOW_HTTP_REQUEST_TIMEOUT. "
            "Without it, the MLflow client can hang indefinitely waiting for "
            "a Cloud Run response, blocking training."
        )
        # Value should be reasonable (not "0" which means no timeout)
        timeout = int(envs["MLFLOW_HTTP_REQUEST_TIMEOUT"])
        assert timeout >= 30, (
            f"MLFLOW_HTTP_REQUEST_TIMEOUT={timeout}s is too low. "
            f"Large artifact uploads to GCS can take >30s. "
            f"Minimum recommended: 30, current best practice: 300."
        )


# ---------------------------------------------------------------------------
# GCS URI validation for any MLFLOW env vars referencing buckets
# ---------------------------------------------------------------------------


class TestArtifactBucketGcsUri:
    """If any MLFLOW env var contains a GCS bucket URI, it must be valid.

    GCS URIs must start with gs:// and have no trailing slash (which can
    cause path-join bugs in MLflow artifact resolution).
    """

    def test_artifact_bucket_env_is_gcs_uri(self) -> None:
        """Any MLFLOW env var with a bucket URI must start with gs:// and have no trailing slash.

        Scans all env vars in train_factorial.yaml whose key contains 'MLFLOW'
        and whose value starts with 'gs://'. Validates:
        1. URI starts with gs:// (not s3://, http://, etc.)
        2. URI has no trailing slash (prevents double-slash in artifact paths)
        3. Bucket name is non-empty
        """
        config = _load_yaml(FACTORIAL_YAML)
        envs = config.get("envs", {})

        gcs_mlflow_vars = {}
        for key, value in envs.items():
            if "MLFLOW" not in key:
                continue
            str_value = str(value)
            # Check if this value looks like a cloud storage URI
            if str_value.startswith("gs://") or str_value.startswith("s3://"):
                gcs_mlflow_vars[key] = str_value

        for key, uri in gcs_mlflow_vars.items():
            assert uri.startswith("gs://"), (
                f"{key}={uri} must use GCS (gs://), not S3 or HTTP. "
                f"MLflow artifact store on GCP must use GCS directly."
            )
            assert not uri.endswith("/"), (
                f"{key}={uri} has trailing slash — this causes double-slash "
                f"bugs in MLflow artifact path resolution. Remove the trailing /."
            )
            # Extract bucket name: gs://bucket-name/optional-prefix
            bucket_part = uri[len("gs://") :]
            bucket_name = bucket_part.split("/")[0]
            assert len(bucket_name) > 0, (
                f"{key}={uri} has empty bucket name after gs://"
            )

    def test_file_mounts_gcs_uris_have_no_trailing_slash(self) -> None:
        """GCS URIs in file_mounts must have no trailing slash.

        file_mounts are the primary mechanism for checkpoint persistence
        via GCS. Trailing slashes cause path-join issues with SkyPilot's
        MOUNT_CACHED mode.
        """
        config = _load_yaml(FACTORIAL_YAML)
        file_mounts = config.get("file_mounts", {})

        for mount_path, mount_config in file_mounts.items():
            source = ""
            if isinstance(mount_config, dict):
                source = mount_config.get("source", "")
            elif isinstance(mount_config, str):
                source = mount_config

            if source.startswith("gs://"):
                assert not source.endswith("/"), (
                    f"file_mount {mount_path}: source={source} has trailing slash. "
                    f"Remove it to prevent path-join bugs in SkyPilot MOUNT_CACHED."
                )

    def test_no_http_artifact_upload_uri(self) -> None:
        """MLFLOW env vars must NOT contain HTTP URIs for artifact storage.

        Artifact uploads via HTTP go through Cloud Run, which has a 32 MB
        request body limit. Checkpoints (~900 MB) MUST go directly to GCS.
        MLFLOW_TRACKING_URI is HTTP (that's correct — it's for the tracking
        API, not artifacts). But any ARTIFACT-specific URI must be gs://.
        """
        config = _load_yaml(FACTORIAL_YAML)
        envs = config.get("envs", {})

        # These env var names, if present, should use GCS not HTTP
        artifact_env_vars = [
            "MLFLOW_DEFAULT_ARTIFACT_ROOT",
            "MLFLOW_ARTIFACTS_DESTINATION",
            "MLFLOW_ARTIFACT_ROOT",
        ]

        for var_name in artifact_env_vars:
            if var_name in envs:
                value = str(envs[var_name])
                # Skip ${VAR} placeholders (expanded at runtime)
                if value.startswith("${"):
                    continue
                assert not value.startswith("http"), (
                    f"{var_name}={value} uses HTTP for artifact storage. "
                    f"This routes artifacts through Cloud Run (32 MB limit). "
                    f"Use gs:// for direct GCS upload. See: issue #878."
                )
