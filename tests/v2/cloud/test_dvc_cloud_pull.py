"""Cloud integration tests for DVC pull from AWS S3 (remote_storage).

Tests connectivity to s3://minivessdataset (public bucket — no credentials needed).
UpCloud S3 archived 2026-03-16 — replaced by AWS S3 public bucket as cloud fallback.
Network Volume is the primary data source (cache-first, no DVC pull needed on repeat runs).

Requires: dvc installed (uv sync --all-extras). Auto-skips if boto3 not available.

Run: uv run pytest tests/v2/cloud/test_dvc_cloud_pull.py -v
"""

from __future__ import annotations

import subprocess

import pytest


@pytest.mark.cloud_mlflow
class TestDvcCloudPull:
    """Integration tests: DVC status check against AWS S3 public bucket."""

    def test_dvc_status_against_remote_storage(self) -> None:
        """dvc status -r remote_storage succeeds (public bucket, no credentials)."""
        result = subprocess.run(
            ["dvc", "status", "-r", "remote_storage"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"dvc status -r remote_storage failed: {result.stderr}\n"
            "Ensure .dvc/config has remote_storage pointing to s3://minivessdataset"
        )

    def test_s3_public_bucket_accessible_via_boto3(self) -> None:
        """AWS S3 public bucket s3://minivessdataset is accessible without credentials."""
        try:
            import boto3
            from botocore import UNSIGNED
            from botocore.client import Config
        except ImportError:
            pytest.skip("boto3 not installed — skipping S3 accessibility test")

        client = boto3.client(
            "s3",
            region_name="eu-north-1",
            config=Config(signature_version=UNSIGNED),
        )
        # Public bucket — should be accessible without credentials
        response = client.head_bucket(Bucket="minivessdataset")
        assert response["ResponseMetadata"]["HTTPStatusCode"] == 200, (
            "s3://minivessdataset should be publicly accessible. "
            "Check bucket policy on AWS console."
        )
