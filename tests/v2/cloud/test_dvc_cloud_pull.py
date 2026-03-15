"""Cloud integration tests for DVC pull from UpCloud S3 (#632, T0.3).

Requires DVC_S3_* env vars to be set. Auto-skips when not configured.

Run: uv run pytest tests/v2/cloud/test_dvc_cloud_pull.py -v
"""

from __future__ import annotations

import os
import subprocess

import pytest


def _has_dvc_s3_creds() -> bool:
    """Check if DVC S3 credentials are available."""
    return bool(
        os.environ.get("DVC_S3_ENDPOINT_URL")
        and os.environ.get("DVC_S3_ACCESS_KEY")
        and os.environ.get("DVC_S3_SECRET_KEY")
    )


@pytest.mark.cloud_mlflow
class TestDvcCloudPull:
    """Integration tests: real DVC pull from UpCloud S3."""

    def test_dvc_status_against_upcloud(self) -> None:
        """dvc status -r upcloud succeeds with real credentials."""
        if not _has_dvc_s3_creds():
            pytest.skip("DVC_S3_* env vars not set — skipping cloud DVC tests")
        result = subprocess.run(
            ["dvc", "status", "-r", "upcloud"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"dvc status -r upcloud failed: {result.stderr}"

    def test_s3_bucket_accessible_via_boto3(self) -> None:
        """UpCloud S3 bucket is accessible with configured credentials."""
        if not _has_dvc_s3_creds():
            pytest.skip("DVC_S3_* env vars not set — skipping cloud DVC tests")
        import boto3

        client = boto3.client(
            "s3",
            endpoint_url=os.environ["DVC_S3_ENDPOINT_URL"],
            aws_access_key_id=os.environ["DVC_S3_ACCESS_KEY"],
            aws_secret_access_key=os.environ["DVC_S3_SECRET_KEY"],
        )
        bucket = os.environ.get("DVC_S3_BUCKET", "minivess-dvc-data")
        # Should not raise — bucket exists and is accessible
        response = client.head_bucket(Bucket=bucket)
        assert response["ResponseMetadata"]["HTTPStatusCode"] == 200
