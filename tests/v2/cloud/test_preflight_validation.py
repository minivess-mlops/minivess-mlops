"""Pre-flight validation tests for RunPod GPU smoke tests (T0.1).

Validates:
- All required env vars are present (including HF_TOKEN for VesselFM)
- Env vars are non-empty strings
- HF_TOKEN specifically required for VesselFM fine-tuning

Auto-skips when MLFLOW_CLOUD_URI not set (same pattern as other cloud tests).
"""

from __future__ import annotations

import os

import pytest


@pytest.mark.cloud_mlflow
class TestPreflightEnvVars:
    """All required env vars for smoke tests must be set and non-empty."""

    REQUIRED_VARS = [
        "DVC_S3_ENDPOINT_URL",
        "DVC_S3_ACCESS_KEY",
        "DVC_S3_SECRET_KEY",
        "DVC_S3_BUCKET",
        "RUNPOD_API_KEY",
        "MLFLOW_CLOUD_URI",
        "MLFLOW_CLOUD_USERNAME",
        "MLFLOW_CLOUD_PASSWORD",
        "HF_TOKEN",
    ]

    def test_all_required_env_vars_present(self) -> None:
        """Every required env var must exist in the environment."""
        if not os.environ.get("MLFLOW_CLOUD_URI"):
            pytest.skip("MLFLOW_CLOUD_URI not set — skipping cloud tests")

        missing = [v for v in self.REQUIRED_VARS if v not in os.environ]
        assert not missing, f"Missing env vars: {', '.join(missing)}"

    def test_env_vars_not_empty_strings(self) -> None:
        """Env vars must have non-empty values (not just '')."""
        if not os.environ.get("MLFLOW_CLOUD_URI"):
            pytest.skip("MLFLOW_CLOUD_URI not set — skipping cloud tests")

        empty = [v for v in self.REQUIRED_VARS if os.environ.get(v, "") == ""]
        assert not empty, f"Empty env vars: {', '.join(empty)}"

    def test_hf_token_set_for_vesselfm(self) -> None:
        """HF_TOKEN must be set for VesselFM pretrained weight download."""
        if not os.environ.get("MLFLOW_CLOUD_URI"):
            pytest.skip("MLFLOW_CLOUD_URI not set — skipping cloud tests")

        hf_token = os.environ.get("HF_TOKEN", "")
        assert hf_token, (
            "HF_TOKEN not set. VesselFM requires HuggingFace token for "
            "weight download. Add HF_TOKEN to .env file."
        )
        assert hf_token.startswith("hf_"), (
            f"HF_TOKEN should start with 'hf_', got '{hf_token[:4]}...'. "
            "Verify token format at huggingface.co/settings/tokens."
        )
