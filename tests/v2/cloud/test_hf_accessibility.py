"""Pre-flight HuggingFace accessibility tests (T0.5).

Validates:
- bwittmann/vesselFM repo exists on HuggingFace Hub
- vesselFM_base.pt weight file is listed in the repo
- HF_TOKEN auth works (repo may require authentication)

Auto-skips when MLFLOW_TRACKING_URI is not a remote URL (same pattern as other cloud tests).
"""

from __future__ import annotations

import os

import pytest


@pytest.mark.cloud_mlflow
class TestHuggingFaceAccessibility:
    """Verify VesselFM weights are downloadable from HuggingFace."""

    def test_vesselfm_hf_repo_exists(self) -> None:
        """bwittmann/vesselFM repo exists and is accessible."""
        uri = os.environ.get("MLFLOW_TRACKING_URI", "")
        if not uri or "localhost" in uri or not uri.startswith("http"):
            pytest.skip(
                "MLFLOW_TRACKING_URI not set to remote URL — skipping cloud tests"
            )

        from huggingface_hub import repo_info

        hf_token = os.environ.get("HF_TOKEN")
        info = repo_info("bwittmann/vesselFM", token=hf_token)
        assert info.id == "bwittmann/vesselFM"

    def test_vesselfm_weight_file_listed(self) -> None:
        """vesselFM_base.pt exists in the repo file listing."""
        uri = os.environ.get("MLFLOW_TRACKING_URI", "")
        if not uri or "localhost" in uri or not uri.startswith("http"):
            pytest.skip(
                "MLFLOW_TRACKING_URI not set to remote URL — skipping cloud tests"
            )

        from huggingface_hub import list_repo_files

        hf_token = os.environ.get("HF_TOKEN")
        files = list_repo_files("bwittmann/vesselFM", token=hf_token)
        assert "vesselFM_base.pt" in files, (
            f"vesselFM_base.pt not found in repo. Available files: {files}"
        )
