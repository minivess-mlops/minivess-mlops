"""Mocked unit tests for GCP preflight checks.

Validates each preflight check function with mocked ``_run`` subprocess calls,
ensuring correct (bool, str) return types and message content without requiring
real cloud infrastructure.

TDD Task 3.4 (#961).
Pattern: tests/v2/unit/deployment/test_docker_freshness_gate.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

PREFLIGHT = Path("scripts/preflight_gcp.py")


def _import_preflight():
    """Import preflight module dynamically (same pattern as freshness gate test)."""
    spec = importlib.util.spec_from_file_location("preflight", PREFLIGHT)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# check_gcs_bucket
# ---------------------------------------------------------------------------


class TestCheckGcsBucketMocked:
    """Mock tests for check_gcs_bucket — GCS DVC bucket accessibility."""

    def test_gcs_bucket_passes_on_success(self) -> None:
        mod = _import_preflight()
        mock = MagicMock(returncode=0, stdout="gs://bucket/file1\ngs://bucket/file2\n")
        with patch.object(mod, "_run", return_value=mock):
            ok, msg = mod.check_gcs_bucket()
        assert ok is True
        assert "accessible" in msg.lower()

    def test_gcs_bucket_fails_on_error(self) -> None:
        mod = _import_preflight()
        mock = MagicMock(returncode=1, stdout="", stderr="AccessDenied")
        with patch.object(mod, "_run", return_value=mock):
            ok, msg = mod.check_gcs_bucket()
        assert ok is False
        assert "not accessible" in msg.lower()

    def test_gcs_bucket_returns_tuple_bool_str(self) -> None:
        mod = _import_preflight()
        mock = MagicMock(returncode=0, stdout="gs://bucket/file1\n")
        with patch.object(mod, "_run", return_value=mock):
            result = mod.check_gcs_bucket()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)


# ---------------------------------------------------------------------------
# check_skypilot_gcp
# ---------------------------------------------------------------------------


class TestCheckSkypilotGcpMocked:
    """Mock tests for check_skypilot_gcp — SkyPilot GCP backend."""

    def test_skypilot_gcp_passes(self) -> None:
        mod = _import_preflight()
        mock = MagicMock(returncode=0, stdout="GCP: enabled")
        with patch.object(mod, "_run", return_value=mock):
            # Also need to mock the sky binary existence check
            with patch.object(Path, "exists", return_value=True):
                ok, msg = mod.check_skypilot_gcp()
        assert ok is True
        assert "enabled" in msg.lower()

    def test_skypilot_gcp_fails(self) -> None:
        mod = _import_preflight()
        mock = MagicMock(returncode=1, stdout="")
        with patch.object(mod, "_run", return_value=mock):
            with patch.object(Path, "exists", return_value=True):
                ok, msg = mod.check_skypilot_gcp()
        assert ok is False
        assert "not enabled" in msg.lower() or "check gcp" in msg.lower()

    def test_skypilot_gcp_fails_when_binary_missing(self) -> None:
        mod = _import_preflight()
        # When sky binary does not exist, check should fail without calling _run
        original_exists = Path.exists

        def mock_exists(self_path: Path) -> bool:
            if "sky" in str(self_path) and ".venv" in str(self_path):
                return False
            return original_exists(self_path)

        with patch.object(Path, "exists", mock_exists):
            ok, msg = mod.check_skypilot_gcp()
        assert ok is False
        assert "not found" in msg.lower()


# ---------------------------------------------------------------------------
# check_docker_image
# ---------------------------------------------------------------------------


class TestCheckDockerImageMocked:
    """Mock tests for check_docker_image — GAR Docker image existence."""

    def test_docker_image_passes(self) -> None:
        mod = _import_preflight()
        mock = MagicMock(returncode=0, stdout="{}")
        with patch.object(mod, "_run", return_value=mock):
            ok, msg = mod.check_docker_image()
        assert ok is True
        assert "exists" in msg.lower()

    def test_docker_image_fails(self) -> None:
        mod = _import_preflight()
        mock = MagicMock(returncode=1, stdout="")
        with patch.object(mod, "_run", return_value=mock):
            ok, msg = mod.check_docker_image()
        assert ok is False
        assert "not found" in msg.lower()

    def test_docker_image_message_contains_image_ref(self) -> None:
        mod = _import_preflight()
        mock = MagicMock(returncode=0, stdout="{}")
        with patch.object(mod, "_run", return_value=mock):
            ok, msg = mod.check_docker_image()
        assert ok is True
        # Verify the message references the actual GAR image (read from config)
        gar_config = yaml.safe_load(
            Path("configs/registry/gar.yaml").read_text(encoding="utf-8")
        )
        expected_server = gar_config["server"]
        assert expected_server in msg


# ---------------------------------------------------------------------------
# check_checkpoint_bucket
# ---------------------------------------------------------------------------


class TestCheckCheckpointBucketMocked:
    """Mock tests for check_checkpoint_bucket — checkpoint GCS bucket."""

    def test_checkpoint_bucket_passes(self) -> None:
        mod = _import_preflight()
        mock = MagicMock(returncode=0, stdout="gs://checkpoints/model1\n")
        with patch.object(mod, "_run", return_value=mock):
            ok, msg = mod.check_checkpoint_bucket()
        assert ok is True
        assert "accessible" in msg.lower()

    def test_checkpoint_bucket_fails(self) -> None:
        mod = _import_preflight()
        mock = MagicMock(returncode=1, stdout="", stderr="BucketNotFound")
        with patch.object(mod, "_run", return_value=mock):
            ok, msg = mod.check_checkpoint_bucket()
        assert ok is False
        assert "not accessible" in msg.lower()

    def test_checkpoint_bucket_returns_tuple_bool_str(self) -> None:
        mod = _import_preflight()
        mock = MagicMock(returncode=0, stdout="gs://checkpoints/file\n")
        with patch.object(mod, "_run", return_value=mock):
            result = mod.check_checkpoint_bucket()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_checkpoint_bucket_message_contains_bucket_name(self) -> None:
        mod = _import_preflight()
        mock = MagicMock(returncode=0, stdout="gs://checkpoints/file\n")
        with patch.object(mod, "_run", return_value=mock):
            ok, msg = mod.check_checkpoint_bucket()
        assert ok is True
        assert "minivess-mlops-checkpoints" in msg


# ---------------------------------------------------------------------------
# check_controller_health
# ---------------------------------------------------------------------------


class TestCheckControllerHealthMocked:
    """Mock tests for check_controller_health — SkyPilot controller state."""

    def test_controller_up(self) -> None:
        mod = _import_preflight()
        mock = MagicMock(
            returncode=0,
            stdout="sky-jobs-controller-abc123  UP  gcp  n2-standard-4  europe-west4\n",
        )
        with patch.object(mod, "_run", return_value=mock):
            ok, msg = mod.check_controller_health()
        assert ok is True
        assert "UP" in msg

    def test_controller_stopped_fails(self) -> None:
        mod = _import_preflight()
        mock = MagicMock(
            returncode=0,
            stdout="sky-jobs-controller-abc123  STOPPED  gcp  n2-standard-4\n",
        )
        with patch.object(mod, "_run", return_value=mock):
            ok, msg = mod.check_controller_health()
        assert ok is False
        assert "STOPPED" in msg

    def test_controller_error_fails(self) -> None:
        mod = _import_preflight()
        mock = MagicMock(
            returncode=0,
            stdout="sky-jobs-controller-abc123  ERROR  gcp  n2-standard-4\n",
        )
        with patch.object(mod, "_run", return_value=mock):
            ok, msg = mod.check_controller_health()
        assert ok is False
        assert "ERROR" in msg

    def test_controller_init_passes_with_caution(self) -> None:
        mod = _import_preflight()
        mock = MagicMock(
            returncode=0,
            stdout="sky-jobs-controller-abc123  INIT  gcp  n2-standard-4\n",
        )
        with patch.object(mod, "_run", return_value=mock):
            ok, msg = mod.check_controller_health()
        assert ok is True
        assert "INIT" in msg

    def test_no_controller_found(self) -> None:
        mod = _import_preflight()
        mock = MagicMock(
            returncode=0,
            stdout="my-cluster  UP  gcp  a2-highgpu-1g  us-central1\n",
        )
        with patch.object(mod, "_run", return_value=mock):
            ok, msg = mod.check_controller_health()
        assert ok is True
        assert "no controller" in msg.lower() or "will be created" in msg.lower()

    def test_sky_status_unavailable(self) -> None:
        mod = _import_preflight()
        mock = MagicMock(returncode=1, stdout="")
        with patch.object(mod, "_run", return_value=mock):
            ok, msg = mod.check_controller_health()
        assert ok is True
        assert "skipped" in msg.lower() or "unavailable" in msg.lower()

    def test_empty_sky_status(self) -> None:
        mod = _import_preflight()
        mock = MagicMock(returncode=0, stdout="")
        with patch.object(mod, "_run", return_value=mock):
            ok, msg = mod.check_controller_health()
        assert ok is True
        assert "will be created" in msg.lower()


# ---------------------------------------------------------------------------
# check_gcp_gpu_quota
# ---------------------------------------------------------------------------


class TestCheckGcpGpuQuotaMocked:
    """Mock tests for check_gcp_gpu_quota — L4 GPU quota in europe-west4."""

    def _make_quota_response(
        self, metric: str = "NVIDIA_L4_GPUS", limit: float = 4, usage: float = 0
    ) -> str:
        """Build a JSON quota response."""
        import json

        return json.dumps({"quotas": [{"metric": metric, "limit": limit, "usage": usage}]})

    def test_gpu_quota_passes(self) -> None:
        mod = _import_preflight()
        mock = MagicMock(returncode=0, stdout=self._make_quota_response(limit=4, usage=0))
        with patch.object(mod, "_run", return_value=mock):
            ok, msg = mod.check_gcp_gpu_quota()
        assert ok is True
        assert "OK" in msg or "available" in msg.lower()

    def test_gpu_quota_zero_fails(self) -> None:
        mod = _import_preflight()
        mock = MagicMock(returncode=0, stdout=self._make_quota_response(limit=0, usage=0))
        with patch.object(mod, "_run", return_value=mock):
            ok, msg = mod.check_gcp_gpu_quota()
        assert ok is False
        assert "0" in msg

    def test_gpu_quota_gcloud_unavailable(self) -> None:
        mod = _import_preflight()
        mock = MagicMock(returncode=1, stdout="")
        with patch.object(mod, "_run", return_value=mock):
            ok, msg = mod.check_gcp_gpu_quota()
        assert ok is True
        assert "skipped" in msg.lower()

    def test_gpu_quota_metric_not_found(self) -> None:
        mod = _import_preflight()
        mock = MagicMock(
            returncode=0,
            stdout=self._make_quota_response(metric="SOME_OTHER_METRIC"),
        )
        with patch.object(mod, "_run", return_value=mock):
            ok, msg = mod.check_gcp_gpu_quota()
        assert ok is True
        assert "not found" in msg.lower() or "check manually" in msg.lower()
