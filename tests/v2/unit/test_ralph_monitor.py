"""Tests for the Ralph monitoring loop — cloud GPU job diagnosis.

The Ralph monitor polls SkyPilot job status, categorizes failures from logs,
and outputs structured JSONL diagnoses for the TDD fix-iterate cycle.

See: docs/planning/ralph-loop-for-cloud-monitoring.md
"""

from __future__ import annotations

import json
from pathlib import Path


class TestFailureCategorization:
    """Test that log lines are correctly categorized into failure types."""

    def test_env_var_literal_detected(self) -> None:
        """DVC endpoint using literal ${VAR} instead of resolved value."""
        from minivess.compute.ralph_monitor import categorize_failure

        log = (
            "ERROR: failed to connect to s3 - Invalid endpoint: ${DVC_S3_ENDPOINT_URL}"
        )
        result = categorize_failure(log)
        assert result.category == "ENV_VAR_LITERAL"
        assert result.auto_fixable is True

    def test_uv_not_found_detected(self) -> None:
        """uv binary missing from runner image."""
        from minivess.compute.ralph_monitor import categorize_failure

        log = "uv: command not found"
        result = categorize_failure(log)
        assert result.category == "UV_NOT_FOUND"
        assert result.auto_fixable is True

    def test_dvc_no_git_detected(self) -> None:
        """DVC init without --no-scm in a container without git."""
        from minivess.compute.ralph_monitor import categorize_failure

        log = "ERROR: not a git repository (or any parent up to mount point /)"
        result = categorize_failure(log)
        assert result.category == "DVC_NO_GIT"

    def test_torch_save_io_detected(self) -> None:
        """torch.save() corrupted write (inline_container.cc)."""
        from minivess.compute.ralph_monitor import categorize_failure

        log = "RuntimeError: [enforce fail at inline_container.cc:668] . unexpected pos"
        result = categorize_failure(log)
        assert result.category == "TORCH_SAVE_IO"

    def test_oom_detected(self) -> None:
        """CUDA out of memory."""
        from minivess.compute.ralph_monitor import categorize_failure

        log = "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"
        result = categorize_failure(log)
        assert result.category == "OOM"

    def test_mlflow_auth_detected(self) -> None:
        """MLflow 401 unauthorized."""
        from minivess.compute.ralph_monitor import categorize_failure

        log = "mlflow.exceptions.RestException: UNAUTHENTICATED: 401 Unauthorized"
        result = categorize_failure(log)
        assert result.category == "MLFLOW_AUTH"
        assert result.auto_fixable is False

    def test_disk_full_detected(self) -> None:
        """No space left on device."""
        from minivess.compute.ralph_monitor import categorize_failure

        log = "OSError: [Errno 28] No space left on device"
        result = categorize_failure(log)
        assert result.category == "DISK_FULL"

    def test_unknown_failure(self) -> None:
        """Unrecognized error falls to UNKNOWN category."""
        from minivess.compute.ralph_monitor import categorize_failure

        log = "SomeRandomError: something unexpected happened"
        result = categorize_failure(log)
        assert result.category == "UNKNOWN"

    def test_data_missing_detected(self) -> None:
        """Training data not pulled by DVC."""
        from minivess.compute.ralph_monitor import categorize_failure

        log = "FATAL: Training data missing — DVC pull failed in setup."
        result = categorize_failure(log)
        assert result.category == "DATA_MISSING"

    def test_resources_unavailable_detected(self) -> None:
        """RunPod spot instances sold out."""
        from minivess.compute.ralph_monitor import categorize_failure

        log = (
            "runpod.error.QueryError: There are no longer any instances "
            "available with the request specifications."
        )
        result = categorize_failure(log)
        assert result.category == "RESOURCES_UNAVAILABLE"
        assert result.auto_fixable is True

    def test_resources_unavailable_exception(self) -> None:
        """SkyPilot ResourcesUnavailableError."""
        from minivess.compute.ralph_monitor import categorize_failure

        log = "sky.exceptions.ResourcesUnavailableError: Failed to acquire resources"
        result = categorize_failure(log)
        assert result.category == "RESOURCES_UNAVAILABLE"


class TestDiagnosisRecord:
    """Test structured diagnosis output."""

    def test_diagnosis_to_json(self) -> None:
        """Diagnosis record serializes to valid JSON."""
        from minivess.compute.ralph_monitor import DiagnosisRecord

        record = DiagnosisRecord(
            job_id=4,
            status="FAILED_SETUP",
            category="ENV_VAR_LITERAL",
            error_line="Invalid endpoint: ${DVC_S3_ENDPOINT_URL}",
            root_cause="Shell vars not expanded in DVC config",
            affected_files=["deployment/skypilot/smoke_test_gpu.yaml"],
            fix_suggestion="Inline DVC remote config using shell expansion",
            auto_fixable=True,
        )
        data = json.loads(record.to_json())
        assert data["job_id"] == 4
        assert data["category"] == "ENV_VAR_LITERAL"
        assert "timestamp" in data

    def test_diagnosis_append_jsonl(self, tmp_path: Path) -> None:
        """Diagnoses are appended to JSONL file."""
        from minivess.compute.ralph_monitor import DiagnosisRecord, append_diagnosis

        out = tmp_path / "diagnoses.jsonl"
        record1 = DiagnosisRecord(
            job_id=1,
            status="FAILED_SETUP",
            category="ENV_VAR_LITERAL",
            error_line="err",
            root_cause="rc",
            affected_files=[],
            fix_suggestion="fix",
            auto_fixable=True,
        )
        record2 = DiagnosisRecord(
            job_id=2,
            status="FAILED",
            category="OOM",
            error_line="err2",
            root_cause="rc2",
            affected_files=[],
            fix_suggestion="fix2",
            auto_fixable=False,
        )
        append_diagnosis(record1, out)
        append_diagnosis(record2, out)

        lines = out.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["job_id"] == 1
        assert json.loads(lines[1])["category"] == "OOM"


class TestJobStatusParsing:
    """Test parsing of sky jobs queue output."""

    def test_parse_job_status_running(self) -> None:
        """Parse RUNNING status from sky jobs queue output."""
        from minivess.compute.ralph_monitor import parse_job_status

        queue_output = (
            "ID  TASK  NAME                 REQUESTED            SUBMITTED   "
            "TOT. DURATION  JOB DURATION  #RECOVERIES  STATUS    POOL\n"
            "4   -     minivess-smoke-test  1x[RTX4090:1][Spot]  5 mins ago  "
            "5m 30s         2m 10s        0            RUNNING   -"
        )
        status = parse_job_status(queue_output, job_id=4)
        assert status == "RUNNING"

    def test_parse_job_status_succeeded(self) -> None:
        """Parse SUCCEEDED status."""
        from minivess.compute.ralph_monitor import parse_job_status

        queue_output = (
            "ID  TASK  NAME                 REQUESTED            SUBMITTED   "
            "TOT. DURATION  JOB DURATION  #RECOVERIES  STATUS      POOL\n"
            "4   -     minivess-smoke-test  1x[RTX4090:1][Spot]  10 min ago  "
            "10m 30s        5m 10s        0            SUCCEEDED   -"
        )
        status = parse_job_status(queue_output, job_id=4)
        assert status == "SUCCEEDED"

    def test_parse_job_status_failed_setup(self) -> None:
        """Parse FAILED_SETUP status."""
        from minivess.compute.ralph_monitor import parse_job_status

        queue_output = (
            "ID  TASK  NAME                 REQUESTED            SUBMITTED   "
            "TOT. DURATION  JOB DURATION  #RECOVERIES  STATUS        POOL\n"
            "2   -     minivess-smoke-test  1x[RTX4090:1][Spot]  2 hrs ago   "
            "3m 48s         1m 46s        0            FAILED_SETUP  -"
        )
        status = parse_job_status(queue_output, job_id=2)
        assert status == "FAILED_SETUP"

    def test_parse_job_not_found(self) -> None:
        """Returns None when job ID not in queue output."""
        from minivess.compute.ralph_monitor import parse_job_status

        queue_output = (
            "ID  TASK  NAME  REQUESTED  SUBMITTED  "
            "TOT. DURATION  JOB DURATION  #RECOVERIES  STATUS  POOL\n"
        )
        status = parse_job_status(queue_output, job_id=99)
        assert status is None


class TestLogAnalysis:
    """Test multi-line log analysis to find the most relevant failure."""

    def test_analyze_setup_failure_logs(self) -> None:
        """Extract failure category from full setup log output."""
        from minivess.compute.ralph_monitor import analyze_logs

        logs = """(setup pid=1915) + dvc pull -r remote_storage
(setup pid=1915) ERROR: failed to connect to s3 (minivessdataset/files/md5) - Invalid endpoint: ${DVC_S3_ENDPOINT_URL}
(setup pid=1915) ERROR: failed to pull data from the cloud - 1 files failed to download
ERROR: Job 1's setup failed."""
        result = analyze_logs(logs, status="FAILED_SETUP")
        assert result.category == "ENV_VAR_LITERAL"

    def test_analyze_runtime_torch_save_failure(self) -> None:
        """Extract torch.save I/O error from runtime logs."""
        from minivess.compute.ralph_monitor import analyze_logs

        logs = """(minivess-smoke-test, pid=1918) RuntimeError: basic_ios::clear: iostream error
(minivess-smoke-test, pid=1918) RuntimeError: [enforce fail at inline_container.cc:668] . unexpected pos 549151040 vs 549150928
ERROR: Job 1 failed with return code list: [1]"""
        result = analyze_logs(logs, status="FAILED")
        assert result.category == "TORCH_SAVE_IO"

    def test_analyze_oom_logs(self) -> None:
        """Extract OOM error from runtime logs."""
        from minivess.compute.ralph_monitor import analyze_logs

        logs = """torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB.
GPU 0 has a total capacity of 23.65 GiB. 21.50 GiB is already allocated."""
        result = analyze_logs(logs, status="FAILED")
        assert result.category == "OOM"
