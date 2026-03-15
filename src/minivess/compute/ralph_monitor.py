"""Ralph Monitor — Cloud GPU job diagnosis and monitoring loop.

Polls SkyPilot managed job status, categorizes failures from log output,
and outputs structured JSONL diagnoses for the TDD fix-iterate cycle.

Named after Geoffrey Huntley's Ralph loop technique for autonomous
AI development cycles.

See: docs/planning/ralph-loop-for-cloud-monitoring.md
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Known failure patterns — (substring, category, auto_fixable, root_cause_template)
# Using exact string matching, NOT regex (per CLAUDE.md regex ban).
_FAILURE_PATTERNS: list[tuple[str, str, bool, str]] = [
    (
        "Invalid endpoint: ${",
        "ENV_VAR_LITERAL",
        True,
        "Shell variables not expanded in DVC config — inline with shell expansion",
    ),
    (
        "command not found",
        "UV_NOT_FOUND",
        True,
        "Binary not in runner image PATH — use python/dvc directly",
    ),
    (
        "not a git repository",
        "DVC_NO_GIT",
        True,
        "DVC needs --no-scm in containers without git",
    ),
    (
        "inline_container.cc",
        "TORCH_SAVE_IO",
        True,
        "torch.save() corrupted write — use atomic save (tmp + rename)",
    ),
    (
        "CUDA out of memory",
        "OOM",
        False,
        "GPU VRAM exhausted — reduce batch_size or patch_size",
    ),
    (
        "401 Unauthorized",
        "MLFLOW_AUTH",
        False,
        "MLflow credentials invalid — check MLFLOW_CLOUD_USERNAME/PASSWORD",
    ),
    (
        "No space left on device",
        "DISK_FULL",
        True,
        "Disk full — increase disk_size in SkyPilot YAML",
    ),
    (
        "Training data missing",
        "DATA_MISSING",
        False,
        "DVC pull failed — verify data pushed to S3 and credentials",
    ),
    (
        "DVC pull failed",
        "DATA_MISSING",
        False,
        "DVC pull failed — verify data pushed to S3 and credentials",
    ),
    (
        "no longer any instances available",
        "RESOURCES_UNAVAILABLE",
        True,
        "Spot instances sold out — try different GPU type, region, or use on-demand",
    ),
    (
        "ResourcesUnavailableError",
        "RESOURCES_UNAVAILABLE",
        True,
        "Spot instances sold out — try different GPU type, region, or use on-demand",
    ),
    (
        "unauthorized",
        "REGISTRY_AUTH",
        False,
        "Docker registry auth failed — make GHCR package public or configure credentials",
    ),
    (
        "denied: permission_denied",
        "REGISTRY_AUTH",
        False,
        "Docker registry denied pull — make GHCR package public or configure credentials",
    ),
]


@dataclass
class FailureInfo:
    """Categorized failure information from a log line."""

    category: str
    auto_fixable: bool
    root_cause: str
    matched_line: str


@dataclass
class DiagnosisRecord:
    """Structured diagnosis of a failed SkyPilot job."""

    job_id: int
    status: str
    category: str
    error_line: str
    root_cause: str
    affected_files: list[str] = field(default_factory=list)
    fix_suggestion: str = ""
    auto_fixable: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(
            {
                "timestamp": self.timestamp,
                "job_id": self.job_id,
                "status": self.status,
                "category": self.category,
                "error_line": self.error_line,
                "root_cause": self.root_cause,
                "affected_files": self.affected_files,
                "fix_suggestion": self.fix_suggestion,
                "auto_fixable": self.auto_fixable,
            },
            ensure_ascii=False,
        )


def categorize_failure(log_line: str) -> FailureInfo:
    """Categorize a single log line into a known failure type.

    Uses exact string matching (NOT regex) per CLAUDE.md regex ban.

    Args:
        log_line: A single line from SkyPilot job logs.

    Returns:
        FailureInfo with category, auto_fixable flag, and root cause.
    """
    for pattern, category, auto_fixable, root_cause in _FAILURE_PATTERNS:
        if pattern in log_line:
            return FailureInfo(
                category=category,
                auto_fixable=auto_fixable,
                root_cause=root_cause,
                matched_line=log_line,
            )
    return FailureInfo(
        category="UNKNOWN",
        auto_fixable=False,
        root_cause="Unrecognized failure — manual investigation required",
        matched_line=log_line,
    )


def analyze_logs(logs: str, status: str) -> FailureInfo:
    """Analyze multi-line log output and find the most relevant failure.

    Scans all lines for known patterns. Returns the first match found,
    prioritizing ERROR lines. Falls back to UNKNOWN if no pattern matches.

    Args:
        logs: Full log output from SkyPilot job.
        status: Job status (FAILED, FAILED_SETUP, etc.)

    Returns:
        FailureInfo for the most relevant failure found.
    """
    # First pass: check ERROR lines for known patterns
    for line in logs.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        info = categorize_failure(stripped)
        if info.category != "UNKNOWN":
            return info

    # No known pattern found
    return FailureInfo(
        category="UNKNOWN",
        auto_fixable=False,
        root_cause=f"No known pattern found in {status} logs",
        matched_line=logs.split("\n")[-1] if logs.strip() else "",
    )


def parse_job_status(queue_output: str, job_id: int) -> str | None:
    """Parse job status from sky jobs queue text output.

    Finds the line starting with the given job ID and extracts the STATUS column.

    Args:
        queue_output: Full text output from `sky jobs queue`.
        job_id: The job ID to look for.

    Returns:
        Status string (RUNNING, SUCCEEDED, FAILED, etc.) or None if not found.
    """
    for line in queue_output.split("\n"):
        parts = line.split()
        if not parts:
            continue
        # First column is the job ID
        if parts[0] == str(job_id):
            # Status is the second-to-last column (before POOL)
            # Format: ID TASK NAME REQUESTED SUBMITTED TOT.DURATION JOB.DURATION #RECOVERIES STATUS POOL
            # Find STATUS by looking for known status values
            known_statuses = {
                "STARTING",
                "RUNNING",
                "SUCCEEDED",
                "FAILED",
                "FAILED_SETUP",
                "CANCELLED",
                "CANCELLING",
                "RECOVERING",
            }
            for part in parts:
                if part in known_statuses:
                    return part
    return None


def append_diagnosis(record: DiagnosisRecord, output_path: Path) -> None:
    """Append a diagnosis record to a JSONL file.

    Args:
        record: The diagnosis to append.
        output_path: Path to the JSONL file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(record.to_json() + "\n")
