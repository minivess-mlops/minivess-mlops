"""Parse SkyPilot `sky jobs queue` output into structured job data.

Handles multi-job batch output, various statuses (PENDING, STARTING, RUNNING,
SUCCEEDED, FAILED, FAILED_SETUP, CANCELLED, RECOVERING), and duration parsing.

Uses str.split() for parsing — NO regex (CLAUDE.md Rule 16).

See: docs/planning/v0-2_archive/original_docs/experiment-harness-improvement-plan.xml T1.2
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# All known SkyPilot job statuses.
KNOWN_STATUSES = frozenset({
    "PENDING",
    "STARTING",
    "RUNNING",
    "SUCCEEDED",
    "FAILED",
    "FAILED_SETUP",
    "CANCELLED",
    "RECOVERING",
    "CANCELLING",
})


@dataclass
class QueuedJob:
    """A single job parsed from sky jobs queue output.

    Attributes
    ----------
    job_id:
        SkyPilot managed job ID.
    name:
        Job name (condition identifier).
    requested:
        Resource request string (e.g., "1x[A100-80GB:1, L4:1][Spot]").
    total_duration_minutes:
        Total wall-clock time since submission (or None).
    job_duration_minutes:
        Actual job execution time (or None if not yet running).
    recovery_count:
        Number of spot preemption recoveries.
    status:
        Current job status.
    """

    job_id: int
    name: str
    requested: str
    total_duration_minutes: float | None
    job_duration_minutes: float | None
    recovery_count: int
    status: str


def parse_duration_to_minutes(duration_str: str) -> float | None:
    """Parse a SkyPilot duration string to minutes.

    Handles formats: "1h 22m", "45m", "2h", "5m", "10h 2m 17s",
    "1d 4h 55m 45s", "3m 20s", "-" (None).

    Uses str.split() — NO regex (Rule 16).

    Parameters
    ----------
    duration_str:
        Duration string from sky jobs queue output.

    Returns
    -------
    Duration in minutes, or None if the input is "-" or empty.
    """
    s = duration_str.strip()
    if not s or s == "-":
        return None

    total_minutes = 0.0
    parts = s.split()
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.endswith("d"):
            total_minutes += float(part[:-1]) * 24 * 60
        elif part.endswith("h"):
            total_minutes += float(part[:-1]) * 60
        elif part.endswith("m"):
            total_minutes += float(part[:-1])
        elif part.endswith("s"):
            total_minutes += float(part[:-1]) / 60.0

    return total_minutes


def parse_jobs_queue(output: str) -> list[QueuedJob]:
    """Parse `sky jobs queue` output into a list of QueuedJob objects.

    Handles header lines, ANSI escape codes, and various output formats.
    Returns empty list for error output or empty input.

    Uses str.split() for column extraction — NO regex (Rule 16).

    Parameters
    ----------
    output:
        Raw stdout from `sky jobs queue`.

    Returns
    -------
    List of parsed jobs, ordered by job ID descending (newest first).
    """
    if not output or not output.strip():
        return []

    jobs: list[QueuedJob] = []
    lines = output.strip().split("\n")

    for line in lines:
        # Strip ANSI escape codes by removing sequences starting with ESC[
        clean = line
        while "\x1b[" in clean:
            start = clean.index("\x1b[")
            # Find the end of the escape sequence (a letter after digits/semicolons)
            end = start + 2
            while end < len(clean) and clean[end] not in "mHJKABCDfsu":
                end += 1
            if end < len(clean):
                end += 1  # include the terminating letter
            clean = clean[:start] + clean[end:]

        clean = clean.strip()
        if not clean:
            continue

        # Skip non-data lines
        tokens = clean.split()
        if not tokens:
            continue

        # First token must be an integer (job ID) for it to be a data line
        try:
            job_id = int(tokens[0])
        except (ValueError, IndexError):
            continue

        # Find the status by looking for a known status token
        status = None
        status_idx = -1
        for i, token in enumerate(tokens):
            if token in KNOWN_STATUSES:
                status = token
                status_idx = i
                break

        if status is None:
            continue

        # Token layout (variable width, space-separated):
        # ID  TASK  NAME  REQUESTED...  SUBMITTED...  TOT.DURATION  JOB.DURATION  #RECOVERIES  STATUS  POOL
        #
        # Strategy: we know job_id (idx 0), status (found above), and POOL (last).
        # Work backwards from status to extract #RECOVERIES and durations.
        # NAME is always at index 2 (after ID and TASK "-").

        # Name is at index 2
        name = tokens[2] if len(tokens) > 2 else ""

        # Recovery count is the token immediately before status
        recovery_count = 0
        if status_idx > 0:
            with contextlib.suppress(ValueError, IndexError):
                recovery_count = int(tokens[status_idx - 1])

        # Job duration is 2 tokens before status (could be multi-word like "1h 22m")
        # Total duration is further before that.
        # Since durations can be multi-word, we need to reconstruct them.
        # The safest approach: everything between NAME and SUBMITTED contains REQUESTED,
        # and everything between SUBMITTED and #RECOVERIES contains durations.

        # Find requested block: tokens between NAME and "ago" keyword
        # "ago" appears in the SUBMITTED column (e.g., "3 mins ago")
        ago_indices = [i for i, t in enumerate(tokens) if t == "ago"]

        total_dur_minutes = None
        job_dur_minutes = None

        if ago_indices:
            last_ago = ago_indices[-1]
            # After "ago" come: TOT.DURATION tokens, JOB.DURATION tokens, #RECOVERIES, STATUS
            # We know status_idx and recovery is status_idx - 1
            # So duration tokens are from last_ago+1 to status_idx-1 (exclusive of recovery)
            dur_tokens = tokens[last_ago + 1 : status_idx - 1]

            # Split duration tokens into total_duration and job_duration
            # They are separated by position — total comes first, then job
            # If job duration is "-", it's a single token
            if dur_tokens:
                # Find where job_duration starts
                # If there's a "-" in the tokens, that's job_duration = None
                if "-" in dur_tokens:
                    dash_idx = dur_tokens.index("-")
                    total_dur_str = " ".join(dur_tokens[:dash_idx])
                    job_dur_minutes = None
                else:
                    # Split roughly in half — total duration comes first
                    # Heuristic: find the boundary where a new duration unit starts
                    # after seeing seconds or minutes
                    mid = len(dur_tokens) // 2
                    if mid == 0:
                        mid = 1
                    total_dur_str = " ".join(dur_tokens[:mid])
                    job_dur_str = " ".join(dur_tokens[mid:])
                    job_dur_minutes = parse_duration_to_minutes(job_dur_str)

                total_dur_minutes = parse_duration_to_minutes(total_dur_str)

        # Build requested string: tokens between name and submitted
        requested = ""
        if ago_indices:
            # Tokens between name (idx 2) and the submitted block
            # Submitted block ends with "ago"
            first_ago = ago_indices[0]
            # Submitted block is typically "N unit ago" (3 tokens before ago)
            submitted_start = max(3, first_ago - 2)
            req_tokens = tokens[3:submitted_start]
            requested = " ".join(req_tokens)

        jobs.append(QueuedJob(
            job_id=job_id,
            name=name,
            requested=requested,
            total_duration_minutes=total_dur_minutes,
            job_duration_minutes=job_dur_minutes,
            recovery_count=recovery_count,
            status=status,
        ))

    return jobs
