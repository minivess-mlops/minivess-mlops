"""Tests for SkyPilot job queue batch parser.

RED phase: these tests define the contract for parse_jobs_queue()
before any implementation exists.

Plan: experiment-harness-improvement-plan.xml Task T1.2
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Fixtures: actual sky jobs queue output from 11th pass
# ---------------------------------------------------------------------------

QUEUE_OUTPUT_EMPTY = ""

QUEUE_OUTPUT_HEADER_ONLY = """Fetching managed job statuses...
Managed jobs
No in-progress managed jobs."""

QUEUE_OUTPUT_SINGLE_RUNNING = """Fetching managed job statuses...
Managed jobs
In progress tasks: 1 RUNNING
ID   TASK  NAME                                           REQUESTED               SUBMITTED    TOT. DURATION  JOB DURATION  #RECOVERIES  STATUS        POOL
165  -     dynunet-dice_ce-calibfalse-f0                  1x[A100-80GB:1, L4:1][Spot]  3 mins ago   3m 20s         1m 5s         0            RUNNING       -"""

QUEUE_OUTPUT_MIXED_STATUS = """Fetching managed job statuses...
Managed jobs
In progress tasks: 2 PENDING, 1 RUNNING
ID   TASK  NAME                                           REQUESTED                    SUBMITTED    TOT. DURATION  JOB DURATION  #RECOVERIES  STATUS        POOL
167  -     sam3_vanilla-zeroshot-minivess-f0              1x[A100-80GB:1, L4:1][Spot]  1 min ago    1m 12s         -             0            PENDING       -
166  -     sam3_hybrid-cbdice_cldice-calibfalse-f0        1x[A100-80GB:1, L4:1]        3 mins ago   3m 5s          -             0            STARTING      -
165  -     dynunet-dice_ce-calibfalse-f0                  1x[A100-80GB:1, L4:1][Spot]  15 mins ago  15m 20s        8m 5s         0            RUNNING       -
164  -     sam3_vanilla-zeroshot-minivess-f0              1x[L4:1][Spot]               12 hrs ago   11h 32m 57s    -             0            CANCELLED     -
163  -     sam3_hybrid-cbdice_cldice-calibfalse-f0        1x[L4:1]                     12 hrs ago   11h 35m 18s    -             0            CANCELLED     -
159  -     dynunet-dice_ce-calibfalse-f0                  1x[L4:1][Spot]               12 hrs ago   11h 47m 30s    -             0            CANCELLED     -"""

QUEUE_OUTPUT_FAILED_SETUP = """Fetching managed job statuses...
Managed jobs
ID   TASK  NAME                                           REQUESTED       SUBMITTED   TOT. DURATION  JOB DURATION  #RECOVERIES  STATUS        POOL
139  -     vesselfm-zeroshot-deepvess-f0                  1x[L4:1][Spot]  2 days ago  22h 43m 41s    10m 44s       3            FAILED_SETUP  -
138  -     sam3_vanilla-zeroshot-minivess-f0              1x[L4:1][Spot]  2 days ago  20h 1m 8s      10m 58s       3            FAILED_SETUP  -"""

QUEUE_OUTPUT_WITH_SUCCEEDED = """Fetching managed job statuses...
Managed jobs
ID   TASK  NAME                                           REQUESTED               SUBMITTED    TOT. DURATION  JOB DURATION  #RECOVERIES  STATUS     POOL
165  -     dynunet-dice_ce-calibfalse-f0                  1x[A100-80GB:1, L4:1][Spot]  25 mins ago  25m 20s        12m 5s        0            SUCCEEDED  -
118  -     mambavesselnet-dice_ce-calibfalse-f0           1x[L4:1][Spot]               2 days ago   22h 34m 41s    13m 53s       3            FAILED     -"""

QUEUE_OUTPUT_SKY_ERROR = """sky.exceptions.ApiServerConnectionError: Could not connect to SkyPilot API server at http://127.0.0.1:46580."""

QUEUE_OUTPUT_11TH_PASS_PENDING = """Fetching managed job statuses...
Managed jobs
In progress tasks: 3 PENDING
ID   TASK  NAME                                           REQUESTED               SUBMITTED   TOT. DURATION  JOB DURATION  #RECOVERIES  STATUS        POOL
161  -     sam3_vanilla-zeroshot-minivess-f0              1x[L4:1][Spot]          10 hrs ago  10h 2m 17s     -             0            PENDING       -
160  -     sam3_hybrid-cbdice_cldice-calibfalse-f0        1x[L4:1]                10 hrs ago  10h 4m 37s     -             0            PENDING       -
159  -     dynunet-dice_ce-calibfalse-f0                  1x[L4:1][Spot]          10 hrs ago  10h 16m 49s    -             0            PENDING       -"""

QUEUE_OUTPUT_RECOVERING = """Fetching managed job statuses...
Managed jobs
In progress tasks: 1 RECOVERING
ID   TASK  NAME                                           REQUESTED       SUBMITTED   TOT. DURATION  JOB DURATION  #RECOVERIES  STATUS      POOL
157  -     sam3_topolora-dice_ce_cldice-calibfalse-f0     1x[L4:1][Spot]  2 days ago  1d 4h 55m 45s  4h 54m 54s    2            RECOVERING  -"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestParseJobsQueue:
    """parse_jobs_queue extracts structured job data from sky jobs queue output."""

    def test_parse_queue_empty_output(self) -> None:
        """Empty output returns empty list."""
        from minivess.compute.sky_queue_parser import parse_jobs_queue

        assert parse_jobs_queue("") == []

    def test_parse_queue_header_only(self) -> None:
        """Header-only output (no jobs) returns empty list."""
        from minivess.compute.sky_queue_parser import parse_jobs_queue

        assert parse_jobs_queue(QUEUE_OUTPUT_HEADER_ONLY) == []

    def test_parse_queue_single_running_job(self) -> None:
        """Single RUNNING job parsed with all fields."""
        from minivess.compute.sky_queue_parser import parse_jobs_queue

        jobs = parse_jobs_queue(QUEUE_OUTPUT_SINGLE_RUNNING)
        assert len(jobs) == 1
        job = jobs[0]
        assert job.job_id == 165
        assert job.name == "dynunet-dice_ce-calibfalse-f0"
        assert job.status == "RUNNING"
        assert job.recovery_count == 0

    def test_parse_queue_multiple_jobs_mixed_status(self) -> None:
        """Multiple jobs with various statuses parsed correctly."""
        from minivess.compute.sky_queue_parser import parse_jobs_queue

        jobs = parse_jobs_queue(QUEUE_OUTPUT_MIXED_STATUS)
        assert len(jobs) == 6
        statuses = {j.name: j.status for j in jobs if j.job_id >= 165}
        assert statuses["dynunet-dice_ce-calibfalse-f0"] == "RUNNING"
        assert statuses["sam3_hybrid-cbdice_cldice-calibfalse-f0"] == "STARTING"
        assert statuses["sam3_vanilla-zeroshot-minivess-f0"] == "PENDING"

    def test_parse_queue_pending_job_no_job_duration(self) -> None:
        """PENDING job has job_duration_minutes = None."""
        from minivess.compute.sky_queue_parser import parse_jobs_queue

        jobs = parse_jobs_queue(QUEUE_OUTPUT_11TH_PASS_PENDING)
        for job in jobs:
            assert job.job_duration_minutes is None

    def test_parse_queue_total_duration_parsing(self) -> None:
        """Duration strings parsed correctly to minutes."""
        from minivess.compute.sky_queue_parser import parse_duration_to_minutes

        assert parse_duration_to_minutes("1h 22m") == 82.0
        assert parse_duration_to_minutes("45m") == 45.0
        assert parse_duration_to_minutes("2h") == 120.0
        assert parse_duration_to_minutes("5m") == 5.0
        assert parse_duration_to_minutes("10h 2m 17s") >= 602.0
        assert parse_duration_to_minutes("1d 4h 55m 45s") >= 1735.0
        assert parse_duration_to_minutes("-") is None
        assert parse_duration_to_minutes("3m 20s") >= 3.3

    def test_parse_queue_recovery_count(self) -> None:
        """#RECOVERIES column parsed as integer."""
        from minivess.compute.sky_queue_parser import parse_jobs_queue

        jobs = parse_jobs_queue(QUEUE_OUTPUT_FAILED_SETUP)
        for job in jobs:
            assert job.recovery_count == 3

    def test_parse_queue_name_matches_condition(self) -> None:
        """NAME field extracted correctly."""
        from minivess.compute.sky_queue_parser import parse_jobs_queue

        jobs = parse_jobs_queue(QUEUE_OUTPUT_FAILED_SETUP)
        names = {j.name for j in jobs}
        assert "vesselfm-zeroshot-deepvess-f0" in names
        assert "sam3_vanilla-zeroshot-minivess-f0" in names

    def test_parse_queue_failed_setup_status(self) -> None:
        """FAILED_SETUP is a distinct status from FAILED."""
        from minivess.compute.sky_queue_parser import parse_jobs_queue

        jobs = parse_jobs_queue(QUEUE_OUTPUT_FAILED_SETUP)
        assert all(j.status == "FAILED_SETUP" for j in jobs)

    def test_parse_queue_recovering_status(self) -> None:
        """RECOVERING status is recognized."""
        from minivess.compute.sky_queue_parser import parse_jobs_queue

        jobs = parse_jobs_queue(QUEUE_OUTPUT_RECOVERING)
        assert len(jobs) == 1
        assert jobs[0].status == "RECOVERING"

    def test_parse_queue_sky_cli_error_output(self) -> None:
        """Error text returns empty list, does not crash."""
        from minivess.compute.sky_queue_parser import parse_jobs_queue

        jobs = parse_jobs_queue(QUEUE_OUTPUT_SKY_ERROR)
        assert jobs == []

    def test_parse_queue_succeeded_job(self) -> None:
        """SUCCEEDED job parsed with job duration."""
        from minivess.compute.sky_queue_parser import parse_jobs_queue

        jobs = parse_jobs_queue(QUEUE_OUTPUT_WITH_SUCCEEDED)
        succeeded = [j for j in jobs if j.status == "SUCCEEDED"]
        assert len(succeeded) == 1
        assert succeeded[0].job_id == 165
        assert succeeded[0].job_duration_minutes is not None
        assert succeeded[0].job_duration_minutes > 12.0

    def test_parse_queue_cancelling_status(self) -> None:
        """CANCELLING is a recognized status."""
        from minivess.compute.sky_queue_parser import parse_jobs_queue

        output = """Fetching managed job statuses...
Managed jobs
ID   TASK  NAME                                           REQUESTED       SUBMITTED    TOT. DURATION  JOB DURATION  #RECOVERIES  STATUS      POOL
170  -     sam3_hybrid-cbdice_cldice-calibfalse-f0        1x[L4:1][Spot]  5 mins ago   5m 20s         3m 5s         0            CANCELLING  -"""
        jobs = parse_jobs_queue(output)
        assert len(jobs) == 1
        assert jobs[0].status == "CANCELLING"

    def test_parse_queue_day_scale_duration(self) -> None:
        """Duration '1d 4h 55m 45s' parsed correctly."""
        from minivess.compute.sky_queue_parser import parse_duration_to_minutes

        result = parse_duration_to_minutes("1d 4h 55m 45s")
        assert result is not None
        # 1d = 1440m, 4h = 240m, 55m, 45s = 0.75m = total 1735.75
        assert abs(result - 1735.75) < 1.0

    def test_parse_queue_realistic_11th_pass_output(self) -> None:
        """Full realistic 11th pass output parses correctly."""
        from minivess.compute.sky_queue_parser import parse_jobs_queue

        jobs = parse_jobs_queue(QUEUE_OUTPUT_11TH_PASS_PENDING)
        assert len(jobs) == 3
        assert all(j.status == "PENDING" for j in jobs)
        ids = {j.job_id for j in jobs}
        assert ids == {159, 160, 161}
