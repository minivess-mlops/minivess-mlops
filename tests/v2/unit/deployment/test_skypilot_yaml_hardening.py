"""SkyPilot YAML hardening tests — cloud robustness for train_factorial.yaml.

Validates setup block resilience patterns:
- No `|| true` after HuggingFace commands (#946)
- Retry loops for DVC pull and HF downloads (#948)
- Timeout guards on long-running setup commands (#953)
- Explicit job_recovery.strategy: EAGER_NEXT_REGION (#955)
- recover_on_exit_codes: [33, 34] (#956)

These tests run against the YAML file on disk, not a live cluster.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

SKYPILOT_DIR = Path("deployment/skypilot")
FACTORIAL_YAML = SKYPILOT_DIR / "train_factorial.yaml"


def _load_yaml() -> dict:
    """Load the factorial YAML as a dict."""
    return yaml.safe_load(FACTORIAL_YAML.read_text(encoding="utf-8"))


def _load_setup_lines() -> list[str]:
    """Load setup block as individual lines."""
    cfg = _load_yaml()
    return cfg["setup"].splitlines()


# ---------------------------------------------------------------------------
# #946 — No `|| true` after HuggingFace commands
# ---------------------------------------------------------------------------


class TestNoOrTrueAfterHfCommands:
    """HuggingFace download failures must NOT be silenced with `|| true`.

    Issue #946: `|| true` masks download failures, causing training to start
    without pretrained weights and silently produce garbage results.
    """

    def test_no_or_true_after_hf_commands(self) -> None:
        """Lines containing hf_hub_download or huggingface-cli must not have `|| true`."""
        lines = _load_setup_lines()
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Skip comments
            if stripped.startswith("#"):
                continue
            # Check if line contains HF commands
            if "hf_hub_download" in stripped or "huggingface-cli" in stripped:
                # The || true could be on the same line
                assert "|| true" not in stripped, (
                    f"Line {i + 1}: HF command has `|| true` — download failures "
                    f"will be silenced. Line: {stripped}"
                )
                # Also check the next non-empty, non-comment line
                for j in range(i + 1, min(i + 3, len(lines))):
                    next_stripped = lines[j].strip()
                    if not next_stripped or next_stripped.startswith("#"):
                        continue
                    assert not next_stripped.startswith("|| true"), (
                        f"Line {j + 1}: `|| true` follows HF command on line {i + 1} — "
                        f"download failures will be silenced."
                    )
                    break  # Only check the first non-empty line after


# ---------------------------------------------------------------------------
# #953 — Timeout guards on long-running setup commands
# ---------------------------------------------------------------------------


class TestTimeoutGuards:
    """Long-running setup commands must have `timeout` guards.

    Issue #953: Without timeouts, a hung DVC pull or HF download can block
    the setup phase indefinitely, burning cloud credits.
    """

    def test_dvc_pull_has_timeout(self) -> None:
        """DVC pull commands must be preceded by or wrapped with `timeout`."""
        lines = _load_setup_lines()
        found_dvc_pull = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "dvc pull" in stripped:
                found_dvc_pull = True
                # Check if timeout is on this line — may be first token or after
                # a shell keyword like `if` (e.g., `if timeout 600 dvc pull ...`)
                tokens = stripped.split()
                has_timeout = "timeout" in tokens
                # Also check within context of a retry loop — timeout may be
                # on a nearby line
                if not has_timeout:
                    # Search up to 5 lines above for a timeout wrapping this command
                    for j in range(max(0, i - 5), i):
                        prev_stripped = lines[j].strip()
                        if prev_stripped.startswith("#"):
                            continue
                        prev_tokens = prev_stripped.split()
                        if prev_tokens and prev_tokens[0] == "timeout":
                            has_timeout = True
                            break
                        # Also check for inline timeout in compound commands
                        if "timeout" in prev_tokens and "dvc" in prev_stripped:
                            has_timeout = True
                            break
                assert has_timeout, (
                    f"Line {i + 1}: `dvc pull` has no `timeout` guard — "
                    f"a hung pull will burn cloud credits indefinitely. "
                    f"Line: {stripped}"
                )
        assert found_dvc_pull, "No `dvc pull` found in setup block"

    def test_hf_download_has_timeout(self) -> None:
        """HuggingFace download commands must have `timeout` guards."""
        lines = _load_setup_lines()
        found_hf = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "hf_hub_download" in stripped or "huggingface-cli" in stripped:
                found_hf = True
                # Check if timeout is on this line — may be after shell keywords
                tokens = stripped.split()
                has_timeout = "timeout" in tokens
                # Check within 5 lines above
                if not has_timeout:
                    for j in range(max(0, i - 5), i):
                        prev_stripped = lines[j].strip()
                        if prev_stripped.startswith("#"):
                            continue
                        prev_tokens = prev_stripped.split()
                        if prev_tokens and prev_tokens[0] == "timeout":
                            has_timeout = True
                            break
                        if "timeout" in prev_tokens and (
                            "hf_hub_download" in prev_stripped
                            or "huggingface-cli" in prev_stripped
                        ):
                            has_timeout = True
                            break
                assert has_timeout, (
                    f"Line {i + 1}: HF download has no `timeout` guard — "
                    f"a hung download will burn cloud credits indefinitely. "
                    f"Line: {stripped}"
                )
        assert found_hf, "No HuggingFace download command found in setup block"

    def test_mlflow_check_has_timeout(self) -> None:
        """MLflow connectivity check must have a `timeout` guard."""
        lines = _load_setup_lines()
        found_mlflow_check = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "MlflowClient" in stripped:
                found_mlflow_check = True
                # Check if timeout is on this line or enclosing lines
                tokens = stripped.split()
                has_timeout = "timeout" in tokens
                if not has_timeout:
                    for j in range(max(0, i - 5), i):
                        prev_stripped = lines[j].strip()
                        if prev_stripped.startswith("#"):
                            continue
                        prev_tokens = prev_stripped.split()
                        if prev_tokens and prev_tokens[0] == "timeout":
                            has_timeout = True
                            break
                assert has_timeout, (
                    f"Line {i + 1}: MLflow connectivity check has no `timeout` guard — "
                    f"a hung server check will block setup indefinitely. "
                    f"Line: {stripped}"
                )
        assert found_mlflow_check, "No MlflowClient check found in setup block"


# ---------------------------------------------------------------------------
# #948 — Retry loops for DVC pull and HF downloads
# ---------------------------------------------------------------------------


class TestRetryLoops:
    """DVC pull and HF downloads must have retry logic.

    Issue #948: Transient network failures (DNS, GCS throttling, HF rate limits)
    cause FAILED_SETUP. Retry loops with backoff prevent this.
    """

    def test_dvc_pull_has_retry_loop(self) -> None:
        """DVC pull must be wrapped in a retry loop (for attempt in ...)."""
        lines = _load_setup_lines()
        found_dvc_pull = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "dvc pull" in stripped:
                found_dvc_pull = True
                # Search within 10 lines above for retry loop markers
                context_start = max(0, i - 10)
                context_lines = lines[context_start : i + 1]
                context_text = " ".join(ctx.strip() for ctx in context_lines)
                has_retry = "for attempt" in context_text or "retry" in context_text.lower()
                assert has_retry, (
                    f"Line {i + 1}: `dvc pull` has no retry loop — "
                    f"transient GCS failures will crash setup. "
                    f"Wrap with: for attempt in 1 2 3; do ... && break; done"
                )
                break  # Only check the first dvc pull for minivess data
        assert found_dvc_pull, "No `dvc pull` found in setup block"

    def test_hf_download_has_retry_loop(self) -> None:
        """HuggingFace downloads must have retry logic."""
        lines = _load_setup_lines()
        found_hf = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "hf_hub_download" in stripped:
                found_hf = True
                # Search within 10 lines above for retry loop markers
                context_start = max(0, i - 10)
                context_lines = lines[context_start : i + 1]
                context_text = " ".join(ctx.strip() for ctx in context_lines)
                has_retry = "for attempt" in context_text or "retry" in context_text.lower()
                assert has_retry, (
                    f"Line {i + 1}: HF download has no retry loop — "
                    f"transient HF rate limits will crash setup. "
                    f"Wrap with: for attempt in 1 2 3; do ... && break; done"
                )
                break  # Only check the first HF download
        assert found_hf, "No HuggingFace download command found in setup block"


# ---------------------------------------------------------------------------
# #955 — Explicit job_recovery.strategy: EAGER_NEXT_REGION
# ---------------------------------------------------------------------------


class TestJobRecoveryStrategy:
    """job_recovery must explicitly set strategy: EAGER_NEXT_REGION.

    Issue #955: Without explicit strategy, SkyPilot defaults to retrying in the
    same region — if that region is capacity-constrained, all retries fail.
    EAGER_NEXT_REGION tries the next region in the ordered list.
    """

    def test_job_recovery_strategy_explicit(self) -> None:
        """job_recovery must have an explicit strategy field."""
        cfg = _load_yaml()
        job_recovery = cfg.get("resources", {}).get("job_recovery", {})
        assert "strategy" in job_recovery, (
            "job_recovery missing 'strategy' — SkyPilot defaults to same-region retry. "
            "Set strategy: EAGER_NEXT_REGION for multi-region failover."
        )

    def test_job_recovery_strategy_is_eager_next_region(self) -> None:
        """Strategy must be EAGER_NEXT_REGION for multi-region failover."""
        cfg = _load_yaml()
        job_recovery = cfg.get("resources", {}).get("job_recovery", {})
        strategy = job_recovery.get("strategy", "")
        assert strategy == "EAGER_NEXT_REGION", (
            f"job_recovery.strategy must be EAGER_NEXT_REGION, got: {strategy}"
        )


# ---------------------------------------------------------------------------
# #956 — recover_on_exit_codes: [33, 34]
# ---------------------------------------------------------------------------


class TestRecoverOnExitCodes:
    """job_recovery must specify recover_on_exit_codes for known transient errors.

    Issue #956: Exit codes 33 (DVC network error) and 34 (HF download timeout)
    are transient — recovery should be attempted instead of marking as FAILED.
    Exit code 137 (OOM kill) must NOT be in the list — OOM requires a different
    GPU or batch size, not a retry.
    """

    def test_recover_on_exit_codes_is_list(self) -> None:
        """recover_on_exit_codes must be a list."""
        cfg = _load_yaml()
        job_recovery = cfg.get("resources", {}).get("job_recovery", {})
        exit_codes = job_recovery.get("recover_on_exit_codes")
        assert exit_codes is not None, (
            "job_recovery missing 'recover_on_exit_codes' — transient errors "
            "(DVC network, HF timeout) will not trigger recovery."
        )
        assert isinstance(exit_codes, list), (
            f"recover_on_exit_codes must be a list, got: {type(exit_codes).__name__}"
        )

    def test_recover_on_exit_codes_contains_expected(self) -> None:
        """Exit codes 33 and 34 must be in the recovery list."""
        cfg = _load_yaml()
        job_recovery = cfg.get("resources", {}).get("job_recovery", {})
        exit_codes = job_recovery.get("recover_on_exit_codes", [])
        assert 33 in exit_codes, (
            "Exit code 33 (DVC network error) must be in recover_on_exit_codes"
        )
        assert 34 in exit_codes, (
            "Exit code 34 (HF download timeout) must be in recover_on_exit_codes"
        )

    def test_oom_exit_137_not_in_recover(self) -> None:
        """Exit code 137 (OOM kill) must NOT be in recover_on_exit_codes.

        OOM requires a different GPU or smaller batch size — retrying on the
        same config will just OOM again, wasting cloud credits.
        """
        cfg = _load_yaml()
        job_recovery = cfg.get("resources", {}).get("job_recovery", {})
        exit_codes = job_recovery.get("recover_on_exit_codes", [])
        assert 137 not in exit_codes, (
            "Exit code 137 (OOM kill) must NOT be in recover_on_exit_codes — "
            "OOM requires a different GPU or batch size, not a retry."
        )


# ---------------------------------------------------------------------------
# Structural — YAML must remain valid after all changes
# ---------------------------------------------------------------------------


class TestYamlStillValid:
    """The modified YAML must still parse correctly."""

    def test_yaml_still_valid(self) -> None:
        """yaml.safe_load() must succeed after all hardening changes."""
        cfg = _load_yaml()
        assert isinstance(cfg, dict), "YAML did not parse to a dict"
        assert "resources" in cfg, "Missing resources section"
        assert "setup" in cfg, "Missing setup section"
        assert "run" in cfg, "Missing run section"

    def test_skypilot_task_from_yaml(self) -> None:
        """sky.Task.from_yaml() must succeed after all hardening changes."""
        try:
            import sky

            task = sky.Task.from_yaml(str(FACTORIAL_YAML))
            assert task is not None
        except ImportError:
            pytest.skip("skypilot not installed")
