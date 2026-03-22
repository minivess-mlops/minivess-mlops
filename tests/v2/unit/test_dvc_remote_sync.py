"""DVC remote sync verification — catches tracked-but-not-pushed data.

Validates dvc.yaml/dvc.lock consistency and verifies that SkyPilot setup
scripts only pull DVC paths that have lock entries (= were actually pushed).

This test would have caught the $5 FAILED_SETUP incident on 2026-03-22 where
dvc pull -r gcs failed because data/processed/minivess was tracked but never
pushed to GCS.

See: docs/planning/dvc-test-suite-improvement.xml
See: .claude/metalearning/2026-03-22-dvc-pull-untested-setup-script-failure.md
"""

from __future__ import annotations

from pathlib import Path

import yaml


DVC_YAML = Path("dvc.yaml")
DVC_LOCK = Path("dvc.lock")
SKYPILOT_YAML = Path("deployment/skypilot/train_factorial.yaml")


def _parse_dvc_stages() -> dict:
    """Parse dvc.yaml stages."""
    return yaml.safe_load(DVC_YAML.read_text(encoding="utf-8")).get("stages", {})


def _parse_dvc_lock_stages() -> set[str]:
    """Return set of stage names that have dvc.lock entries."""
    if not DVC_LOCK.exists():
        return set()
    lock = yaml.safe_load(DVC_LOCK.read_text(encoding="utf-8"))
    return set(lock.get("stages", {}).keys())


def _parse_dvc_lock_output_paths() -> set[str]:
    """Return set of output paths that have dvc.lock entries."""
    if not DVC_LOCK.exists():
        return set()
    lock = yaml.safe_load(DVC_LOCK.read_text(encoding="utf-8"))
    paths: set[str] = set()
    for stage in lock.get("stages", {}).values():
        for out in stage.get("outs", []):
            paths.add(out.get("path", ""))
    return paths


# ---------------------------------------------------------------------------
# Module 1: DVC yaml/lock consistency
# ---------------------------------------------------------------------------


class TestDvcYamlLockConsistency:
    """Every unfrozen DVC stage with outputs must have a lock entry."""

    def test_all_unfrozen_stages_have_lock_entries(self) -> None:
        """Unfrozen stages with outputs but no lock = tracked but never pushed."""
        stages = _parse_dvc_stages()
        locked = _parse_dvc_lock_stages()

        issues: list[str] = []
        for stage_name, stage in stages.items():
            is_frozen = stage.get("frozen", False)
            has_outs = bool(stage.get("outs"))
            has_lock = stage_name in locked

            if has_outs and not has_lock and not is_frozen:
                issues.append(
                    f"Stage '{stage_name}' has outputs but no dvc.lock entry "
                    "and is NOT frozen — data won't exist on remote"
                )

        assert not issues, (
            "DVC stages with outputs but no lock entries (tracked, never pushed):\n"
            + "\n".join(f"  - {i}" for i in issues)
        )

    def test_frozen_stages_have_comments(self) -> None:
        """Every frozen stage should explain WHY it's frozen."""
        raw = DVC_YAML.read_text(encoding="utf-8")
        stages = _parse_dvc_stages()

        for stage_name, stage in stages.items():
            if not stage.get("frozen", False):
                continue
            # Check if there's a comment near the frozen: true line
            # Look for "frozen" in the raw YAML near a comment
            lines = raw.splitlines()
            for i, line in enumerate(lines):
                if f"frozen: true" in line:
                    # Check if this line or adjacent lines have a comment
                    context = "\n".join(lines[max(0, i - 1) : i + 2])
                    assert "#" in context, (
                        f"Stage '{stage_name}' is frozen but has no comment "
                        f"explaining why (line {i + 1})"
                    )
                    break

    def test_download_stage_has_lock(self) -> None:
        """The download stage (data/raw/minivess) must have a lock entry."""
        locked = _parse_dvc_lock_stages()
        assert "download" in locked, (
            "download stage has no dvc.lock entry — raw data was never DVC-tracked"
        )

    def test_download_stage_has_nfiles(self) -> None:
        """The download stage lock must have nfiles (data integrity check)."""
        lock = yaml.safe_load(DVC_LOCK.read_text(encoding="utf-8"))
        download = lock.get("stages", {}).get("download", {})
        outs = download.get("outs", [])
        assert len(outs) > 0, "No outputs in download stage lock"
        nfiles = outs[0].get("nfiles", 0)
        assert nfiles >= 300, (
            f"Expected ≥300 files in raw MiniVess data, got {nfiles}"
        )


# ---------------------------------------------------------------------------
# Module 2: SkyPilot setup script DVC path validation
# ---------------------------------------------------------------------------


class TestSkypilotDvcPullPaths:
    """Setup script dvc pull paths must match dvc.lock outputs."""

    def _extract_dvc_pull_paths(self) -> list[str]:
        """Extract paths from dvc pull commands in SkyPilot setup."""
        config = yaml.safe_load(SKYPILOT_YAML.read_text(encoding="utf-8"))
        setup = config.get("setup", "")
        paths: list[str] = []
        for line in setup.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "dvc pull" in stripped:
                parts = stripped.split()
                if "pull" in parts:
                    pull_idx = parts.index("pull")
                    for p in parts[pull_idx + 1 :]:
                        if not p.startswith("-"):
                            paths.append(p)
                            break
        return paths

    def test_dvc_pull_paths_have_lock_entries(self) -> None:
        """Every path in 'dvc pull <path>' must have a corresponding lock entry."""
        pull_paths = self._extract_dvc_pull_paths()
        locked_paths = _parse_dvc_lock_output_paths()

        for pull_path in pull_paths:
            matched = any(
                lo.startswith(pull_path) or pull_path.startswith(lo)
                for lo in locked_paths
            )
            assert matched, (
                f"Setup pulls '{pull_path}' but no matching output in dvc.lock. "
                "This path was never pushed to the DVC remote."
            )

    def test_no_bare_dvc_pull_in_setup(self) -> None:
        """Setup must NOT use bare 'dvc pull -r gcs' without path filter."""
        config = yaml.safe_load(SKYPILOT_YAML.read_text(encoding="utf-8"))
        setup = config.get("setup", "")
        for line in setup.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "dvc pull" in stripped and "-r" in stripped:
                parts = stripped.split()
                if "pull" in parts:
                    pull_idx = parts.index("pull")
                    next_non_flag = [
                        p for p in parts[pull_idx + 1 :] if not p.startswith("-")
                    ]
                    assert len(next_non_flag) > 0, (
                        f"Bare 'dvc pull -r <remote>' found without path filter. "
                        f"This will fail if any tracked output is not pushed. "
                        f"Line: {stripped}"
                    )

    def test_setup_pulls_raw_data(self) -> None:
        """Setup must pull data/raw/minivess (the training data)."""
        pull_paths = self._extract_dvc_pull_paths()
        has_raw = any("data/raw/minivess" in p for p in pull_paths)
        assert has_raw, (
            f"Setup does not pull data/raw/minivess. Pull paths: {pull_paths}"
        )
