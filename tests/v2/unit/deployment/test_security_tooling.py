"""Tests for MLSecOps tooling: Trivy scan integration and seccomp scaffolding.

T-04.1: Trivy scan — Makefile targets, CRITICAL-only policy, --ignore-unfixed.
T-04.2: Seccomp — deployment/seccomp/ dir, audit.json valid JSON, README.

Rule #16: No regex. Use str methods, json.loads(), pathlib.Path.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent.parent
MAKEFILE = ROOT / "Makefile"
SECCOMP_DIR = ROOT / "deployment" / "seccomp"
AUDIT_PROFILE = SECCOMP_DIR / "audit.json"
SECCOMP_README = SECCOMP_DIR / "README.md"


# ── T-04.1: Trivy scan Makefile targets ──────────────────────────────────────


def _makefile_content() -> str:
    return MAKEFILE.read_text(encoding="utf-8")


def test_makefile_has_scan_target() -> None:
    """Makefile must have a scan: target (Trivy vulnerability scan)."""
    assert "scan:" in _makefile_content(), (
        "Makefile missing 'scan:' target. "
        "Add Trivy scan target for MLSecOps image scanning."
    )


def test_makefile_has_sbom_target() -> None:
    """Makefile must have an sbom: target (CycloneDX SBOM generation)."""
    assert "sbom:" in _makefile_content(), (
        "Makefile missing 'sbom:' target. "
        "Add CycloneDX SBOM generation target for supply-chain transparency."
    )


def test_trivy_policy_uses_ignore_unfixed() -> None:
    """Trivy scan must use --ignore-unfixed (NVIDIA base images have many unfixable CVEs)."""
    assert "--ignore-unfixed" in _makefile_content(), (
        "Trivy scan missing --ignore-unfixed flag. "
        "NVIDIA CUDA base images carry many HIGH CVEs with no upstream fix — "
        "blocking on them prevents any scan from succeeding."
    )


def test_trivy_policy_blocks_on_critical() -> None:
    """Trivy scan must include CRITICAL in --severity (CVE-2025-23266 = CVSS 9.0)."""
    content = _makefile_content()
    has_severity_critical = "CRITICAL" in content and "--severity" in content
    assert has_severity_critical, (
        "Trivy scan missing '--severity CRITICAL'. "
        "CVSS 9.0+ vulnerabilities like CVE-2025-23266 must trigger scan failure."
    )


# ── T-04.2: Seccomp scaffolding ───────────────────────────────────────────────


def test_seccomp_profiles_dir_exists() -> None:
    """deployment/seccomp/ directory must exist for seccomp profiles."""
    assert SECCOMP_DIR.exists() and SECCOMP_DIR.is_dir(), (
        "deployment/seccomp/ directory not found. "
        "Create it to hold seccomp profiles for container hardening."
    )


def test_seccomp_audit_profile_exists() -> None:
    """deployment/seccomp/audit.json must exist (syscall audit profile)."""
    assert AUDIT_PROFILE.exists(), (
        "deployment/seccomp/audit.json not found. "
        "Create audit profile with defaultAction: SCMP_ACT_LOG to discover "
        "required syscalls before building allowlist profile."
    )


def test_seccomp_audit_profile_is_valid_json() -> None:
    """audit.json must be valid JSON (parsed with json.loads, not regex)."""
    content = AUDIT_PROFILE.read_text(encoding="utf-8")
    try:
        json.loads(content)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"deployment/seccomp/audit.json is not valid JSON: {exc}"
        ) from exc


def test_seccomp_audit_profile_action_is_log() -> None:
    """audit.json defaultAction must be SCMP_ACT_LOG (log, don't block)."""
    content = AUDIT_PROFILE.read_text(encoding="utf-8")
    parsed = json.loads(content)
    assert parsed.get("defaultAction") == "SCMP_ACT_LOG", (
        f"audit.json defaultAction={parsed.get('defaultAction')!r}, expected 'SCMP_ACT_LOG'. "
        "Audit profile must log all syscalls without blocking — use in staging to "
        "discover required syscalls before building a restrictive allowlist."
    )


def test_seccomp_readme_exists() -> None:
    """deployment/seccomp/README.md must exist (workflow documentation)."""
    assert SECCOMP_README.exists(), (
        "deployment/seccomp/README.md not found. "
        "Add README documenting: how to run with audit profile, extract syscalls, "
        "build allowlist, IPC_LOCK caveat for MONAI pin_memory."
    )
