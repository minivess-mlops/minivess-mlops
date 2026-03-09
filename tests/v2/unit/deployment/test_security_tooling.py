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


# ── T-01.1: Trivy CI gate (#548) ──────────────────────────────────────────────

PR_READINESS = ROOT / "scripts" / "pr_readiness_check.sh"
WEEKLY_SCAN = ROOT / "scripts" / "weekly_security_scan.sh"


def test_pr_readiness_check_has_trivy_step() -> None:
    """pr_readiness_check.sh must include a numbered trivy step."""
    content = PR_READINESS.read_text(encoding="utf-8")
    lines = content.splitlines()
    # Step lines start with "[N/N]" pattern — not comments, not echoes of other content
    step_lines = [
        line
        for line in lines
        if line.strip().startswith("echo")
        and "[" in line
        and "/" in line
        and "Trivy" in line
    ]
    assert step_lines, (
        "No Trivy step found in pr_readiness_check.sh. "
        'Add: echo "[5/5] Trivy CRITICAL scan..."'
    )


def test_pr_readiness_check_calls_trivy_critical() -> None:
    """pr_readiness_check.sh must invoke make trivy-critical."""
    content = PR_READINESS.read_text(encoding="utf-8")
    assert "trivy-critical" in content or "trivy_critical" in content, (
        "pr_readiness_check.sh must call 'make trivy-critical' to gate on CRITICAL CVEs."
    )


def test_pr_readiness_check_trivy_skips_gracefully_when_no_images() -> None:
    """Trivy step must skip (not fail) when no minivess-* images are built yet."""
    content = PR_READINESS.read_text(encoding="utf-8")
    # Must have a guard checking if images exist before calling trivy
    has_image_guard = (
        "docker images" in content
        or "|| true" in content
        or "|| echo" in content
        or "SKIP" in content
    )
    assert has_image_guard, (
        "Trivy step must skip gracefully if no minivess-* images are built. "
        "Add: if docker images --format '{{.Repository}}' | grep -q '^minivess-'; then ..."
    )


def test_weekly_security_scan_exists() -> None:
    """scripts/weekly_security_scan.sh must exist."""
    assert WEEKLY_SCAN.exists(), (
        "scripts/weekly_security_scan.sh not found. "
        "Create weekly scan script covering all 12 flow images."
    )


def test_weekly_security_scan_covers_all_flow_images() -> None:
    """weekly_security_scan.sh must reference all 12 flow image names (not service keys)."""
    content = WEEKLY_SCAN.read_text(encoding="utf-8")
    required_images = [
        "minivess-train",
        "minivess-data",
        "minivess-analyze",
        "minivess-hpo",
        "minivess-deploy",
        "minivess-dashboard",
        "minivess-acquisition",
        "minivess-annotation",
        "minivess-biostatistics",
        "minivess-post-training",
        "minivess-pipeline",
        "minivess-qa",
    ]
    for image in required_images:
        assert image in content, (
            f"weekly_security_scan.sh missing image reference: {image}. "
            "Must scan all 12 flow images."
        )
