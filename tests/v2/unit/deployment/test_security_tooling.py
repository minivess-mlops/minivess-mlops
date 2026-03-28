"""Tests for MLSecOps tooling: Grype scan integration and seccomp scaffolding.

T-04.1: Grype scan — Makefile targets, CRITICAL-only policy, --ignore-unfixed.
T-04.2: Seccomp — deployment/seccomp/ dir, audit.json valid JSON, README.

Trivy was replaced by Grype after Trivy's supply chain compromise (2026-03-19).
Grype is Anchore's open-source vulnerability scanner (Apache-2.0).

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


# ── T-04.1: Grype scan Makefile targets ──────────────────────────────────────


def _makefile_content() -> str:
    return MAKEFILE.read_text(encoding="utf-8")


def test_makefile_has_scan_target() -> None:
    """Makefile must have a scan: target (Grype vulnerability scan)."""
    assert "scan:" in _makefile_content(), (
        "Makefile missing 'scan:' target. "
        "Add Grype scan target for MLSecOps image scanning."
    )


def test_makefile_has_sbom_target() -> None:
    """Makefile must have an sbom: target (CycloneDX SBOM generation)."""
    assert "sbom:" in _makefile_content(), (
        "Makefile missing 'sbom:' target. "
        "Add CycloneDX SBOM generation target for supply-chain transparency."
    )


def test_makefile_has_install_grype_target() -> None:
    """Makefile must have an install-grype: target (Grype installer)."""
    assert "install-grype:" in _makefile_content(), (
        "Makefile missing 'install-grype:' target. "
        "Add Grype install target replacing the old install-trivy target."
    )


def test_makefile_has_grype_critical_target() -> None:
    """Makefile must have a grype-critical: target (CRITICAL-only gate for PR readiness)."""
    assert "grype-critical:" in _makefile_content(), (
        "Makefile missing 'grype-critical:' target. "
        "Add CRITICAL-only scan target for PR readiness gating."
    )


def test_grype_version_pinned() -> None:
    """install-grype target must pin Grype to v0.90.0."""
    content = _makefile_content()
    assert "v0.90.0" in content, (
        "Grype version not pinned to v0.90.0 in Makefile. "
        "Pin to a specific version for reproducible security scanning."
    )


def test_scan_target_uses_grype_not_trivy() -> None:
    """scan: target must use grype, not trivy (Trivy supply chain compromised 2026-03-19)."""
    content = _makefile_content()
    # Find the scan: target lines — look for "grype" in lines after "scan:"
    lines = content.splitlines()
    in_scan_target = False
    scan_lines: list[str] = []
    for line in lines:
        if line.startswith("scan:"):
            in_scan_target = True
            scan_lines.append(line)
            continue
        if in_scan_target:
            if line.startswith("\t") or line.startswith(" ") or line.strip() == "":
                scan_lines.append(line)
            else:
                break
    scan_block = "\n".join(scan_lines)
    assert "grype" in scan_block, (
        "scan: target does not use grype. "
        "Replace trivy with grype for Docker image vulnerability scanning."
    )
    assert "trivy" not in scan_block, (
        "scan: target still references trivy. "
        "Trivy supply chain was compromised 2026-03-19 — use grype instead."
    )


def test_grype_scan_uses_ignore_unfixed() -> None:
    """Grype scan must use --only-fixed (equivalent of Trivy's --ignore-unfixed)."""
    content = _makefile_content()
    has_only_fixed = "--only-fixed" in content
    has_ignore_unfixed = "--ignore-unfixed" in content
    assert has_only_fixed or has_ignore_unfixed, (
        "Grype scan missing --only-fixed flag. "
        "NVIDIA CUDA base images carry many HIGH CVEs with no upstream fix — "
        "blocking on them prevents any scan from succeeding."
    )


def test_grype_policy_blocks_on_critical() -> None:
    """Grype scan must include severity filtering for CRITICAL CVEs."""
    content = _makefile_content()
    has_fail_on_critical = "--fail-on critical" in content
    has_severity_critical = "CRITICAL" in content and ("--severity" in content or "--fail-on" in content)
    assert has_fail_on_critical or has_severity_critical, (
        "Grype scan missing CRITICAL severity gating. "
        "CVSS 9.0+ vulnerabilities like CVE-2025-23266 must trigger scan failure."
    )


def test_makefile_no_trivy_references() -> None:
    """Makefile must not reference trivy anywhere (supply chain compromised 2026-03-19)."""
    content = _makefile_content()
    # Check that no executable lines reference trivy
    lines = content.splitlines()
    trivy_lines = [
        line.strip()
        for line in lines
        if "trivy" in line.lower()
        and not line.strip().startswith("#")
    ]
    assert not trivy_lines, (
        f"Makefile still references trivy in {len(trivy_lines)} non-comment line(s). "
        "Trivy supply chain was compromised 2026-03-19 — replace all references with grype."
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


# ── T-01.1: Grype CI gate ────────────────────────────────────────────────────

PR_READINESS = ROOT / "scripts" / "pr_readiness_check.sh"
WEEKLY_SCAN = ROOT / "scripts" / "weekly_security_scan.sh"


def test_pr_readiness_check_has_grype_step() -> None:
    """pr_readiness_check.sh must include a numbered Grype step."""
    content = PR_READINESS.read_text(encoding="utf-8")
    lines = content.splitlines()
    # Step lines start with "[N/N]" pattern — not comments, not echoes of other content
    step_lines = [
        line
        for line in lines
        if line.strip().startswith("echo")
        and "[" in line
        and "/" in line
        and "Grype" in line
    ]
    assert step_lines, (
        "No Grype step found in pr_readiness_check.sh. "
        'Add: echo "[5/5] Grype CRITICAL scan..."'
    )


def test_pr_readiness_calls_grype_critical() -> None:
    """pr_readiness_check.sh must invoke make grype-critical."""
    content = PR_READINESS.read_text(encoding="utf-8")
    assert "grype-critical" in content, (
        "pr_readiness_check.sh must call 'make grype-critical' to gate on CRITICAL CVEs."
    )


def test_pr_readiness_check_grype_skips_gracefully_when_no_images() -> None:
    """Grype step must skip (not fail) when no minivess-* images are built yet."""
    content = PR_READINESS.read_text(encoding="utf-8")
    # Must have a guard checking if images exist before calling grype
    has_image_guard = (
        "docker images" in content
        or "|| true" in content
        or "|| echo" in content
        or "SKIP" in content
    )
    assert has_image_guard, (
        "Grype step must skip gracefully if no minivess-* images are built. "
        "Add: if docker images --format '{{.Repository}}' | grep -q '^minivess-'; then ..."
    )


def test_pr_readiness_no_trivy_references() -> None:
    """pr_readiness_check.sh must not reference trivy (supply chain compromised)."""
    content = PR_READINESS.read_text(encoding="utf-8")
    lines = content.splitlines()
    trivy_lines = [
        line.strip()
        for line in lines
        if "trivy" in line.lower()
        and not line.strip().startswith("#")
    ]
    assert not trivy_lines, (
        f"pr_readiness_check.sh still references trivy in {len(trivy_lines)} non-comment line(s). "
        "Trivy supply chain was compromised 2026-03-19 — replace all references with grype."
    )


def test_weekly_security_scan_exists() -> None:
    """scripts/weekly_security_scan.sh must exist."""
    assert WEEKLY_SCAN.exists(), (
        "scripts/weekly_security_scan.sh not found. "
        "Create weekly scan script covering all flow images."
    )


def test_weekly_scan_uses_grype() -> None:
    """weekly_security_scan.sh must use grype, not trivy."""
    content = WEEKLY_SCAN.read_text(encoding="utf-8")
    lines = content.splitlines()
    # Check executable lines (non-comment) for scanner tool references
    executable_lines = [
        line for line in lines if line.strip() and not line.strip().startswith("#")
    ]
    executable_block = "\n".join(executable_lines)
    assert "grype" in executable_block, (
        "weekly_security_scan.sh does not use grype in executable lines. "
        "Replace trivy with grype for vulnerability scanning."
    )
    trivy_exec_lines = [
        line.strip()
        for line in executable_lines
        if "trivy" in line.lower()
    ]
    assert not trivy_exec_lines, (
        f"weekly_security_scan.sh still uses trivy in {len(trivy_exec_lines)} executable line(s). "
        "Trivy supply chain was compromised 2026-03-19 — use grype instead."
    )


def test_weekly_security_scan_covers_all_flow_images() -> None:
    """weekly_security_scan.sh must reference all 11 flow image names (not service keys)."""
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
    ]
    for image in required_images:
        assert image in content, (
            f"weekly_security_scan.sh missing image reference: {image}. "
            "Must scan all 11 flow images."
        )
