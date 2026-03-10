"""Tests for seccomp enforce profile (SCMP_ACT_ERRNO) — T-02.1 / issue #547.

Verifies:
- train-enforce.json exists and is valid JSON
- defaultAction is SCMP_ACT_ERRNO (not ACT_LOG)
- Syscall allow-list is non-empty (>= 200 names from Docker default)
- No duplicate syscall names
- GPU-specific syscalls are present
- docker-compose.flows.yml GPU services reference train-enforce.json

Rule #16: No regex — json.loads(), yaml.safe_load(), str methods only.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent.parent.parent
SECCOMP_DIR = ROOT / "deployment" / "seccomp"
AUDIT_PROFILE = SECCOMP_DIR / "audit.json"
ENFORCE_PROFILE = SECCOMP_DIR / "train-enforce.json"
COMPOSE_FLOWS = ROOT / "deployment" / "docker-compose.flows.yml"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _all_allowed_names(profile: dict) -> list[str]:
    """Extract all syscall names with SCMP_ACT_ALLOW from a profile."""
    names: list[str] = []
    for entry in profile.get("syscalls", []):
        if entry.get("action") == "SCMP_ACT_ALLOW":
            names.extend(entry.get("names", []))
    return names


def test_train_enforce_profile_exists() -> None:
    assert ENFORCE_PROFILE.exists(), (
        f"Missing: {ENFORCE_PROFILE}. "
        "Create enforce profile derived from Docker moby default + GPU syscalls."
    )


def test_train_enforce_is_valid_json() -> None:
    content = ENFORCE_PROFILE.read_text(encoding="utf-8")
    try:
        json.loads(content)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"train-enforce.json is not valid JSON: {exc}") from exc


def test_train_enforce_default_action_is_errno() -> None:
    profile = _load_json(ENFORCE_PROFILE)
    assert profile.get("defaultAction") == "SCMP_ACT_ERRNO", (
        f"Expected defaultAction=SCMP_ACT_ERRNO, got {profile.get('defaultAction')!r}. "
        "Enforce profile must block unlisted syscalls."
    )


def test_train_enforce_has_large_allow_list() -> None:
    """Docker default seccomp profile allows ~305 syscalls; enforce must be similar."""
    profile = _load_json(ENFORCE_PROFILE)
    names = _all_allowed_names(profile)
    assert len(names) >= 200, (
        f"Expected >= 200 allowed syscalls (Docker default ~305), got {len(names)}. "
        "Derive enforce.json from the Docker moby default seccomp profile."
    )


def test_train_enforce_no_empty_syscall_names() -> None:
    profile = _load_json(ENFORCE_PROFILE)
    names = _all_allowed_names(profile)
    empty = [n for n in names if not n]
    assert not empty, f"Found {len(empty)} empty syscall name(s) in train-enforce.json."


def test_train_enforce_no_duplicate_syscall_names() -> None:
    profile = _load_json(ENFORCE_PROFILE)
    names = _all_allowed_names(profile)
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in names:
        if name in seen:
            duplicates.append(name)
        seen.add(name)
    assert not duplicates, (
        f"Duplicate syscall names in train-enforce.json: {duplicates[:10]}"
    )


def test_train_enforce_includes_gpu_specific_syscalls() -> None:
    """GPU/NUMA syscalls beyond Docker default required for CUDA/PyTorch training."""
    profile = _load_json(ENFORCE_PROFILE)
    names = set(_all_allowed_names(profile))
    gpu_required = ["perf_event_open", "get_mempolicy", "mbind", "set_mempolicy"]
    missing = [sc for sc in gpu_required if sc not in names]
    assert not missing, (
        f"GPU-specific syscalls missing from train-enforce.json: {missing}. "
        "Add perf_event_open, get_mempolicy, mbind, set_mempolicy for CUDA training."
    )


def test_compose_gpu_services_reference_enforce_profile() -> None:
    """train, hpo, hpo-worker must have seccomp security_opt pointing to train-enforce.json."""
    compose = yaml.safe_load(COMPOSE_FLOWS.read_text(encoding="utf-8"))
    gpu_services = ["train", "hpo", "hpo-worker", "post_training", "analyze"]
    for svc_name in gpu_services:
        svc = compose["services"][svc_name]
        security_opts: list[str] = svc.get("security_opt", [])
        seccomp_opts = [o for o in security_opts if "seccomp" in o]
        assert seccomp_opts, (
            f"Service '{svc_name}' missing seccomp in security_opt. "
            'Add: security_opt: ["seccomp=seccomp/train-enforce.json"]'
        )
        assert "train-enforce.json" in seccomp_opts[0], (
            f"Service '{svc_name}' seccomp points to wrong profile: {seccomp_opts[0]}. "
            "Must reference seccomp/train-enforce.json (relative to compose dir)."
        )


def test_compose_seccomp_paths_resolve_from_compose_dir() -> None:
    """Seccomp paths in compose file must resolve relative to the compose file's directory.

    Docker Compose V2 resolves security_opt seccomp= paths relative to
    the compose file's parent directory, NOT the working directory.
    """
    compose = yaml.safe_load(COMPOSE_FLOWS.read_text(encoding="utf-8"))
    compose_dir = COMPOSE_FLOWS.parent

    for svc_name, svc in compose.get("services", {}).items():
        security_opts: list[str] = svc.get("security_opt", [])
        for opt in security_opts:
            if not opt.startswith("seccomp="):
                continue
            seccomp_path = opt.split("=", 1)[1]
            resolved = compose_dir / seccomp_path
            assert resolved.exists(), (
                f"Service '{svc_name}' seccomp path does not resolve: {opt}. "
                f"Resolved to: {resolved}. "
                "Paths are relative to compose file directory (deployment/), "
                "not the repo root."
            )
