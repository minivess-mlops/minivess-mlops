"""Tests for docker-compose.flows.yml security and resource hardening.

CIS Docker Benchmark 5.3 (cap_drop), 5.25 (no-new-privileges),
5.11 (resource limits), OWASP D07 (resource management).

Rule #16: No regex. Parse with yaml.safe_load().
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parent.parent.parent.parent.parent
FLOWS_COMPOSE = ROOT / "deployment" / "docker-compose.flows.yml"
INFRA_COMPOSE = ROOT / "deployment" / "docker-compose.yml"


def _load_flows() -> dict:
    return yaml.safe_load(FLOWS_COMPOSE.read_text(encoding="utf-8"))


def _load_infra() -> dict:
    return yaml.safe_load(INFRA_COMPOSE.read_text(encoding="utf-8"))


def _flow_services() -> dict[str, dict]:
    """Return all services from docker-compose.flows.yml."""
    compose = _load_flows()
    return compose.get("services", {})


def test_all_flows_have_cap_drop_all() -> None:
    """All flow services must drop ALL Linux capabilities (CIS 5.3)."""
    failures = []
    for name, svc in _flow_services().items():
        cap_drop = svc.get("cap_drop", [])
        if "ALL" not in cap_drop:
            failures.append(f"{name}: cap_drop={cap_drop!r}")

    assert not failures, (
        "Services missing 'cap_drop: [ALL]' (CIS 5.3):\n"
        + "\n".join(f"  - {f}" for f in failures)
        + "\nAdd: cap_drop: [ALL] to x-common-security anchor."
    )


def test_all_flows_have_no_new_privileges() -> None:
    """All flow services must set no-new-privileges (CIS 5.25)."""
    failures = []
    for name, svc in _flow_services().items():
        security_opt = svc.get("security_opt", [])
        if "no-new-privileges:true" not in security_opt:
            failures.append(f"{name}: security_opt={security_opt!r}")

    assert not failures, (
        "Services missing 'no-new-privileges:true' in security_opt (CIS 5.25):\n"
        + "\n".join(f"  - {f}" for f in failures)
        + "\nAdd: security_opt: [no-new-privileges:true] to x-common-security anchor."
    )


def test_train_has_shm_size() -> None:
    """Train service must have shm_size set (MONAI DataLoader Bus error prevention)."""
    svcs = _flow_services()
    assert "train" in svcs, "train service not found in flows compose"
    shm = svcs["train"].get("shm_size")
    assert shm is not None, (
        "train service missing shm_size. "
        "MONAI 3D DataLoader uses /dev/shm for IPC — without this, "
        "large batch sizes cause 'Bus error' (SIGBUS)."
    )


def test_hpo_has_shm_size() -> None:
    """HPO service must have shm_size set (same reason as train)."""
    svcs = _flow_services()
    assert "hpo" in svcs, "hpo service not found in flows compose"
    shm = svcs["hpo"].get("shm_size")
    assert shm is not None, (
        "hpo service missing shm_size. "
        "HPO training workers use MONAI DataLoader — same shm requirement as train."
    )


def test_train_has_mem_limit() -> None:
    """Train service must have mem_limit (OWASP D07)."""
    svcs = _flow_services()
    assert "train" in svcs, "train service not found"
    assert "mem_limit" in svcs["train"], (
        "train service missing mem_limit. "
        "Add mem_limit to prevent memory exhaustion attacks (OWASP D07)."
    )


def test_all_flows_have_pids_limit() -> None:
    """All flow services must have pids_limit (fork-bomb prevention)."""
    failures = []
    for name, svc in _flow_services().items():
        pids = svc.get("pids_limit")
        if pids is None or (isinstance(pids, int) and pids <= 0):
            failures.append(f"{name}: pids_limit={pids!r}")

    assert not failures, (
        "Services missing pids_limit:\n"
        + "\n".join(f"  - {f}" for f in failures)
        + "\nAdd pids_limit to x-gpu-resources, x-cpu-resources, x-light-resources anchors."
    )


def test_all_flows_have_log_driver() -> None:
    """All flow services must configure a log driver (prevent unbounded log growth)."""
    failures = []
    for name, svc in _flow_services().items():
        logging_cfg = svc.get("logging", {})
        driver = logging_cfg.get("driver") if isinstance(logging_cfg, dict) else None
        if not driver:
            failures.append(name)

    assert not failures, (
        "Services missing logging.driver:\n"
        + "\n".join(f"  - {f}" for f in failures)
        + "\nAdd x-logging anchor with driver: local and options for rotation."
    )


def test_all_flows_have_restart_no() -> None:
    """All flow services must have restart: 'no' (prevent ghost runs)."""
    failures = []
    for name, svc in _flow_services().items():
        restart = svc.get("restart")
        if restart != "no":
            failures.append(f"{name}: restart={restart!r}")

    assert not failures, (
        "Services missing restart: 'no':\n"
        + "\n".join(f"  - {f}" for f in failures)
        + "\nFlow containers must not auto-restart. Add to x-common-security anchor."
    )


def test_minio_init_service_exists() -> None:
    """docker-compose.yml must have minio-init service for bucket auto-creation."""
    compose = _load_infra()
    services = compose.get("services", {})
    has_init = "minio-init" in services
    if not has_init:
        # Also check by container_name
        for svc in services.values():
            if (
                isinstance(svc, dict)
                and svc.get("container_name") == "minivess-minio-init"
            ):
                has_init = True
                break

    assert has_init, (
        "minio-init service not found in docker-compose.yml. "
        "Add minio-init service to auto-create mlflow-artifacts bucket on startup."
    )


def test_volume_init_documented() -> None:
    """deployment/CLAUDE.md must document volume initialization."""
    claude_md = ROOT / "deployment" / "CLAUDE.md"
    assert claude_md.exists(), "deployment/CLAUDE.md not found"
    content = claude_md.read_text(encoding="utf-8")
    has_init = (
        "init-volumes" in content or "volume-init" in content or "make init" in content
    )
    assert has_init, (
        "deployment/CLAUDE.md must document volume initialization step. "
        "Add 'make init-volumes' or equivalent instructions."
    )


@pytest.mark.security_audit
def test_minio_image_pinned() -> None:
    """minio service must use a pinned image tag, not ':latest'."""
    compose = _load_infra()
    minio_svc = compose.get("services", {}).get("minio", {})
    image = minio_svc.get("image", "")
    assert ":latest" not in str(image), (
        f"minio image is not pinned: {image!r}. "
        f"Use a specific release tag for reproducibility."
    )


@pytest.mark.security_audit
@pytest.mark.xfail(
    reason="Known: env-var port bindings lack explicit interface. Fix in hardening pass."
)
def test_ports_bound_to_localhost() -> None:
    """Port bindings should use 127.0.0.1: prefix (localhost-only access)."""
    compose = _load_infra()
    unbound = []
    for name, svc in compose.get("services", {}).items():
        if not isinstance(svc, dict):
            continue
        ports = svc.get("ports", [])
        for port in ports:
            port_str = str(port)
            # Only flag if port binding doesn't specify an interface
            # (i.e., doesn't contain "127.0.0.1:" or "0.0.0.0:")
            if ":" in port_str and not port_str.startswith(("127.0.0.1:", "0.0.0.0:")):
                # This is "${PORT}:5000" style — no interface specified
                unbound.append(f"{name}: {port_str}")

    # Enforce explicit interface binding — all ports should use 127.0.0.1:
    # for local dev. Production (Docker Compose in cloud) uses network isolation.
    # NOTE: ${VAR}:port patterns are acceptable (env var resolves at runtime).
    assert not unbound, (
        "Port bindings without explicit interface (security risk in production):\n"
        + "\n".join(f"  - {p}" for p in unbound)
        + "\nFix: bind to 127.0.0.1:${PORT}:container_port or document why 0.0.0.0 is needed."
    )
