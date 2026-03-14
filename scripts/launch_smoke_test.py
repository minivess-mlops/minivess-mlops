"""Launch GPU smoke test on RunPod via SkyPilot Python API.

Handles private GHCR registry authentication that the CLI YAML cannot.

Usage:
    uv run python scripts/launch_smoke_test.py --model sam3_vanilla
    uv run python scripts/launch_smoke_test.py --model sam3_vanilla --spot

See: docs/planning/ralph-loop-for-cloud-monitoring.md
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path

# Force IPv4 — Python tries IPv6 first but many networks have broken IPv6,
# causing "Network is unreachable" errors when connecting to RunPod/GHCR APIs.
_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_getaddrinfo(*args, **kwargs):
    return [r for r in _orig_getaddrinfo(*args, **kwargs) if r[0] == socket.AF_INET]


socket.getaddrinfo = _ipv4_getaddrinfo

# Ensure .env is loaded
_ROOT = Path(__file__).resolve().parent.parent


def _load_env() -> None:
    """Load .env file variables into os.environ."""
    env_file = _ROOT / ".env"
    if not env_file.exists():
        print(
            f"ERROR: {env_file} not found. Copy .env.example → .env and fill in values."
        )
        sys.exit(1)
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            os.environ.setdefault(key, value)


def main() -> int:
    """Launch smoke test with private GHCR auth."""
    parser = argparse.ArgumentParser(description="Launch GPU smoke test on RunPod")
    parser.add_argument("--model", default="sam3_vanilla", help="Model family to test")
    parser.add_argument(
        "--spot", action="store_true", help="Use spot instances (cheaper)"
    )
    args = parser.parse_args()

    _load_env()

    # Validate required env vars
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        print("ERROR: GITHUB_TOKEN not set in .env (needed for GHCR pull)")
        return 1

    import sky
    from sky.provision.docker_utils import DockerLoginConfig

    # Load the task YAML
    yaml_path = str(_ROOT / "deployment" / "skypilot" / "smoke_test_gpu.yaml")
    task = sky.Task.from_yaml(yaml_path)

    # Set model family env var
    task.update_envs({"MODEL_FAMILY": args.model})

    # Load all env vars from .env file into the task
    env_file = _ROOT / ".env"
    env_vars = {}
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            env_vars[key] = value
    task.update_envs(env_vars)

    # Set private GHCR registry credentials
    docker_config = DockerLoginConfig(
        username=os.environ.get("GHCR_USERNAME", "petteriteikari"),
        password=github_token,
        server="ghcr.io",
    )

    # Apply docker_login_config to all resource alternatives
    new_resources = set()
    for r in task.resources:
        new_r = r.copy(
            use_spot=args.spot,
            _docker_login_config=docker_config,
        )
        new_resources.add(new_r)
    task.set_resources(new_resources)

    print(f"=== Launching smoke test: {args.model} ===")
    print(f"Spot: {args.spot}")
    print(
        "Docker image: ghcr.io/petteriteikari/minivess-base:latest (private, auth via GITHUB_TOKEN)"
    )

    # Launch directly (not managed job) — sky.launch() provisions a cluster
    # and runs the task. sky.jobs.launch() delegates to a controller VM which
    # doesn't forward _docker_login_config correctly.
    request_id = sky.launch(
        task,
        cluster_name="minivess-smoke-test",
        idle_minutes_to_autostop=5,
        down=True,  # Tear down cluster after job completes
    )
    print(f"Cluster launched. Request ID: {request_id}")
    print("Monitor: uv run sky logs minivess-smoke-test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
