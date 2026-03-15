"""Launch dev GPU environment on RunPod via SkyPilot (no custom Docker image).

Uses RunPod's pre-cached base image + installs deps via uv in setup.
Much faster pod start than the 21.4 GB Docker image approach.

Usage:
    uv run python scripts/launch_dev_runpod.py --model dynunet
    uv run python scripts/launch_dev_runpod.py --model sam3_vanilla --spot
    uv run python scripts/launch_dev_runpod.py --model dynunet --experiment debug_single_model

See: docs/planning/runpod-for-quick-dev-env-use.md
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

_ROOT = Path(__file__).resolve().parent.parent


def _load_env() -> None:
    """Load .env file variables into os.environ."""
    env_file = _ROOT / ".env"
    if not env_file.exists():
        print(
            f"ERROR: {env_file} not found. Copy .env.example -> .env and fill in values."
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
    """Launch dev environment on RunPod."""
    parser = argparse.ArgumentParser(
        description="Launch dev GPU environment on RunPod (no Docker)"
    )
    parser.add_argument(
        "--model", default="dynunet", help="Model family to train (default: dynunet)"
    )
    parser.add_argument(
        "--spot", action="store_true", help="Use spot instances (cheaper, can preempt)"
    )
    parser.add_argument(
        "--experiment", default="", help="Override Hydra experiment config name"
    )
    parser.add_argument(
        "--idle-minutes",
        type=int,
        default=15,
        help="Auto-stop after N minutes idle (default: 15)",
    )
    parser.add_argument(
        "--down",
        action="store_true",
        help="Tear down cluster after job (default: keep alive for sky exec)",
    )
    args = parser.parse_args()

    _load_env()

    import sky

    # Load the task YAML
    yaml_path = str(_ROOT / "deployment" / "skypilot" / "dev_runpod.yaml")
    task = sky.Task.from_yaml(yaml_path)

    # Set model family
    env_updates: dict[str, str] = {"MODEL_FAMILY": args.model}

    # Set experiment override if provided
    if args.experiment:
        env_updates["EXPERIMENT"] = args.experiment

    # Load all env vars from .env file into the task.
    env_file = _ROOT / ".env"
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            env_updates[key] = value

    # dev_runpod.yaml uses file-based MLflow on the Network Volume
    # (/opt/vol/mlruns). Do NOT override MLFLOW_TRACKING_URI with the
    # cloud URI — that causes checkpoint upload retries to block training
    # if the remote server is unreachable.
    # Remove any MLFLOW_TRACKING_* from env_updates that came from .env
    for key in list(env_updates):
        if key.startswith("MLFLOW_TRACKING_") or key == "MLFLOW_CLOUD_URI":
            del env_updates[key]

    task.update_envs(env_updates)

    # Configure spot if requested
    if args.spot:
        new_resources = set()
        for r in task.resources:
            new_r = r.copy(use_spot=True)
            new_resources.add(new_r)
        task.set_resources(new_resources)

    print(f"=== Launching dev environment: {args.model} ===")
    print(f"Spot: {args.spot}")
    print(f"Auto-stop: {args.idle_minutes} min idle")
    print(f"Tear down after: {args.down}")
    print("Image: RunPod default (no custom Docker, deps installed via uv)")
    print()

    # Launch the pod
    request_id = sky.launch(
        task,
        cluster_name="minivess-dev",
        idle_minutes_to_autostop=args.idle_minutes,
        down=args.down,
    )
    print(f"Cluster launched. Request ID: {request_id}")
    print()
    print("=== Next steps ===")
    print("Monitor logs:   uv run sky logs minivess-dev")
    print("SSH into pod:   uv run sky ssh minivess-dev")
    print("Re-run training: uv run sky exec minivess-dev -- 'cd /app && ...")
    print("Stop pod:       uv run sky stop minivess-dev")
    print("Tear down:      uv run sky down minivess-dev -y")
    return 0


if __name__ == "__main__":
    sys.exit(main())
