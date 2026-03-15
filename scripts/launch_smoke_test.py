"""Launch GPU smoke test on cloud GPU via SkyPilot Python API.

Handles private GHCR registry authentication that the CLI YAML cannot.
Supports Lambda Labs, RunPod, and GCP cloud providers.

Multi-region rotation (Lambda): automatically tries all 17 Lambda regions,
starting from unpopular EU/Asia regions where capacity is more likely available.
3 retries per region before moving to the next. Zero manual work required.

GCP: Uses GAR (same-region as GCS/Cloud SQL/MLflow in europe-north1).
No GHCR auth needed — GAR image is public.

Usage:
    uv run python scripts/launch_smoke_test.py --model sam3_vanilla --cloud lambda
    uv run python scripts/launch_smoke_test.py --model sam3_vanilla --cloud runpod
    uv run python scripts/launch_smoke_test.py --model sam3_vanilla --cloud gcp
    uv run python scripts/launch_smoke_test.py --model sam3_vanilla --cloud lambda --spot

See: docs/runpod-vs-lambda-vs-rest-for-skypilot-with-docker.md
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

# Force IPv4 — Python tries IPv6 first but many networks have broken IPv6,
# causing "Network is unreachable" errors when connecting to RunPod/GHCR APIs.
_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_getaddrinfo(*args, **kwargs):
    return [r for r in _orig_getaddrinfo(*args, **kwargs) if r[0] == socket.AF_INET]


socket.getaddrinfo = _ipv4_getaddrinfo

# Ensure .env is loaded
_ROOT = Path(__file__).resolve().parent.parent

# Lambda regions ordered by expected availability (unpopular first).
# EU/Asia/ME regions have less competition for GPU capacity.
# US regions are most popular (us-east-1 = most popular, tried last).
LAMBDA_REGIONS_BY_PRIORITY = [
    # Tier 1: EU + ME (least popular, highest chance of availability)
    "europe-south-1",
    "europe-central-1",
    "me-west-1",
    # Tier 2: Asia-Pacific (moderately unpopular)
    "asia-south-1",
    "asia-northeast-2",
    "asia-northeast-1",
    "australia-east-1",
    # Tier 3: US peripheral (less popular US regions)
    "us-midwest-1",
    "us-south-3",
    "us-south-2",
    "us-south-1",
    # Tier 4: US west (moderately popular)
    "us-west-3",
    "us-west-2",
    "us-west-1",
    # Tier 5: US east (most popular, tried last)
    "us-east-3",
    "us-east-2",
    "us-east-1",
]

# GPU types ordered by price (cheapest first).
LAMBDA_GPU_PRIORITY = ["A10", "A100", "GH200", "H100"]


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


def _load_env_vars() -> dict[str, str]:
    """Load .env file into a dict for SkyPilot task env injection."""
    env_file = _ROOT / ".env"
    env_vars: dict[str, str] = {}
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            env_vars[key] = value
    return env_vars


def _resolve_mlflow_env(env_vars: dict[str, str]) -> dict[str, str]:
    """Resolve MLflow env vars (SkyPilot can't resolve inter-env refs)."""
    mlflow_uri = env_vars.get(
        "MLFLOW_CLOUD_URI", os.environ.get("MLFLOW_CLOUD_URI", "")
    )
    if mlflow_uri:
        env_vars["MLFLOW_TRACKING_URI"] = mlflow_uri
    mlflow_user = env_vars.get(
        "MLFLOW_CLOUD_USERNAME", os.environ.get("MLFLOW_CLOUD_USERNAME", "admin")
    )
    env_vars["MLFLOW_TRACKING_USERNAME"] = mlflow_user
    mlflow_pass = env_vars.get(
        "MLFLOW_CLOUD_PASSWORD", os.environ.get("MLFLOW_CLOUD_PASSWORD", "")
    )
    if mlflow_pass:
        env_vars["MLFLOW_TRACKING_PASSWORD"] = mlflow_pass
    return env_vars


def _check_lambda_availability() -> dict[str, list[str]]:
    """Query Lambda API for real-time GPU availability per region.

    Returns: dict mapping region_name -> list of available instance types.
    """
    api_key = os.environ.get("LAMBDA_API_KEY", "")
    if not api_key:
        # Try reading from lambda_keys file
        keys_file = Path.home() / ".lambda_cloud" / "lambda_keys"
        if keys_file.exists():
            for line in keys_file.read_text(encoding="utf-8").splitlines():
                if "api_key" in line and "=" in line:
                    api_key = line.split("=", 1)[1].strip()

    if not api_key:
        print("WARNING: Cannot check Lambda availability (no API key)")
        return {}

    try:
        req = urllib.request.Request(
            "https://cloud.lambdalabs.com/api/v1/instance-types",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read())

        availability: dict[str, list[str]] = {}
        for name, info in data.get("data", {}).items():
            regions = info.get("regions_with_capacity_available", [])
            price = info.get("instance_type", {}).get("price_cents_per_hour", 0) / 100
            for r in regions:
                rname = r["name"]
                if rname not in availability:
                    availability[rname] = []
                availability[rname].append(f"{name} (${price:.2f}/hr)")
        return availability
    except Exception as e:
        print(f"WARNING: Lambda API check failed: {e}")
        return {}


def main() -> int:
    """Launch smoke test with private GHCR auth and multi-region rotation."""
    parser = argparse.ArgumentParser(description="Launch GPU smoke test via SkyPilot")
    parser.add_argument("--model", default="sam3_vanilla", help="Model family to test")
    parser.add_argument(
        "--cloud",
        default="lambda",
        choices=["lambda", "runpod", "gcp"],
        help="Cloud provider (default: lambda). GCP uses GAR image + spot T4/L4.",
    )
    parser.add_argument(
        "--spot", action="store_true", help="Use spot instances (cheaper)"
    )
    parser.add_argument(
        "--retries-per-region",
        type=int,
        default=3,
        help="Retries per region before moving to next (default: 3)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=10,
        help="Max full rounds through all regions (default: 10, ~30 min total)",
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=10,
        help="Seconds between retries (default: 10)",
    )
    args = parser.parse_args()

    _load_env()

    import sky

    # GCP uses GAR (public, no GHCR auth needed)
    if args.cloud == "gcp":
        return _launch_gcp(args)

    # GHCR-based clouds (Lambda, RunPod) need GitHub token
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        print("ERROR: GITHUB_TOKEN not set in .env (needed for GHCR pull)")
        return 1

    from sky.provision.docker_utils import DockerLoginConfig

    # Load the cloud-specific task YAML
    yaml_name = (
        "smoke_test_lambda.yaml" if args.cloud == "lambda" else "smoke_test_gpu.yaml"
    )
    yaml_path = str(_ROOT / "deployment" / "skypilot" / yaml_name)
    task = sky.Task.from_yaml(yaml_path)

    # Load and resolve env vars
    env_vars = _load_env_vars()
    env_vars = _resolve_mlflow_env(env_vars)
    env_vars["MODEL_FAMILY"] = args.model
    task.update_envs(env_vars)

    # Docker credentials for private GHCR
    docker_config = DockerLoginConfig(
        username=os.environ.get("GHCR_USERNAME", "petteriteikari"),
        password=github_token,
        server="ghcr.io",
    )

    if args.cloud == "lambda":
        return _launch_lambda_multi_region(task, docker_config, args)

    # RunPod path (original behavior)
    new_resources = set()
    for r in task.resources:
        new_r = r.copy(
            use_spot=args.spot,
            _docker_login_config=docker_config,
        )
        new_resources.add(new_r)
    task.set_resources(new_resources)

    print(f"=== Launching smoke test: {args.model} (RunPod) ===")
    request_id = sky.launch(
        task,
        cluster_name="minivess-smoke-test",
        idle_minutes_to_autostop=5,
        down=True,
    )
    print(f"Cluster launched. Request ID: {request_id}")
    print("Monitor: uv run sky logs minivess-smoke-test")
    return 0


def _launch_gcp(args) -> int:
    """Launch smoke test on GCP spot instance using managed jobs.

    Uses sky.jobs.launch() (managed jobs) instead of sky.launch() to get
    automatic spot recovery. sky.launch() with use_spot=True fails permanently
    on first preemption — managed jobs re-provision automatically.

    GCP uses GAR (public Docker image in same region as Cloud SQL/GCS/MLflow).
    No GHCR auth needed.
    """
    import sky

    yaml_path = str(_ROOT / "deployment" / "skypilot" / "smoke_test_gcp.yaml")
    task = sky.Task.from_yaml(yaml_path)

    # Load and resolve env vars (GCP MLflow uses MLFLOW_GCP_URI, not MLFLOW_CLOUD_URI)
    env_vars = _load_env_vars()
    gcp_uri = env_vars.get("MLFLOW_GCP_URI", os.environ.get("MLFLOW_GCP_URI", ""))
    if gcp_uri:
        env_vars["MLFLOW_TRACKING_URI"] = gcp_uri
    env_vars["MODEL_FAMILY"] = args.model
    task.update_envs(env_vars)

    # Managed jobs always use spot — auto-recovery handles preemptions
    new_resources = set()
    for r in task.resources:
        new_r = r.copy(use_spot=True)
        new_resources.add(new_r)
    task.set_resources(new_resources)

    # Unique job name per model — managed jobs use job names, not cluster names
    job_name = f"minivess-gcp-{args.model.replace('_', '-')}"

    print(f"=== Launching smoke test: {args.model} (GCP managed spot) ===")
    print(f"Job name: {job_name}")
    print(f"MLflow: {gcp_uri or '(not set — check MLFLOW_GCP_URI in .env)'}")
    print("Docker: GAR (public, no auth — SkyPilot picks cheapest region)")
    print("Mode: Managed job (auto-recovery from spot preemptions)")

    request_id = sky.jobs.launch(
        task,
        name=job_name,
    )
    print(f"Managed job submitted. Request ID: {request_id}")
    print(f"Monitor: uv run sky jobs logs {job_name}")
    print("Status:  uv run sky jobs queue")
    return 0


def _launch_lambda_multi_region(
    task,
    docker_config,
    args,
) -> int:
    """Launch on Lambda Labs with automatic multi-region rotation.

    Strategy: rotate through all 17 Lambda regions starting from unpopular
    EU/Asia regions (where capacity is more likely available). Try each GPU
    type in each region. 3 retries per region before moving to the next.
    """
    import sky

    print("=== Lambda Labs Multi-Region Launcher ===")
    print(f"Model: {args.model}")
    print(f"Regions: {len(LAMBDA_REGIONS_BY_PRIORITY)} (EU/Asia first → US last)")
    print(f"GPU priority: {' → '.join(LAMBDA_GPU_PRIORITY)}")
    print(f"Retries/region: {args.retries_per_region}")
    print(f"Max rounds: {args.max_rounds}")
    print("Docker: ghcr.io/petteriteikari/minivess-base:latest (private, GHCR auth)")
    print()

    # First, check real-time availability via Lambda API
    print("Checking real-time Lambda GPU availability...")
    availability = _check_lambda_availability()
    if availability:
        print("Regions with capacity:")
        for region, gpus in sorted(availability.items()):
            print(f"  {region}: {', '.join(gpus)}")
        # Reorder regions: available ones first
        available_regions = [r for r in LAMBDA_REGIONS_BY_PRIORITY if r in availability]
        unavailable_regions = [
            r for r in LAMBDA_REGIONS_BY_PRIORITY if r not in availability
        ]
        ordered_regions = available_regions + unavailable_regions
        print(f"\nPrioritized order: {', '.join(ordered_regions[:5])}...")
    else:
        print("No regions with availability (or API check failed)")
        ordered_regions = list(LAMBDA_REGIONS_BY_PRIORITY)
        print(f"Using default priority: {', '.join(ordered_regions[:5])}...")

    print()

    for round_num in range(1, args.max_rounds + 1):
        print(f"--- Round {round_num}/{args.max_rounds} ---")
        for region in ordered_regions:
            for retry in range(1, args.retries_per_region + 1):
                now = datetime.now(UTC).strftime("%H:%M:%S")
                # Try each GPU type for this region
                for gpu in LAMBDA_GPU_PRIORITY:
                    resource_spec = sky.Resources(
                        cloud=sky.clouds.Lambda(),
                        region=region,
                        accelerators=f"{gpu}:1",
                        image_id="docker:ghcr.io/petteriteikari/minivess-base:latest",
                        disk_size=40,
                        use_spot=args.spot,
                        _docker_login_config=docker_config,
                    )
                    task.set_resources(resource_spec)

                    print(
                        f"[{now}] Trying {gpu} in {region} "
                        f"(retry {retry}/{args.retries_per_region})...",
                        end=" ",
                        flush=True,
                    )
                    try:
                        request_id = sky.launch(
                            task,
                            cluster_name="minivess-smoke-test",
                            idle_minutes_to_autostop=5,
                            down=True,
                        )
                        print("SUCCESS!")
                        print(f"\nCluster launched in {region} with {gpu}!")
                        print(f"Request ID: {request_id}")
                        print("Monitor: uv run sky logs minivess-smoke-test")
                        return 0
                    except Exception as e:
                        err_msg = str(e)
                        if (
                            "insufficient-capacity" in err_msg.lower()
                            or "ResourcesUnavailable" in err_msg
                        ):
                            print("no capacity")
                        else:
                            print(f"error: {err_msg[:80]}")

                # Brief pause between retries within same region
                if retry < args.retries_per_region:
                    time.sleep(args.retry_delay)

        # Refresh availability between full rounds
        print(f"\nCompleted round {round_num}. Refreshing availability...")
        availability = _check_lambda_availability()
        if availability:
            available_regions = [
                r for r in LAMBDA_REGIONS_BY_PRIORITY if r in availability
            ]
            unavailable_regions = [
                r for r in LAMBDA_REGIONS_BY_PRIORITY if r not in availability
            ]
            ordered_regions = available_regions + unavailable_regions
            print(f"Available: {', '.join(available_regions) or 'none'}")
        else:
            print("Still no availability. Waiting 30s before next round...")
            time.sleep(30)
        print()

    print(
        f"\nExhausted all {args.max_rounds} rounds across {len(ordered_regions)} regions."
    )
    print("Lambda GPUs are globally sold out. Options:")
    print(
        "  1. Run again later: uv run python scripts/launch_smoke_test.py --cloud lambda"
    )
    print("  2. Use GCP (T4/L4 spot, same-region infra): --cloud gcp")
    print("  3. Use RunPod (container-based): --cloud runpod")
    return 1


if __name__ == "__main__":
    sys.exit(main())
