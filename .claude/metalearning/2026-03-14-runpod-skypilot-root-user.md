# RunPod + SkyPilot: Container Must Run as Root

**Date**: 2026-03-14
**Session**: RunPod SkyPilot integration (#626)
**Severity**: Critical — blocks all RunPod deployments

## Discovery

SkyPilot's RunPod provisioner SSHes into the container as `root`. The `setup_cmd`
(base64-encoded shell script injected as Docker CMD) installs openssh-server,
configures SSH for root login, and places SSH public keys in `~/.ssh/authorized_keys`
(which resolves to `/root/.ssh/authorized_keys` for root).

If the Dockerfile sets `USER minivess`, three layers of failure occur:

### Layer 1: Missing sudo
SkyPilot's `$(prefix_cmd)` returns "sudo" when the container user is non-root.
If `sudo` is not installed, all commands fail.

**Fix**: Added `sudo` package to runner stage.

### Layer 2: UID 1000 conflict
Ubuntu 24.04 base image (`nvidia/cuda:12.6.3-runtime-ubuntu24.04`) has a pre-existing
user `ubuntu` at UID 1000. Creating `minivess` with `-u 1000 -o` makes both users
share UID 1000. `whoami` returns `ubuntu` (first match), so the sudoers entry for
`minivess` doesn't apply.

**Fix**: Added `ubuntu ALL=(ALL) NOPASSWD:ALL` to sudoers.

### Layer 3: SSH key misplacement (root cause)
Even with sudo working, SkyPilot SSHes as `root` but the container runs as
`ubuntu`/`minivess` (UID 1000). The setup_cmd places SSH keys in
`/home/ubuntu/.ssh/authorized_keys` (because `~` resolves to `/home/ubuntu` for the
running user), but SSH server checks `/root/.ssh/authorized_keys` for root login.
Result: SSH auth fails, SkyPilot can't connect, pod crash-loops.

**Fix**: Disabled `USER minivess` directive. Container runs as root.

## Final Dockerfile Pattern

```dockerfile
# NOTE: USER minivess is intentionally NOT set here.
# SkyPilot/RunPod requires root: the setup_cmd installs packages and configures
# SSH as root, and SkyPilot SSHes as root. Setting USER to non-root causes the
# SSH key to be placed in the wrong home directory, preventing SSH auth.
# For local Docker Compose, use docker-compose.yml user: directive or
# --user flag to run as non-root when needed.
# USER minivess  # DISABLED — breaks SkyPilot SSH auth
```

## Additional Requirements for SkyPilot on RunPod

The runner stage must include these packages for SkyPilot setup_cmd:
- `openssh-server` — SSH server for SkyPilot to connect
- `rsync` — file transfer
- `curl` — network utilities
- `patch` — SkyPilot runtime patching
- `sudo` — privilege escalation for non-root scenarios
- `mkdir -p /run/sshd` — SSH server needs this directory

## Network Issues (Transient)

RunPod's API is fronted by Cloudflare. Intermittent connection failures (~15% rate)
affect the `run_graphql_query()` function. Fixed with retry logic (5 retries,
exponential backoff) patched into `.venv/.../runpod/api/graphql.py`.

Also patched `sitecustomize.py` for IPv4 preference, though the actual issue was
transient connectivity, not IPv6.

## Cost

- A40 (46 GB VRAM), CA region, on-demand: ~$0.39/hr
- sam3_vanilla smoke test: ~74 seconds of GPU time
- Total pod time including setup: ~3 minutes
- Estimated cost: ~$0.02

## Verified End-to-End Pipeline

```
UpCloud S3 --DVC pull--> RunPod A40 --train--> MLflow (local) --Prefect--> Completed
```

All stages succeeded:
1. DVC init + remote config
2. DVC pull from UpCloud S3 (350 files, ~2.7 GB)
3. nvidia-smi: NVIDIA A40, CUDA 12.8
4. SAM3 weight loading (1468/1468 params)
5. Prefect temporary server
6. MLflow experiment creation
7. Training fold completed
8. Job status: SUCCEEDED

## Known Issues

- `MLFLOW_TRACKING_URI` shows `${MLFLOW_CLOUD_URI}` literally in echo output —
  env var resolution may not work correctly with SkyPilot Python API + YAML envs.
  Training still completes (falls back to local mlruns).
- Git not installed in runner image — MLflow warns but continues without git SHA.
- `GITHUB_TOKEN` visible in plaintext in `~/sky_logs/.../provision.log` — needs rotation.
- All .venv patches are ephemeral — lost on `uv sync`.
