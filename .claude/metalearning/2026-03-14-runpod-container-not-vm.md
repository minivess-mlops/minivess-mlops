# RunPod Pods Are Containers, Not VMs — Docker Impossible (2026-03-14)

## The Failure

Spent 8 hours debugging SkyPilot + RunPod + Docker. Clusters torn down after 2 minutes,
region bouncing, image pull timeouts. Root cause: **architectural mismatch**.

## What I Got Wrong

Treated RunPod as equivalent to AWS/GCP/Lambda. Assumed `image_id: docker:` works the
same everywhere. It does NOT.

## The Correct Understanding

**RunPod pods ARE Docker containers.** There is no VM. No Docker daemon. No Docker-in-Docker.

| Provider Type | Examples | `image_id: docker:` means |
|---------------|----------|--------------------------|
| **VM-based** | Lambda, AWS, GCP, Azure | VM boots → Docker daemon pulls image → runs container |
| **Container-based** | RunPod | Image IS the pod — no Docker layer |

SkyPilot docs explicitly state:
> "Running docker containers is not supported on RunPod."

This means for Docker-mandatory workflows (our staging + prod environments):
- **RunPod is architecturally incompatible**
- **Lambda Labs is the correct choice** (VM-based, Docker preinstalled)
- RunPod can still serve the dev environment (non-Docker)

## Compounding Issues

1. 20+ GB Docker image + GHCR CDN latency from EU = 20-60 min pulls
2. RunPod may timeout pod creation before pull completes
3. SkyPilot injects its runtime INTO our container (dependency conflicts possible)
4. Private registry auth goes through RunPod API, not Docker daemon

## Rule

**ALWAYS verify the provider's execution model before choosing it for Docker workloads.**
VM-based providers (Lambda, AWS, GCP) support full Docker. Container-based providers
(RunPod) do NOT — they use the image as a runtime environment, which is fundamentally
different.

## Action Taken

- Created issue #681: Switch to Lambda Labs
- Created report: `docs/runpod-vs-lambda-vs-rest-for-skypilot-with-docker.md`
- RunPod demoted to dev environment only
