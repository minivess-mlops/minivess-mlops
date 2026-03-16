# Metalearning: GHCR Private Registry Caused 40+ Min RunPod Hangs

**Date:** 2026-03-16
**Severity:** P0 — Recurring blocker (PR #756 QA + this session)
**Issue:** #757

---

## What Happened

1. SkyPilot YAML for RunPod uses `image_id: docker:ghcr.io/petteriteikari/minivess-base:mamba-latest`
2. GHCR package was PRIVATE — RunPod pods cannot pull private GHCR images without credentials
3. RunPod provisioner hangs in STARTING state for 40+ minutes (silent Docker pull failure)
4. No error message — just perpetual STARTING until timeout
5. Same issue occurred in PR #756 QA (3 attempts, 40+ min each)

---

## Root Cause

**GHCR defaults to private for user-owned packages.** The KG decision node said
"GHCR for RunPod" but never verified that GHCR images were publicly pullable.
The assumption was "open-source repo = public Docker images" but GHCR doesn't
inherit repository visibility.

Additionally, the GitHub REST API does NOT support changing user-owned package
visibility programmatically — it requires the web UI. This makes autonomous
fixes impossible.

---

## Decision: Docker Hub for RunPod, GAR for GCP

| Environment | Registry | Reason |
|-------------|----------|--------|
| RunPod (dev) | **Docker Hub** | Public by default, zero-auth pull, simplest for researchers |
| GCP (staging/prod) | **GAR** | Same-region (europe-north1), ADC auth, no cross-continent pull |
| GitHub Actions CI | **GHCR** | Internal CI build artifacts only, not deployment images |

**Why not just make GHCR public?**
- GitHub API doesn't support it programmatically for user packages
- Requires manual web UI click every time a new package is created
- Docker Hub is public by default — zero friction

---

## Files Updated

- `deployment/skypilot/smoke_test_mamba.yaml` — `docker:petteriteikari/minivess-base:mamba-latest`
- `deployment/skypilot/smoke_test_gpu.yaml` — `docker:petteriteikari/minivess-base:latest`
- `configs/registry/dockerhub.yaml` — new registry config
- `configs/cloud/runpod_dev.yaml` — docker_registry: dockerhub
- `knowledge-graph/domains/infrastructure.yaml` — decision updated to dockerhub_and_gar
- `.env.example` — DOCKERHUB_TOKEN added

---

## Rule

When choosing a Docker registry for cloud GPU environments:
1. **Default to public registries** (Docker Hub) for any environment where pods
   pull images without pre-configured credentials
2. **Private registries** (GHCR, private GAR) only for environments with built-in
   auth (GitHub Actions has GITHUB_TOKEN, GCP VMs have ADC)
3. **Never assume** a registry is public without verifying with an unauthenticated pull
