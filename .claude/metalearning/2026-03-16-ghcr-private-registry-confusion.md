# Metalearning: GHCR Private Registry Caused 40+ Min RunPod Hangs

**Date:** 2026-03-16
**Severity:** P0 — Recurring blocker (PR #756 QA + this session)
**Issue:** #757

---

## What Happened

1. SkyPilot YAML for RunPod uses `image_id: docker:ghcr.io/...`
2. GHCR package was PRIVATE — RunPod pods cannot pull without credentials
3. Pod hangs in STARTING for 40+ min (silent Docker pull failure)
4. Same issue in PR #756 QA (3 attempts, 40+ min each)

## Decision: Docker Hub for RunPod, GAR for GCP

| Environment | Registry | Reason |
|-------------|----------|--------|
| RunPod (dev) | Docker Hub | Public by default, zero-auth pull |
| GCP (staging/prod) | GAR | Same-region, ADC auth |
| GitHub Actions CI | GHCR | Internal CI only |

## Rule

Default to public registries for cloud GPU environments where pods pull without
pre-configured credentials. Never assume a registry is public without verifying
with an unauthenticated pull.
