# Metalearning: Docker Research Report Assumed Missing Multi-Stage Builds

**Date:** 2026-03-17
**Severity:** MEDIUM — research report wasted hypothesis slots on already-implemented features
**Trigger:** User asked "I thought we already had multi-stage builds?"

---

## What Happened

1. Research report for #751 (Docker pull optimization) listed "H1: Multi-stage Dockerfile
   + uv cache mounts" as the top recommendation with "60% size reduction"
2. User correctly identified that 3-tier multi-stage builds were ALREADY implemented
3. Audit confirmed: ALL 3 base images use 2-stage builder→runner, uv cache mounts present,
   layer ordering optimized, .dockerignore comprehensive (75 rules)
4. The planning doc `docker-base-improvement-plan.md` is marked `status: implemented`

## Root Cause

The research agent searched the web for generic Docker optimization advice without first
reading the existing Dockerfiles in `deployment/docker/`. It assumed the repo had a naive
single-stage Dockerfile because that's the most common case in tutorials.

**This violates CLAUDE.md Rule #13**: "Read Context Before Implementing" — and its
corollary: read existing code before recommending changes.

## What Was Actually Already Implemented

| Feature | Status | File |
|---------|--------|------|
| 3-tier hierarchy (GPU/CPU/Light) | ✅ DONE | Dockerfile.base, .base-cpu, .base-light |
| Multi-stage builder→runner | ✅ DONE | All 3 bases (2 stages each) |
| uv cache mounts | ✅ DONE | `--mount=type=cache,target=/root/.cache/uv` |
| Layer ordering (deps before code) | ✅ DONE | uv sync → COPY src/ |
| BuildKit enabled | ✅ DONE | `# syntax=docker/dockerfile:1` + DOCKER_BUILDKIT=1 |
| Thin flow Dockerfiles | ✅ DONE | 14 flows, only COPY + ENV + CMD |
| Comprehensive .dockerignore | ✅ DONE | 75 rules |
| OCI labels | ✅ DONE | All images annotated |

## What the Report SHOULD Have Focused On

The 9 GB image is already optimized via multi-stage. The remaining 15 min pull is due to:

1. **gzip decompression bottleneck** — zstd would be 3-5x faster decompression
2. **Layer-by-layer sequential pull** — image streaming or eStargz would lazy-pull
3. **Image size floor** — CUDA runtime + PyTorch + MONAI + all deps = ~8-9 GB minimum
   (multi-stage already strips build tools, this is the irreducible size)
4. **RunPod pulls from Docker Hub** — pre-cached RunPod base images would skip base layers

## Corrective Action

- Update research report to note multi-stage IS implemented
- Re-rank hypotheses: skip H1 (already done), promote zstd/image-streaming/RunPod-base
- Always `Read deployment/docker/Dockerfile*` before writing Docker optimization advice

## Rule to Add

**Before recommending Docker optimizations, read ALL existing Dockerfiles first.**
The planning doc `docker-base-improvement-plan.md` has the implementation history.
