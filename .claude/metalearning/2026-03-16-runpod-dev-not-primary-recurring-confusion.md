# Metalearning: RunPod is dev/env ONLY — GCP is the primary target

**Date**: 2026-03-16
**Severity**: HIGH — recurring confusion that angers the user
**Occurrences**: Multiple sessions, this is the Nth time

## The Failure Pattern

Claude Code repeatedly treats RunPod as if it's a co-equal deployment target alongside GCP.
It keeps asking questions about RunPod deployment, RunPod serverless endpoints, and
RunPod infrastructure as if they matter equally to GCP.

## The Truth (NON-NEGOTIABLE)

```
┌─────────────────────────────────────────────────────────────┐
│  GCP = staging + prod = THE PRIMARY TARGET                  │
│  - All production-grade infrastructure                      │
│  - Pulumi IaC for provisioning                              │
│  - BentoML deployment to Cloud Run / GKE                    │
│  - GCS for data storage                                     │
│  - This is what the PAPER describes                         │
│  - This is what external researchers replicate              │
│                                                             │
│  RunPod = dev/env = BACKUP / QUICK EXPERIMENTS ONLY         │
│  - For researchers who like RunPod for quick GPU access      │
│  - NOT the focus of the repo                                │
│  - NOT where production deployment happens                  │
│  - NOT where drift monitoring runs in production            │
│  - A convenience, not the architecture                      │
└─────────────────────────────────────────────────────────────┘
```

## Why This Keeps Happening

1. RunPod appears in cloud config alongside GCP → Claude treats them as equals
2. The three-environment model (local/env/staging+prod) lists RunPod → Claude
   over-indexes on it
3. Claude sees RunPod Serverless screenshots → assumes it's a deployment target
4. Claude doesn't internalize the HIERARCHY: GCP is primary, RunPod is secondary

## The Rule

**NEVER ask about RunPod deployment specifics unless the user explicitly brings it up.**
**NEVER treat RunPod as co-equal to GCP.**
**When planning deployment/monitoring/drift features, ASSUME GCP.**
**RunPod support is "nice to have" that comes FOR FREE via SkyPilot abstraction.**

Both clouds are accessed through SkyPilot (compute) and Pulumi (IaC). The abstraction
layer means RunPod support is automatic — it does NOT need special attention.

## How to Apply

- When planning drift monitoring pipeline → target GCP (Cloud Run BentoML)
- When discussing deployment → default to GCP staging/prod
- When the user mentions RunPod → it's for quick dev experiments, not architecture decisions
- NEVER open questions about "should we also support RunPod for X?" — YES, via SkyPilot, automatically
