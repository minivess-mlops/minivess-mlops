# Decision: GCS Pretrained Weight Caching

**Date**: 2026-03-28
**Status**: PROPOSED (awaiting user authorization)
**Category**: Infrastructure optimization, cost reduction, reliability improvement

## Context

SkyPilot factorial jobs download pretrained model weights from HuggingFace during
VM setup: SAM3 (~9 GB) and VesselFM (~2 GB). This takes 10-15 min per job,
during which spot VMs are vulnerable to preemption. With 34 factorial conditions,
total wasted download time is 340-510 minutes per experiment pass.

## Decision

Cache pretrained weights on GCS in europe-west4 (`gs://minivess-mlops-checkpoints/pretrained/`).
GCP SkyPilot jobs pull from same-region GCS (free, ~1-2 min) with HuggingFace fallback.
RunPod and local environments are unchanged.

## Key Design Choices

1. **Bucket**: Repurpose orphaned `minivess-mlops-checkpoints` bucket with `pretrained/` prefix.
   Does NOT create a competing artifact persistence mechanism — pretrained weights are
   external inputs (like training data), not training artifacts.

2. **NOT DVC-tracked**: Pretrained weights are immutable upstream artifacts that change
   on model release cadence (months/years). Path-based versioning with SHA256 checksums
   provides sufficient provenance without DVC complexity.

3. **NOT baked into Docker image**: SAM3 alone is ~9 GB. Baking it doubles image size
   to ~17-18 GB, penalizing all 22 non-SAM3 factorial conditions with unnecessary
   download time. Violates model-agnostic principle (TOP-1).

4. **GCS-first, HF-fallback**: Two independent paths to weights improves reliability.
   Same-region GCS is 99.95% available and eliminates exit code 34 (HF timeout) events.

## Not a Contract Violation

The `mlflow_only_artifact_contract` (KG invariant) governs **training artifacts** (outputs).
Pretrained weights are **external inputs**, analogous to DVC-managed training data.
This distinction was validated by the iterated LLM council (5 expert perspectives).

## Implementation Files

- Planning doc: `docs/planning/v0-2_archive/original_docs/gcs-weights-caching-report.md`
- KG decision: `knowledge-graph/domains/cloud.yaml` (pretrained_weight_caching node)
- Upload script: `scripts/upload_pretrained.sh` (to be created)
- Verify script: `scripts/verify_pretrained_gcs.sh` (to be created)
- SkyPilot YAML: `deployment/skypilot/train_factorial.yaml` (lines 168-203 to be replaced)
- Config: `.env.example` (GCS_PRETRAINED_BUCKET, version vars to be added)

## Cost

- Storage: $0.25/month ($3.04/year)
- Savings: ~$1.50/experiment pass in GPU idle time
- Break-even: First experiment pass

## CLAUDE.md Sections That Need Updating (after user approval)

- Line 99: Note checkpoints bucket role change to "pretrained weight cache"
- No new rules needed — fits within existing rules 22 (config) and 31 (YAML authorization)

## Tags

`#gcs` `#pretrained-weights` `#caching` `#cost-optimization` `#spot-preemption` `#reliability`
