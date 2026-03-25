# Spot-Preferred / On-Demand Fallback Design

**Issue**: #964 (TDD Task 3.7)
**Status**: Draft -- awaiting user decision
**Author**: Claude Code
**Date**: 2026-03-25

## Problem Statement

SkyPilot spot instances provide significant cost savings (50-70%) but can be
preempted at any time. For long-running factorial production runs (34 jobs),
a single preemption without recovery wastes the partial compute spend. The
current configuration uses `use_spot: true` unconditionally, meaning jobs fail
on preemption unless SkyPilot's managed spot recovery (`spot_recovery: FAILOVER`)
is enabled.

This document evaluates the options for spot-preferred execution with on-demand
fallback, using SkyPilot's `any_of` resource specification.

## Cost Comparison

### Debug Runs (2 epochs, 4 volumes, 3 folds)

| Mode | GPU | Duration/job | Jobs | Cost/job | Total |
|------|-----|-------------|------|----------|-------|
| Spot (L4) | 1x L4 | ~5 min | 34 | ~$0.07 | ~$2.50 |
| On-demand (L4) | 1x L4 | ~5 min | 34 | ~$0.16 | ~$5.50 |

**Spot savings for debug**: ~$3.00 (55% cheaper)

### Production Runs (50 epochs, full data, 3 folds)

| Mode | GPU | Duration/job | Jobs | Cost/job | Total |
|------|-----|-------------|------|----------|-------|
| Spot (L4) | 1x L4 | ~45 min | 34 | ~$2.55 | ~$87 |
| On-demand (L4) | 1x L4 | ~45 min | 34 | ~$6.47 | ~$220 |

**Spot savings for production**: ~$133 (60% cheaper)

### Worst-Case Preemption Cost

If a job is preempted at 90% completion without checkpointing:

- **Debug**: ~$0.06 wasted per preemption (negligible)
- **Production**: ~$2.30 wasted per preemption (significant at scale)

With MLflow async checkpointing (every 5 epochs), maximum waste per preemption
is limited to ~5 epochs of compute (~$0.25 for production).

## SkyPilot `any_of` Syntax

SkyPilot supports resource alternatives via `any_of` (since v0.6). This allows
specifying spot-preferred with on-demand fallback in a single resource block:

```yaml
# train_factorial.yaml (proposed change -- NOT YET APPLIED)
resources:
  any_of:
    - accelerators: L4:1
      use_spot: true
      spot_recovery: FAILOVER
    - accelerators: L4:1
      use_spot: false    # on-demand fallback
  cloud: gcp
  region: europe-north1
  disk_size: 100
  image_id: docker:europe-north1-docker.pkg.dev/minivess-mlops/minivess/minivess-train:latest
```

SkyPilot tries resources in order: spot L4 first, then on-demand L4 if spot is
unavailable or preempted beyond retry limits.

### Alternative: `spot_recovery: FAILOVER_TO_ONDEMAND`

SkyPilot also supports `spot_recovery: FAILOVER_TO_ONDEMAND` which automatically
falls back to on-demand when spot is preempted. This is simpler than `any_of`
but provides less control:

```yaml
resources:
  accelerators: L4:1
  use_spot: true
  spot_recovery: FAILOVER_TO_ONDEMAND
```

## yaml_contract.yaml Changes Needed

If approved, the golden contract (`configs/cloud/yaml_contract.yaml`) would need
a new field to explicitly allow on-demand fallback:

```yaml
# Proposed addition to yaml_contract.yaml (NOT YET APPLIED)
spot_policy:
  allow_ondemand_fallback: true   # permits any_of with use_spot: false
  max_ondemand_cost_multiplier: 3.0  # reject if on-demand > 3x spot price
```

**Per Rule #31**: No changes to `yaml_contract.yaml` or any SkyPilot YAML will
be made until the user explicitly authorizes the modification.

## Options Summary

| Option | Pros | Cons | Estimated Cost (prod) |
|--------|------|------|----------------------|
| **A: Spot-only** (current) | Cheapest when available | Fails on preemption | ~$87 (no preemption) |
| **B: `any_of` spot→on-demand** | Resilient, cost-optimized | Slightly more complex YAML | ~$87-$220 (depends on availability) |
| **C: `FAILOVER_TO_ONDEMAND`** | Simple, resilient | Less control over fallback | ~$87-$220 |
| **D: On-demand only** | Never preempted | 2.5x more expensive | ~$220 |

## Recommendation

**Recommended: Option C (`spot_recovery: FAILOVER_TO_ONDEMAND`)** for the
following reasons:

1. **Simplicity**: Single resource block, no `any_of` complexity
2. **Cost-optimized**: Starts with spot pricing, only falls back when preempted
3. **Resilience**: Jobs complete even during spot shortage periods
4. **Checkpoint-aware**: Combined with MLflow async checkpointing, maximum waste
   per preemption is ~5 epochs of compute
5. **Contract-compatible**: Requires minimal change to `yaml_contract.yaml`
   (one new field: `allow_ondemand_fallback: true`)

**For debug runs**: Spot-only is acceptable (low cost, fast re-run if preempted).
**For production runs**: `FAILOVER_TO_ONDEMAND` protects against the ~$2.30/job
preemption waste across 34 factorial jobs.

**Action required**: User must authorize changes to:
- `configs/cloud/yaml_contract.yaml` (add `allow_ondemand_fallback` field)
- `deployment/skypilot/train_factorial.yaml` (add `spot_recovery` setting)

No YAML files will be modified until explicit user authorization per Rule #31.
