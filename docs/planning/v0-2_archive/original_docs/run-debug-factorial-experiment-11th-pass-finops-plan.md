# 11th Pass FinOps Plan — Cost Optimization + Shift-Left Policy-as-Code

**Date**: 2026-03-28 (v3 — corrected with actual billing SKU data)
**Status**: Ready for execution
**Target**: Infrastructure costs < €20/month
**Issues**: #878 (MLflow 413), new: FinOps shift-left test suite

## Actual Cost Breakdown (March 2026, from Gemini Cloud Assist + SKU report)

**Total: €90.80/month** — for an academic project that should cost <€20.

### Artifact Registry: €89.59 (99% of GAR cost is EGRESS, not storage)

| SKU | Usage | EUR | % of GAR |
|-----|-------|-----|----------|
| **Intercontinental egress** (Excl Oceania) | 850 GB | **€57.71** | 64% |
| **Europe-to-Europe cross-region egress** | 1,879 GB | **€31.87** | 36% |
| Storage | 15 GB-months | €1.23 | 1% |
| Internet egress Europe-to-Europe | 0 GB | €0.00 | 0% |

**Root cause**: GAR is in **europe-north1 (Finland)** which has **ZERO L4 GPUs**.
Every SkyPilot job runs in a different region → every Docker pull (6.4 GB) is
cross-region or intercontinental egress. Same-region pulls are FREE.

**850 GB intercontinental** = ~133 image pulls to US/Asia (SkyPilot fallback regions).
**1,879 GB Europe-to-Europe** = ~294 image pulls to europe-west4/west1 (EU L4 regions).
**Total: ~427 pulls × 6.4 GB = ~2,729 GB** — matches billing exactly (850 + 1,879).

Preemption multiplies pulls: 3 recoveries per job × 6.4 GB = 4 pulls per job.

### Compute Engine: €53.97

| SKU | Usage | EUR | What |
|-----|-------|-----|------|
| **N4 Instance Core (Belgium)** | 380 hours | **€11.06** | SkyPilot controller (24/7) |
| **N4 Instance RAM (Belgium)** | 1,519 GB-hours | **€5.02** | SkyPilot controller RAM |
| Nvidia L4 GPU Spot (Seoul) | 69 hours | €9.10 | Training jobs (intercontinental!) |
| Nvidia L4 GPU Spot (Netherlands) | 38.65 hours | €8.63 | Training jobs (correct region) |
| Other compute | — | ~€20 | Various VMs, disks |

**Controller problem**: n4-standard-4 (4 vCPU, 16 GB) running 380h = €16.08/month.
This is a scheduler that submits jobs — it doesn't need 4 vCPUs and 16 GB RAM.

### Other Services

| Service | EUR/mo | Action |
|---------|--------|--------|
| Cloud Storage | €17.94 | Productive (DVC data) — keep |
| Cloud SQL | €10.55 | db-g1-small 24/7, no consumers — disable |
| Cloud Run | €7.04 | Possibly orphaned (disabled in Pulumi) — delete |
| Networking | €1.62 | Normal — keep |

---

## The Fix: Move Everything to europe-west4

**europe-west4 (Netherlands)** is the optimal region:
- **3 L4 zones** (most capacity of any EU region)
- L4 spot: $0.340/hr | L4 on-demand: $0.742/hr
- GAR, Cloud SQL, Cloud Run all supported
- Already the #1 priority in our `europe.yaml` region config

### Cost Impact of Co-Locating GAR + GPU in europe-west4

| Scenario | GPU Cost | GAR Egress | Total/month |
|----------|----------|------------|-------------|
| **Current** (GAR north1, jobs everywhere) | €17.73 | **€89.58** | **€107.31** |
| **Fixed** (GAR west4, jobs EU-only spot) | €17.73 | **€0.00** | **€17.73** |
| **Savings** | — | **€89.58** | **83% reduction** |

For a 32-job debug factorial:

| Scenario | Cost |
|----------|------|
| Current (spot + cross-region egress) | ~€114 |
| **Fixed (spot + co-located GAR)** | **~€22** |
| Fixed (on-demand + co-located GAR) | ~€47 |

**Spot + co-located GAR wins decisively.** On-demand is only worth considering for
SAM3 if preemption rate >60% (each preemption = wasted setup + extra image pull).

---

## Controller Optimization: n4-standard-4 → e2-small

| Instance | vCPU | RAM | $/hr | $/month (24/7) | $/month (4h/day) |
|----------|------|-----|------|----------------|------------------|
| **n4-standard-4 (current)** | 4 | 16 GB | $0.169 | **$121** | **$20** |
| n4-standard-2 | 2 | 8 GB | $0.084 | $61 | $10 |
| **e2-medium (recommended)** | 2 | 4 GB | $0.034 | $24 | **$4** |
| e2-small | 2 | 2 GB | $0.017 | $12 | $2 |

The controller already has 10-min autostop configured. With ~4h/day active use:
- Current: €16/month
- e2-medium: ~€4/month (**75% savings**)
- Also move from europe-west1 → europe-west4 (co-locate with jobs)

---

## EU Region Comparison

| Region | L4 Zones | L4 Spot $/hr | L4 On-Demand $/hr | Actual Provisions (March) |
|--------|----------|-------------|-------------------|--------------------------|
| **europe-west4 (Netherlands)** | **3** | $0.340 | $0.742 | 0 (not in config!) |
| europe-west1 (Belgium) | 2 | **$0.294** | $0.778 | 7 |
| europe-west3 (Frankfurt) | 2 | $0.416 | $0.834 | 0 |
| europe-west2 (London) | 2 | $0.391 | $0.805 | 0 |
| europe-north1 (Finland) | **0** | N/A | N/A | 0 (no L4!) |

**europe-west4 has the best availability (3 zones) and competitive spot pricing.**
europe-west1 has the cheapest spot rate but only 2 zones (less resilient to preemption).

**Recommendation**: Primary region = europe-west4. Fallback = europe-west1.
**No US/Asia fallback for debug runs** (use `europe.yaml`, not `europe_us.yaml`).

---

## Decision: Region Config Strategy

### New region subconfigs:

**`configs/cloud/regions/europe_strict.yaml`** (for debug + cost-sensitive runs):
- europe-west4 only (same region as GAR → zero egress)
- Spot by default
- On-demand fallback for SAM3 if preemption >3x

**`configs/cloud/regions/europe.yaml`** (existing, for production runs):
- europe-west4 → europe-west1 → europe-west3
- Spot by default
- Allows cross-region within EU (small egress: $0.01/GB)

**`configs/cloud/regions/europe_us.yaml`** (existing, for when EU is exhausted):
- EU first, then US fallback
- **BANNED for debug runs** (causes intercontinental egress)

---

## Shift-Left FinOps: Policy-as-Code Test Suite

The 12h job and €89 egress bill share the same root cause: **no pre-deployment governance.**

Reference: [Shift-Left FinOps (Firefly)](https://www.firefly.ai/blog/shift-left-finops-how-governance-policy-as-code-are-enabling-cloud-cost-optimization)

### Proposed: `tests/v2/unit/finops/test_cost_governance.py`

```
1. TestGARRegionGovernance (NEW — would have prevented the €89 egress)
   - test_gar_region_has_l4_gpus() — assert GAR region is in a region with L4 GPUs
   - test_gar_region_matches_primary_gpu_region() — assert GAR region == first region in region config
   - test_gar_has_cleanup_policy() — assert cleanup_policies defined in Pulumi
   - test_no_intercontinental_fallback_in_debug() — parse debug.yaml region_config, reject US/Asia

2. TestSkyPilotCostGovernance
   - test_spot_instance_used_by_default() — parse train_factorial.yaml, assert use_spot: true
   - test_max_restarts_bounded() — assert max_restarts_on_errors <= 5
   - test_disk_size_not_excessive() — assert disk_size <= 200 GB
   - test_no_a100_without_explicit_approval() — reject A100-80GB
   - test_experiment_declares_expected_cost() — assert <cost-estimate> in experiment XML
   - test_experiment_declares_max_duration() — assert <max-duration-minutes> per job type
   - test_debug_uses_europe_strict_region() — debug.yaml must use europe_strict, not europe_us

3. TestControllerGovernance
   - test_controller_instance_type_cost_bounded() — assert cpus <= 2 in .sky.yaml
   - test_controller_autostop_configured() — assert idle_minutes_to_autostop <= 15
   - test_controller_region_matches_gar() — assert controller region has L4 GPUs

4. TestCloudSQLGovernance
   - test_cloud_sql_gated_on_cloud_run() — assert Cloud SQL provisioned ONLY when enable_cloud_run=true
   - test_cloud_sql_tier_not_oversized() — assert tier is db-f1-micro or db-g1-small

5. TestExperimentBudgetGovernance
   - test_debug_experiment_cost_under_budget() — compute n_conditions × expected_duration × rate, assert < $10
   - test_no_on_demand_for_dynunet() — DynUNet must use spot (cheap, fast)
```

### Why This Works (Prevention of Each March Incident)

| Incident | Test That Would Have Caught It |
|----------|-------------------------------|
| €89 GAR egress | `test_gar_region_matches_primary_gpu_region()` |
| 12h debug job | `test_experiment_declares_max_duration()` |
| Jobs in Seoul/Iowa | `test_no_intercontinental_fallback_in_debug()` |
| Oversized controller | `test_controller_instance_type_cost_bounded()` |
| Idle Cloud SQL | `test_cloud_sql_gated_on_cloud_run()` |

---

## Executable Plan (Priority Order)

### Phase 1: Region Migration (saves €89/month — 83% of total)

| # | Action | Savings | Effort |
|---|--------|---------|--------|
| 1 | Create GAR repo in europe-west4 | — | Pulumi code change |
| 2 | Push base:latest + mlflow-gcp to new GAR | — | Docker push (~30 min) |
| 3 | Update SkyPilot YAMLs with new image_id path | — | 5 YAML files |
| 4 | Create `europe_strict.yaml` (europe-west4 only) | — | New config file |
| 5 | Set `region_config: europe_strict` in debug.yaml | — | 1 line |
| 6 | Delete old GAR repo in europe-north1 | €89/mo | After verification |
| 7 | `pulumi up` to deploy | — | Requires auth |

### Phase 2: Controller Downsizing (saves €12/month)

| # | Action | Savings | Effort |
|---|--------|---------|--------|
| 8 | Edit `.sky.yaml`: cpus 4+ → 2+, region → europe-west4 | €12/mo | 2 lines |
| 9 | Tear down current controller: `uv run sky down -a` | — | 1 command |

### Phase 3: Orphan Cleanup (saves €17/month)

| # | Action | Savings | Effort |
|---|--------|---------|--------|
| 10 | Delete orphaned Cloud Run service | €7/mo | 1 command |
| 11 | Disable Cloud SQL (gate on enable_cloud_run) | €10/mo | Pulumi change |

### Phase 4: Shift-Left Tests (prevents recurrence)

| # | Action | Savings | Effort |
|---|--------|---------|--------|
| 12 | Create `tests/v2/unit/finops/` test suite | Future prevention | 2-3 hours TDD |
| 13 | Add duration/cost config to debug.yaml | Future prevention | 30 min |

### Post-Optimization Projected Costs

| Service | Current EUR/mo | Optimized EUR/mo |
|---------|---------------|-----------------|
| Artifact Registry | 89.59 | **~1.23** (storage only, zero egress) |
| Compute Engine | 53.97 | **~22** (same GPU, smaller controller) |
| Cloud Storage | 17.94 | ~17.94 (productive) |
| Cloud SQL | 10.55 | **~0** (disabled) |
| Cloud Run | 7.04 | **~0** (deleted) |
| Networking | 1.62 | ~1.62 |
| **Total** | **€180.71/yr (€15.06/mo)** | **~€42.79/yr (~€3.57/mo)** |

**76% total reduction. From €15.06/month to ~€3.57/month.**
The dominant remaining cost is actual GPU training — the productive spend.

---

## What NOT to Do

1. **Do NOT keep GAR in europe-north1** — zero L4 GPUs, 100% cross-region egress
2. **Do NOT use `europe_us.yaml` for debug runs** — intercontinental egress costs 5x more than the GPU time
3. **Do NOT use SQLite on GCS** — no file locking support
4. **Do NOT rely on LLM monitoring for cost governance** — deterministic pytest tests are the shift-left approach
5. **Do NOT keep n4-standard-4 for the controller** — a scheduler doesn't need 16 GB RAM

---

## Cross-References

- `deployment/pulumi/gcp/__main__.py` — GAR repo, Cloud SQL, Cloud Run
- `deployment/skypilot/train_factorial.yaml` — image_id, cloud, regions
- `configs/cloud/regions/europe.yaml` — EU region priority list
- `configs/cloud/regions/europe_us.yaml` — EU+US fallback (BANNED for debug)
- `configs/factorial/debug.yaml` — region_config setting (line 133)
- `.sky.yaml` — controller placement and sizing
- `.claude/metalearning/2026-03-27-no-job-duration-monitoring-12h-debug-run.md`
- `.claude/metalearning/2026-03-27-mlflow-413-10-passes-never-fixed-self-reflection.md`
- [Shift-Left FinOps (Firefly)](https://www.firefly.ai/blog/shift-left-finops-how-governance-policy-as-code-are-enabling-cloud-cost-optimization)
