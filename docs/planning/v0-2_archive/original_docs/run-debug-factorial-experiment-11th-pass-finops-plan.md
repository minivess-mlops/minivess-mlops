# 11th Pass FinOps Plan — Cost Optimization + Shift-Left Policy-as-Code

**Date**: 2026-03-28
**Status**: Ready for execution
**Target**: Infrastructure costs < €20/month (currently ~€13.43/month, but €89.59/yr on GAR alone)
**Issues**: #878 (MLflow 413), new: FinOps shift-left test suite

## Current Cost Breakdown (March 2025 – March 2026)

| GCP Service | EUR/yr | EUR/mo | Root Cause |
|---|---|---|---|
| **Artifact Registry** | **89.59** | **7.47** | 73 GB stored, 11 stale untagged images, NO cleanup policy |
| Compute Engine | 53.97 | 4.50 | SkyPilot spot VMs + controller (productive spend) |
| Cloud Storage | 17.94 | 1.50 | 3 GCS buckets (DVC data, MLflow artifacts, checkpoints) |
| Cloud SQL | 10.55 | 0.88 | db-g1-small ALWAYS ON, serves disabled Cloud Run MLflow |
| Cloud Run | 7.04 | 0.59 | MLflow server (currently DISABLED but possibly orphaned) |
| Networking | 1.62 | 0.14 | Cross-region egress |
| **Total** | **180.71** | **15.06** | |

## Root Cause: CORRECTION — Original Analysis Was Wrong

**IMPORTANT: The original analysis underestimated GAR costs by ~30x.**

Storage alone explains only ~€3 of the €89.59 GAR bill (73 GB × $0.10/GB/mo × 13 days).
The remaining ~€86 must come from **network egress** — every SkyPilot job pull is a
6.4 GB download billed under Artifact Registry, not under "Networking."

**Egress cost breakdown (hypothesis — needs billing SKU export to confirm):**
- Same-region pulls (europe-north1 VM ← europe-north1 GAR): **FREE**
- Cross-region pulls (europe-west4 VM ← europe-north1 GAR): $0.01/GB = $0.064/pull
- Internet pulls (RunPod/local ← GAR): $0.12/GB = **$0.77/pull**

**Estimated pulls in 2 weeks:**
- ~120 SkyPilot job launches (10 passes, some with 20-50 jobs)
- ~20 local dev pulls (docker pull during development)
- Some RunPod pulls

**At ~117 internet-rate pulls × $0.77 = ~$90 — this matches the €89.59 bill.**

This means the GAR cost is **dominated by EGRESS, not storage.** Cleaning up stale
images (my original recommendation) would save only ~€3/yr, not €82/yr. The real fix
is reducing image pulls and/or image size.

**Cloud SQL (6% of total, but 100% waste when Cloud Run disabled):**
- db-g1-small runs 24/7 ($0.88/mo) even when Cloud Run MLflow is disabled
- No consumer currently uses it — Cloud Run is `enable_cloud_run: false`

**Cloud Run (4% of total):**
- Charged despite being disabled in Pulumi — likely orphaned from a previous `pulumi up`
- `min_instance_count: 1` means it never scales to zero even if re-enabled

**ACTION NEEDED: Enable billing export to BigQuery to get SKU-level cost breakdown.**
Without this, we're guessing at the split between storage vs egress.
Go to: https://console.cloud.google.com/billing/01DCCF-E3B6B4-0616FE/reports
and filter by Artifact Registry → group by SKU to see the actual breakdown.

---

## Decision Matrix: Registry Strategy

| Criterion (weight) | H1: GAR + cleanup | H2: Docker Hub | H3: GHCR | H4: DockerHub + GAR cache | H5: GHCR + GAR cache |
|---|---|---|---|---|---|
| Monthly cost (25%) | **4** €0.62/mo | **5** €0/mo | **5** €0/mo | **5** ~€0/mo | **5** ~€0/mo |
| GCP pull speed (20%) | **5** Same-region, ~30s | **2** Cross-Atlantic, 5-10 min | **2** Cross-Atlantic, 5-10 min | **4** First slow, cached after | **4** First slow, cached after |
| RunPod pull speed (10%) | **2** Cross-region, 2-5 min | **5** DockerHub CDN, ~1-2 min | **4** GHCR CDN, ~2-3 min | **5** Same as H2 | **4** Same as H3 |
| Reliability (15%) | **5** GCP SLA 99.95% | **3** Rate limits (200/6hr anon) | **4** GitHub SLA | **4** Redundant | **4** Redundant |
| Setup complexity (10%) | **5** 1 cleanup policy in Pulumi | **3** Re-tag, update 5 YAMLs | **2** Private-by-default incident | **3** Dual registry | **3** Dual registry |
| Free tier (5%) | **4** $0.10/GB/mo | **3** 1 private repo, rate limits | **4** 10 GB free private | **3** Rate limits | **4** 10 GB free |
| Egress costs (5%) | **5** €0 same-region | **3** €0 but slower | **3** €0 but slower | **4** €0 after cache | **4** €0 after cache |
| Academic suitability (10%) | **4** Needs GCP billing | **4** Universal | **5** Academic standard | **4** Complex | **5** Academic + perf |
| **Weighted Total** | **4.30** | **3.55** | **3.45** | **4.10** | **4.10** |

**REVISED: If egress dominates (not storage), the decision changes.**

Storage alone explains only ~€3 of the €89.59. The rest is likely **network egress**
from internet pulls (local dev + RunPod pulling 6.4 GB images from GAR at $0.12/GB).

**Contingent decision (needs billing SKU confirmation):**
- If >80% is egress → **H4 wins** (Docker Hub primary + GAR pull-through cache)
  - Store on Docker Hub (free), cache in GAR for GCP same-region pulls
  - Eliminates all internet egress from GAR
  - GCP SkyPilot still gets fast same-region pulls via cache
- If >80% is storage → **H1 wins** (cleanup policy saves ~€80/yr)
- **Either way**: Add GAR cleanup policy AND reduce image size

**Action**: Check GCP Console → Billing → Reports → Artifact Registry → group by SKU
to see the actual storage vs egress split before committing to a strategy.

**Decision**: Keep dual-registry (Docker Hub for RunPod/local, GAR for GCP). Add cleanup
policy regardless. Investigate egress before any registry migration.

---

## Decision Matrix: Cloud SQL

| Option | Cost | Latency | Complexity | Verdict |
|---|---|---|---|---|
| Keep db-g1-small always-on | €10.55/yr | Low | None | Current, wasteful when idle |
| Downgrade to db-f1-micro | €7.67/yr | Low | 1-line Pulumi | Better but still always-on |
| **Disable when Cloud Run disabled** | **€0/yr idle** | N/A | Pulumi gate | **Winner** — no consumer = no DB |
| SQLite on GCS | €0/yr | N/A | N/A | **REJECTED** — GCS has no file locking, SQLite requires it |
| Neon PostgreSQL free tier | €0/yr | Med (US-east) | Medium | Good for always-on, but latency from EU |

**Decision**: Gate Cloud SQL on `enable_cloud_run`. When Cloud Run is off, Cloud SQL should be off. The Pulumi code already has the `enable_cloud_run` gate — extend it to Cloud SQL.

---

## Shift-Left FinOps: Policy-as-Code Test Suite

The 12-hour job disaster and the €89.59 GAR bill both share the same root cause: **no pre-deployment governance checks.** The solution is a deterministic test suite that runs in `make test-staging` and blocks PRs that introduce cost violations.

Reference: [Shift-Left FinOps (Firefly)](https://www.firefly.ai/blog/shift-left-finops-how-governance-policy-as-code-are-enabling-cloud-cost-optimization)

### Principles

1. **Deterministic, not stochastic** — pytest assertions, not LLM monitoring
2. **Pre-commit, not post-incident** — catch cost violations BEFORE merge
3. **Config-driven thresholds** — costs limits in YAML, not hardcoded
4. **Static analysis of IaC** — parse Pulumi/SkyPilot YAML for policy violations
5. **Budget gates in experiment XML** — every experiment declares expected cost

### Proposed Test File: `tests/v2/unit/finops/test_cost_governance.py`

```
Test Classes:

1. TestGARCleanupPolicy
   - test_gar_repo_has_cleanup_policy() — parse Pulumi __main__.py, assert cleanup_policies defined
   - test_cleanup_policy_retains_max_n_tagged() — assert keep-count <= 3
   - test_cleanup_policy_deletes_untagged_after_days() — assert untagged TTL <= 14 days

2. TestSkyPilotCostGovernance
   - test_spot_instance_used_by_default() — parse train_factorial.yaml, assert use_spot: true
   - test_max_restarts_bounded() — assert max_restarts_on_errors <= 5
   - test_disk_size_not_excessive() — assert disk_size <= 200 GB
   - test_no_a100_without_explicit_approval() — parse accelerators, reject A100-80GB (yaml_contract.yaml)
   - test_experiment_declares_expected_cost() — parse experiment XML, assert <cost-estimate> present
   - test_experiment_declares_max_duration() — assert <max-duration-minutes> per job type

3. TestCloudSQLGovernance
   - test_cloud_sql_gated_on_cloud_run() — parse Pulumi, assert Cloud SQL provisioned ONLY when enable_cloud_run=true
   - test_cloud_sql_tier_not_oversized() — assert tier is db-f1-micro or db-g1-small (not db-n1-standard-*)
   - test_cloud_sql_has_auto_stop() — assert activation_policy or equivalent

4. TestCloudRunGovernance
   - test_cloud_run_min_instances_zero_for_dev() — assert min_instance_count == 0 for dev stack
   - test_cloud_run_max_instances_bounded() — assert max_instance_count <= 3

5. TestDockerImageGovernance
   - test_base_image_uses_multistage_build() — parse Dockerfile.base, assert "FROM.*AS builder" AND "FROM.*AS runner"
   - test_no_dev_deps_in_production_image() — assert "--no-dev" in uv sync command
   - test_image_size_budget() — (integration) docker image inspect, assert < 8 GB

6. TestExperimentBudgetGovernance
   - test_debug_experiment_cost_under_budget() — parse debug.yaml factorial, compute:
     n_conditions × expected_duration × hourly_rate, assert < $10
   - test_production_experiment_cost_under_budget() — same, assert < $50
   - test_no_on_demand_for_dynunet() — DynUNet is cheap, must use spot
   - test_on_demand_allowed_for_sam3() — SAM3 can use on-demand (80% preemption rate)
```

### Why This Works

- **GAR cleanup**: `test_gar_repo_has_cleanup_policy()` would have caught the missing cleanup policy at PR review time, before 11 stale images accumulated
- **12h job**: `test_experiment_declares_max_duration()` would have forced the experiment XML to declare a max duration, which the monitoring skill would enforce
- **A100 cost explosion**: `test_no_a100_without_explicit_approval()` already exists in `test_yaml_contract_enforcement.py` — extend the pattern to cost governance
- **Cloud SQL waste**: `test_cloud_sql_gated_on_cloud_run()` would catch the always-on DB when no consumer exists

---

## Optimization Actions

### Immediate (saves ~€85/yr)

| # | Action | Annual Savings | Effort |
|---|---|---|---|
| 1 | **Delete 11 stale GAR images** | €82.15 | 5 min (gcloud commands) |
| 2 | **Add GAR cleanup policy in Pulumi** | prevents recurrence | 10 min |
| 3 | **Delete orphaned Cloud Run service** (if exists) | €7.04 | 5 min |
| 4 | **Verify no orphaned Compute Engine VMs/disks** | €0-10 | 5 min |

### Medium-term (saves ~€10/yr)

| # | Action | Annual Savings | Effort |
|---|---|---|---|
| 5 | Gate Cloud SQL on enable_cloud_run | €10.55 when idle | 15 min |
| 6 | Set Cloud Run min_instance_count: 0 | ~€3 | 1 line |
| 7 | Create `tests/v2/unit/finops/` test suite (6 classes, ~20 tests) | prevents future waste | 2-3 hours |

### Post-Optimization Projected Costs

| Service | Current EUR/yr | Optimized EUR/yr |
|---|---|---|
| Artifact Registry | 89.59 | **~7.44** |
| Compute Engine | 53.97 | ~50 (minor cleanup) |
| Cloud Storage | 17.94 | ~17.94 (productive) |
| Cloud SQL | 10.55 | **~0** (gated on Cloud Run) |
| Cloud Run | 7.04 | **~0** (orphan deleted, or min=0) |
| Networking | 1.62 | ~1.62 |
| **Total** | **180.71** | **~77** |
| **Monthly** | **15.06** | **~6.42** |

**57% total reduction. Monthly cost: ~€6.42 (well under €20 target).**

---

## Executable Plan Sequence

### Phase 1: Immediate Cleanup (run NOW, no code changes)

```bash
# 1. Delete stale GAR images
gcloud artifacts docker images list europe-north1-docker.pkg.dev/minivess-mlops/minivess/base \
  --include-tags --format="table(version,tags,metadata.imageSizeBytes)" \
  --filter="NOT tags:latest"

# Delete each untagged image (11 commands from investigation report)
# ... (see cleanup commands in investigation)

# 2. Check for orphaned Cloud Run services
gcloud run services list --region europe-north1 --project minivess-mlops

# 3. Check for orphaned Compute Engine VMs/disks
gcloud compute instances list --project minivess-mlops
gcloud compute disks list --project minivess-mlops
```

### Phase 2: Pulumi Governance (code changes + deploy)

Add GAR cleanup policy, gate Cloud SQL, set Cloud Run min=0.
These are IaC changes in `deployment/pulumi/gcp/__main__.py`.

### Phase 3: Shift-Left Test Suite (TDD via /self-learning-iterative-coder)

Create `tests/v2/unit/finops/` with 6 test classes parsing Pulumi, SkyPilot YAML,
Dockerfiles, and experiment configs for cost governance violations.

### Phase 4: Experiment Budget Config

Add `expected_duration_minutes`, `max_duration_minutes`, and `cost_budget_usd` to
`configs/factorial/debug.yaml` and the experiment XML schema.

---

## What NOT to Do

1. **Do NOT switch from GAR to Docker Hub for GCP jobs** — cross-Atlantic pulls waste GPU time ($0.85-1.70 per factorial pass)
2. **Do NOT use SQLite on GCS** — GCS has no file locking, SQLite would corrupt
3. **Do NOT rely on LLM-based monitoring for cost governance** — deterministic tests in pytest are the shift-left approach
4. **Do NOT set budget alerts as the primary control** — alerts are reactive (post-incident), tests are proactive (pre-deploy)

---

## Cross-References

- `deployment/pulumi/gcp/__main__.py` — GAR repo (line 135-141), Cloud SQL (line 88-123), Cloud Run (line 224-289)
- `deployment/skypilot/train_factorial.yaml` — SkyPilot cost-relevant config
- `configs/factorial/debug.yaml` — factorial experiment definition
- `knowledge-graph/domains/infrastructure.yaml` — docker_registry resolved decision
- `.claude/metalearning/2026-03-27-no-job-duration-monitoring-12h-debug-run.md`
- `.claude/metalearning/2026-03-27-mlflow-413-10-passes-never-fixed-self-reflection.md`
- [Shift-Left FinOps (Firefly)](https://www.firefly.ai/blog/shift-left-finops-how-governance-policy-as-code-are-enabling-cloud-cost-optimization)
