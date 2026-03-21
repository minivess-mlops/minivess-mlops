# Self-Reflection: Low-Confidence Regions Requiring Human Input

**Date**: 2026-03-20
**Purpose**: Identify documentation and KG regions where confidence is low and
human-in-the-loop decisions are needed before executing the debug factorial run.

---

## Methodology

Scanned all KG decision nodes for:
1. `status: partial | not_started | planned` — decisions not yet resolved
2. Low posterior probabilities (max option < 0.5) — tied or uncertain decisions
3. Contradictions between docs and implementation
4. Implementation gaps between KG decisions and actual code

---

## TIER 1: Blocking the Debug Run (Must Resolve Before Execution)

### 1.1 Composable Factorial YAML Does Not Yet Exist

**Current state**: `configs/hpo/paper_factorial.yaml` defines only Layer A (training) factors.
The full 6-factor design (4×3×2×3×2×5=720 from `pre-gcp-master-plan.xml` line 16) requires
a new `configs/factorial/` directory with sectioned YAML per user decision.

**What exists**: Only training factors in `paper_factorial.yaml` and `debug_factorial.yaml`.
**What's missing**: `factors.post_training` and `factors.analysis` sections.
**Confidence**: LOW — the YAML structure was just decided (2026-03-20) but not implemented.

**Human input needed?**: NO — structure is decided. Implementation work only.

### 1.2 Post-Training Flow Creates Runs in Wrong Experiment

**Current code**: `post_training_flow.py` logs to `minivess_post_training` (separate experiment).
**Decided**: Should log to `minivess_training` (SAME experiment) per synthesis Part 2.3.
**Confidence**: MEDIUM — decision made but code not yet updated. Risk of Analysis Flow
not discovering post-training variants if experiment name mismatch persists.

**Human input needed?**: NO — decided. Code change needed.

### 1.3 Analysis Flow: One Master Run vs Separate Runs Per Strategy

**Current code**: Analysis logs all evaluations under ONE master run in `minivess_evaluation`.
**Decided**: Each ensemble strategy should create its OWN run for cleaner Biostatistics discovery.
**Confidence**: MEDIUM — decision made but implementation not verified.

**Human input needed?**: NO — decided. Implementation work.

---

## TIER 2: Uncertain Decisions (May Need User Input)

### 2.1 Calibration Method (4-Way Tie, posterior 0.25 each)

**Decision**: `knowledge-graph/decisions/L3-technology/calibration_method.yaml`
**Status**: `partial`
**Options (all at 0.25)**:
- Temperature Scaling (netcal)
- Local Temperature Scaling
- Mondrian Conformal (MAPIE)
- Isotonic Regression (netcal)

**Impact on factorial**: Recalibration factor (#5) currently has 2 levels (none, temperature_scaling).
If more calibration methods are added, the factorial expands.

**Human input needed?**: PROBABLY YES — which calibration methods to include as factorial levels
for the publication. Currently defaulting to {none, temperature_scaling} for debug.

### 2.2 Topology Metrics (Not Started, 3-Way Tie)

**Decision**: `knowledge-graph/decisions/L3-technology/topology_metrics.yaml`
**Status**: `not_started`
**Options (roughly equal)**:
- Betti Numbers via GUDHI (0.35)
- Skeleton Precision/Recall (0.30)
- clDice as Metric (0.25)
- None (0.10)

**Impact**: These would be additional metrics for biostatistics. clDice IS already
a primary metric, but Betti errors (BE₀, BE₁) are listed as secondary.

**Human input needed?**: PROBABLY NOT for debug — BE₀, BE₁ already in secondary metrics.
GUDHI integration is future work. Not blocking.

### 2.3 API Protocol (Nearly Tied)

**Decision**: `knowledge-graph/decisions/L2-architecture/api_protocol.yaml`
**Options**:
- REST (0.45)
- gRPC (0.40)
- GraphQL (0.10)

**Impact on factorial**: NONE — this is infrastructure, not an experimental factor.

**Human input needed?**: NO for debug run. Defer to deployment phase.

### 2.4 Portfolio Role / Target (4-Way Tie, 0.25 each)

**Decision**: `knowledge-graph/decisions/L1-research-goals/portfolio_role_target.yaml`
**Status**: `partial`
**All options at 0.25** — completely unresolved.

**Impact**: Affects long-term project direction but NOT the factorial experiment.

**Human input needed?**: NOT for debug run. Strategic decision for post-publication.

### 2.5 Model Diagnostics (3-Way Near-Tie)

**Decision**: `knowledge-graph/decisions/L3-technology/model_diagnostics.yaml`
**Options**:
- Deepchecks Vision (0.30)
- WeightWatcher Spectral (0.30)
- Both (0.35)

**Impact**: Post-training flow has hooks for diagnostics but they're not wired yet.
NOT a factorial factor — diagnostics are observability, not experiment conditions.

**Human input needed?**: NOT for debug run.

---

## TIER 3: Implementation Gaps (Code Doesn't Match KG/Plans)

### 3.1 train_flow.py Bypasses Hydra (B1)

**KG says**: `compose_experiment_config()` should produce resolved config.
**Code does**: argparse + 9-key dict. `log_hydra_config()` never called.
**Workaround**: Factorial SkyPilot YAML passes all args explicitly.

**Confidence**: LOW that the workaround is sufficient for all 720 conditions.

### 3.2 Loss Function Tag Mismatch (B3)

**KG says**: Tag name should be `loss_function`.
**Training code logs**: `loss_name` (sometimes).
**Ensemble builder reads**: `loss_function`, falls back to `loss_name`, then `loss_type`.

**Confidence**: MEDIUM — fallback works but is fragile.

### 3.3 Cloud Artifact Sync (Issue #882)

**Plan says**: Local flows need cloud artifacts.
**Implementation**: `make sync-cloud-artifacts` doesn't exist yet.
**Workaround**: Manual `gsutil rsync` or `make dev-gpu-sync` (RunPod only).

**Confidence**: LOW that local post-training/analysis/biostatistics will work without this.

### 3.4 Post-Training Volume Mount (B7)

**Docker compose**: Missing `post_training_out` volume.
**Impact**: Container exits → artifacts lost.

**Confidence**: LOW for Docker execution (but local MINIVESS_ALLOW_HOST=1 bypasses this).

---

## TIER 4: Not Blocking, But Worth Noting

### 4.1 VLM Calibration (`not_started`)
Future work for SAM3-specific calibration (SaE, CalibPrompt). Not in current factorial.

### 4.2 Federated Learning (`not_started`)
Not in scope for Nature Protocols paper.

### 4.3 Air Gap Strategy (`not_started`)
Offline deployment. Not in scope.

### 4.4 GitOps Engine (`not_started`)
ArgoCD vs FluxCD. Not in scope for factorial experiment.

### 4.5 Retraining Trigger (`not_started`)
Drift-based retraining. Post-publication feature.

---

## Summary: All Low-Confidence Decisions RESOLVED (2026-03-20)

| # | Topic | Decision | Resolution |
|---|-------|----------|------------|
| 1 | Recalibration factor levels | `{none, temperature_scaling}` only (2 levels) | User decided — keeps factorial at 720/384 |
| 2 | Multi-SWA MLflow run structure | N separate MLflow runs | User decided — each sub-model individually trackable |
| 3 | Model soup alpha sweep | Optimal-only (both debug + full) | User decided — not a continuous sweep |
| 4 | Tag mismatch (B3) | Already correct in code; guard test added | Verified: train_flow.py line 953 uses `loss_function`. `test_loss_tag_consistency.py` prevents regression. |

**All low-confidence regions that needed human input are now RESOLVED.**
The remaining implementation gaps (B1 Hydra bypass, B7 volume mount, Issue #882 sync)
are pure coding work that does not require user decisions.
