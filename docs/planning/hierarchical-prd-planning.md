# Hierarchical Probabilistic PRD — Planning Document

> **Status:** Draft v1 — 2026-02-23
> **Scope:** Design blueprint for the MinIVess MLOps v2 probabilistic PRD system
> **Adapted from:** music-attribution-scaffold probabilistic PRD (85 nodes, 5 levels)
> **Target:** 52 decision nodes across 5 hierarchy levels

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Adapted Format](#2-adapted-format)
3. [Decision Network Overview](#3-decision-network-overview)
4. [Bayesian Network Semantics](#4-bayesian-network-semantics)
5. [Archetype Profiles](#5-archetype-profiles)
6. [Scenario Composition](#6-scenario-composition)
7. [Domain Overlay System](#7-domain-overlay-system)
8. [Maintenance Workflow](#8-maintenance-workflow)
9. [Validation Rules](#9-validation-rules)

---

## 1. Introduction

### 1.1 Why a Probabilistic PRD for This Project?

MinIVess MLOps v2 serves three simultaneous goals that create irreducible tension in
technology selection:

1. **Self-learning vehicle** — Maximize exposure to production-grade tools (40+ tools across
   12 validation layers). The "best" technology is often the one that teaches the most, not
   the one that ships fastest.

2. **Portfolio project** — Demonstrate competency for specific roles (ICEYE: air-gapped
   deployment + MCP; Cohere: production agents + evaluation; Curio: end-to-end MLOps
   curriculum). Different roles pull toward different technology emphasis.

3. **Research scaffold** — Produce two manuscripts (STAR Protocols + Medical Image Analysis)
   requiring reproducible benchmarks, ablation studies, and defensible architecture decisions.

A traditional PRD would force premature commitment. The probabilistic format encodes
uncertainty as a first-class concept: every technology decision carries a prior probability
reflecting current evidence, and these probabilities update as new papers, benchmarks, and
implementation experience arrive.

### 1.2 Relationship to the Modernization Plan

The 60KB modernization plan (`docs/modernize-minivess-mlops-plan.md`) is a comprehensive
narrative document covering architecture, phases, and tooling. The probabilistic PRD
**does not replace it** — it provides a complementary structured representation:

| Modernization Plan | Probabilistic PRD |
|---|---|
| Narrative, sequential | Graph-structured, non-sequential |
| Single recommended path | Multiple weighted paths |
| Implicit trade-offs | Explicit conditional probabilities |
| Updated by rewriting prose | Updated by adjusting probabilities |
| Human-readable only | Machine-readable (YAML) + human-readable |

The PRD is the "living" version: when a new paper shifts the landscape (e.g., SAMv4 release),
you update one decision node's probabilities rather than rewriting paragraphs of prose.

### 1.3 Current Implementation Status

Phases 0-6 are complete (103 tests passing). The PRD captures both **resolved** decisions
(already implemented) and **open** decisions (for future work beyond the current Phase 6
completion). Resolved decisions have collapsed probabilities (one option at ~1.0), while
open decisions maintain active probability distributions.

---

## 2. Adapted Format

### 2.1 Differences from the Music-Attribution-Scaffold Original

The music-attribution-scaffold PRD was designed for a greenfield SaaS product with
business-focused decisions (revenue model, target market, build-vs-buy). MinIVess v2 is
a research/portfolio project with fundamentally different decision drivers. Key adaptations:

| Aspect | Music-Attribution Original | MinIVess Adaptation |
|--------|---|---|
| **L1 level name** | L1-business | L1-research-goals |
| **L1 decisions** | Revenue model, target market, regulatory | Project purpose, impact targets, MONAI alignment |
| **Decision count** | 85 nodes | 52 nodes (more focused scope) |
| **Domain overlays** | Music attribution, DPP traceability | Vascular segmentation, cardiac, neuro, general medical |
| **Archetypes** | Solo hacker, engineer-heavy, musician-first | Solo researcher, lab group, clinical deployment |
| **Scenario emphasis** | Business models | Learning paths and role-targeted builds |
| **Skip connections** | L1 build-vs-buy to L4 compute | L1 MONAI alignment to L3 export format, L5 federated |
| **Volatility drivers** | Market dynamics, funding | Paper releases, library versions, GPU availability |

### 2.2 Schema Reuse

The decision node schema (`_schema.yaml`) is reused verbatim from music-attribution-scaffold.
The schema is domain-agnostic — it defines options, conditional probability tables, archetype
modulation, volatility classification, and domain applicability. Only the concrete decision
content changes.

### 2.3 Decision Level Relabeling

| Level | Music-Attribution | MinIVess v2 |
|-------|---|---|
| L1 | L1_business | L1_research_goals |
| L2 | L2_architecture | L2_architecture (unchanged) |
| L3 | L3_implementation | L3_technology |
| L4 | L4_deployment | L4_infrastructure |
| L5 | L5_operations | L5_operations (unchanged) |

The schema `enum` for `decision_level` should be updated to reflect these names, though
the underlying semantics (strategic to operational, high to low) remain identical.

---

## 3. Decision Network Overview

The network contains 52 decision nodes organized across 5 levels. Below is the complete
decision inventory with options and prior probabilities for the default (solo researcher)
archetype.

### 3.1 L1 — Research Goals (7 nodes, root layer)

These are root nodes with no parents. They represent the strategic context that conditions
all downstream decisions.

| # | Decision ID | Title | Options (prior) | Status |
|---|---|---|---|---|
| 1 | `project_purpose` | Primary Project Purpose | `self_learning` (0.50), `portfolio` (0.35), `research_publication` (0.15) | resolved |
| 2 | `impact_target` | Impact Measurement Target | `tool_breadth` (0.40), `benchmark_depth` (0.30), `manuscript_quality` (0.20), `deployment_readiness` (0.10) | resolved |
| 3 | `monai_alignment` | MONAI Ecosystem Alignment | `monai_native` (0.55), `monai_plus_external` (0.35), `framework_agnostic` (0.10) | resolved |
| 4 | `model_philosophy` | Model Strategy Philosophy | `model_agnostic_platform` (0.60), `foundation_model_first` (0.25), `ensemble_first` (0.15) | resolved |
| 5 | `compliance_posture` | Compliance Rigor | `samd_principled` (0.50), `full_iec_62304` (0.20), `research_only` (0.30) | resolved |
| 6 | `reproducibility_level` | Reproducibility Target | `bit_reproducible` (0.15), `statistically_reproducible` (0.55), `conceptually_reproducible` (0.30) | resolved |
| 7 | `portfolio_role_target` | Portfolio Role Priority | `iceye_forward_deployed` (0.30), `cohere_applied_ai` (0.30), `curio_mlops_sme` (0.25), `general_ml_engineer` (0.15) | partial |

### 3.2 L2 — Architecture (10 nodes)

Architecture decisions are conditioned on L1 strategic choices. These define the system
shape without committing to specific tools.

| # | Decision ID | Title | Options (prior) | Status |
|---|---|---|---|---|
| 8 | `model_adapter_pattern` | Model Integration Pattern | `abc_protocol` (0.70), `plugin_registry` (0.20), `duck_typing` (0.10) | resolved |
| 9 | `ensemble_strategy` | Ensemble Architecture | `heterogeneous_multi_model` (0.40), `homogeneous_multi_seed` (0.30), `model_soups_only` (0.20), `single_model_baseline` (0.10) | resolved |
| 10 | `uncertainty_framework` | Uncertainty Quantification Approach | `conformal_prediction` (0.40), `mc_dropout` (0.25), `deep_ensemble_disagreement` (0.20), `swag_bayesian` (0.15) | resolved |
| 11 | `config_architecture` | Configuration System Design | `dual_config` (0.65), `hydra_only` (0.20), `dynaconf_only` (0.15) | resolved |
| 12 | `serving_architecture` | Model Serving Architecture | `bentoml_primary` (0.50), `torchserve` (0.20), `monai_deploy_map` (0.20), `custom_fastapi` (0.10) | resolved |
| 13 | `xai_strategy` | Explainability Strategy | `multi_layer_xai` (0.55), `gradcam_only` (0.25), `uncertainty_as_xai` (0.15), `no_xai` (0.05) | partial |
| 14 | `data_validation_depth` | Data Validation Strategy | `validation_onion_12_layer` (0.45), `schema_plus_drift` (0.35), `schema_only` (0.15), `no_validation` (0.05) | resolved |
| 15 | `agent_architecture` | Agent Orchestration Approach | `langgraph_stateful` (0.55), `langchain_sequential` (0.20), `custom_state_machine` (0.15), `no_agents` (0.10) | resolved |
| 16 | `observability_depth` | Observability Stack Depth | `full_stack_otel` (0.45), `metrics_plus_traces` (0.30), `logging_only` (0.20), `no_observability` (0.05) | resolved |
| 17 | `api_protocol` | API and Interface Protocol | `rest_grpc_dual` (0.40), `rest_only` (0.35), `grpc_only` (0.15), `graphql` (0.10) | config_only |

### 3.3 L3 — Technology (20 nodes)

Specific tool and library selections. These are the most volatile decisions, as new
releases and benchmarks constantly shift the landscape.

| # | Decision ID | Title | Options (prior) | Status |
|---|---|---|---|---|
| 18 | `primary_3d_model` | Primary 3D Segmentation Model | `vista3d` (0.45), `swinunetr` (0.30), `segresnet` (0.20), `unet_monai` (0.05) | resolved |
| 19 | `foundation_model` | Foundation Model for Adaptation | `sam3_lora` (0.35), `medsam3_adapter` (0.25), `med_persam` (0.20), `biomedclip` (0.10), `none` (0.10) | partial |
| 20 | `loss_function` | Primary Loss Function | `dice_ce_combined` (0.40), `cldice_soft_skeleton` (0.30), `generalized_dice` (0.20), `focal_tversky` (0.10) | resolved |
| 21 | `primary_metrics` | Primary Evaluation Metrics | `metricsreloaded_full` (0.50), `torchmetrics_dice_f1` (0.30), `cldice_nsd_hd95` (0.15), `dice_only` (0.05) | resolved |
| 22 | `topology_metrics` | Topology-Aware Evaluation | `betti_gudhi` (0.35), `skeleton_precision_recall` (0.30), `cldice_as_metric` (0.25), `none` (0.10) | not_started |
| 23 | `experiment_tracker` | Experiment Tracking Tool | `mlflow_local` (0.60), `mlflow_plus_wandb` (0.20), `wandb_only` (0.10), `tensorboard_only` (0.10) | resolved |
| 24 | `hpo_engine` | Hyperparameter Optimization | `optuna_multi_objective` (0.55), `optuna_single` (0.25), `ray_tune` (0.15), `manual_grid` (0.05) | resolved |
| 25 | `data_versioning` | Data Versioning Tool | `dvc_local_minio` (0.55), `dvc_s3` (0.25), `lakefs` (0.10), `manual_hashing` (0.10) | resolved |
| 26 | `augmentation_library` | Augmentation Strategy | `torchio_monai_combined` (0.50), `monai_transforms_only` (0.30), `albumentations3d` (0.10), `custom_only` (0.10) | resolved |
| 27 | `calibration_method` | Calibration Approach | `temperature_scaling_netcal` (0.35), `local_temperature_scaling` (0.25), `mondrian_conformal_mapie` (0.25), `isotonic_netcal` (0.15) | partial |
| 28 | `label_quality_tool` | Label Quality Assessment | `cleanlab` (0.40), `label_studio_agreement` (0.30), `custom_topology_checks` (0.20), `none` (0.10) | config_only |
| 29 | `data_profiling` | Data Profiling Tool | `whylogs` (0.45), `great_expectations` (0.30), `pandera_only` (0.20), `none` (0.05) | config_only |
| 30 | `model_diagnostics` | Model Diagnostic Tool | `deepchecks_vision` (0.35), `weightwatcher_spectral` (0.30), `both_deepchecks_ww` (0.30), `none` (0.05) | resolved |
| 31 | `xai_voxel_tool` | Voxel-Level XAI Tool | `captum_gradcam` (0.45), `captum_integrated_gradients` (0.30), `monai_occlusion` (0.15), `none` (0.10) | config_only |
| 32 | `xai_meta_tool` | Meta-Decision XAI Tool | `shap_tabular` (0.40), `quantus_meta_eval` (0.25), `both_shap_quantus` (0.25), `none` (0.10) | config_only |
| 33 | `llm_tracing` | LLM Observability Tool | `langfuse_self_hosted` (0.50), `langfuse_plus_braintrust` (0.30), `braintrust_only` (0.10), `none` (0.10) | resolved |
| 34 | `llm_evaluation` | LLM Evaluation Framework | `braintrust_autoevals` (0.45), `trulens_rag` (0.15), `custom_scorers` (0.25), `none` (0.15) | resolved |
| 35 | `llm_provider_strategy` | LLM Provider Abstraction | `litellm_multi_provider` (0.50), `anthropic_direct` (0.25), `ollama_local_only` (0.15), `no_llm` (0.10) | resolved |
| 36 | `dataframe_validation` | DataFrame Validation Tool | `pandera_schemas` (0.45), `great_expectations_suites` (0.30), `pydantic_only` (0.20), `none` (0.05) | resolved |
| 37 | `lineage_tracking` | Data Lineage Tool | `openlineage_marquez` (0.45), `mlflow_lineage` (0.30), `custom_audit_log` (0.20), `none` (0.05) | config_only |

### 3.4 L4 — Infrastructure (8 nodes)

Infrastructure decisions covering compute, containers, CI/CD, and deployment artifacts.

| # | Decision ID | Title | Options (prior) | Status |
|---|---|---|---|---|
| 38 | `container_strategy` | Container Orchestration | `docker_compose_profiles` (0.55), `k8s_kustomize` (0.20), `k3d_local_k8s` (0.15), `podman` (0.10) | resolved |
| 39 | `ci_cd_platform` | CI/CD Platform | `github_actions_cml` (0.60), `github_actions_only` (0.25), `gitlab_ci` (0.10), `local_just_only` (0.05) | resolved |
| 40 | `iac_tool` | Infrastructure as Code | `pulumi_python` (0.40), `terraform` (0.25), `helm_charts` (0.20), `kustomize_only` (0.10), `none` (0.05) | not_started |
| 41 | `gitops_engine` | GitOps Deployment Engine | `argocd` (0.45), `fluxcd` (0.20), `github_actions_deploy` (0.25), `manual_kubectl` (0.10) | not_started |
| 42 | `model_export_format` | Model Export Format | `onnx_runtime` (0.40), `torchscript` (0.25), `monai_bundle` (0.25), `raw_pytorch` (0.10) | resolved |
| 43 | `gpu_compute` | GPU Compute Strategy | `local_gpu` (0.35), `cloud_spot_instances` (0.25), `lambda_labs` (0.15), `colab_pro` (0.15), `cpu_only` (0.10) | partial |
| 44 | `secrets_management` | Secrets Management | `dynaconf_dotenv` (0.40), `sealed_secrets_k8s` (0.25), `vault_hashicorp` (0.15), `env_vars_only` (0.15), `none` (0.05) | resolved |
| 45 | `air_gap_strategy` | Air-Gapped Deployment | `offline_docker_bundle` (0.45), `helm_oci_vendored` (0.25), `not_applicable` (0.20), `custom_tarball` (0.10) | not_started |

### 3.5 L5 — Operations (7 nodes)

Runtime operational decisions covering monitoring, drift, governance, and retraining.

| # | Decision ID | Title | Options (prior) | Status |
|---|---|---|---|---|
| 46 | `drift_monitoring` | Drift Detection Strategy | `evidently_lib` (0.45), `evidently_plus_whylogs` (0.25), `custom_ks_test` (0.20), `none` (0.10) | resolved |
| 47 | `dashboarding` | Monitoring Dashboard | `grafana_prometheus` (0.55), `mlflow_ui_only` (0.25), `custom_streamlit` (0.10), `none` (0.10) | config_only |
| 48 | `retraining_trigger` | Retraining Trigger Mechanism | `drift_alert_webhook` (0.35), `scheduled_periodic` (0.25), `manual_only` (0.30), `active_learning_loop` (0.10) | not_started |
| 49 | `model_governance` | Model Governance Approach | `mlflow_registry_gates` (0.40), `custom_approval_flow` (0.25), `weightwatcher_gate` (0.20), `none` (0.15) | partial |
| 50 | `audit_trail` | Audit Trail Implementation | `structured_json_otel` (0.50), `openlineage_marquez` (0.25), `mlflow_tags_only` (0.15), `none` (0.10) | resolved |
| 51 | `sbom_generation` | SBOM and Supply Chain | `syft_uv_lock` (0.45), `cyclonedx` (0.25), `manual_lockfile` (0.20), `none` (0.10) | not_started |
| 52 | `federated_learning` | Federated Learning Strategy | `monai_fl` (0.30), `flower_framework` (0.20), `custom_aggregation` (0.15), `not_applicable` (0.35) | not_started |

---

## 4. Bayesian Network Semantics

### 4.1 Prior Probabilities

Each decision node has 2-5 options with prior probabilities that sum to 1.0. These priors
represent the **unconditional** likelihood of choosing each option before considering:
- Parent decision outcomes (conditional probabilities)
- Team archetype (archetype modulation)
- Domain specifics (domain overlay)

Priors are calibrated from three sources:
1. **Implementation evidence** — What is already built in the codebase
2. **Modernization plan consensus** — Three-reviewer convergence from plan draft v2
3. **Ecosystem maturity** — Library stability, community size, documentation quality

### 4.2 Conditional Probability Tables (CPTs)

Edges in the DAG encode conditional dependencies. When a parent decision is resolved,
it shifts the probability distribution of child decisions via a CPT.

**Example:** The `monai_alignment` (L1) decision conditions several L3 technology choices:

```yaml
# In decisions/L3-technology/model-export-format.decision.yaml
conditional_on:
  - parent_decision_id: monai_alignment
    influence_strength: strong
    conditional_table:
      - given_parent_option: monai_native
        then_probabilities:
          onnx_runtime: 0.20
          torchscript: 0.15
          monai_bundle: 0.55
          raw_pytorch: 0.10
      - given_parent_option: monai_plus_external
        then_probabilities:
          onnx_runtime: 0.45
          torchscript: 0.25
          monai_bundle: 0.20
          raw_pytorch: 0.10
      - given_parent_option: framework_agnostic
        then_probabilities:
          onnx_runtime: 0.55
          torchscript: 0.25
          monai_bundle: 0.05
          raw_pytorch: 0.15
```

### 4.3 Influence Strength

Each conditional dependency is annotated with a qualitative influence strength:

| Strength | Meaning | Typical Edge Pattern |
|---|---|---|
| **strong** | Parent choice nearly determines this decision | L1 monai_alignment to L3 export format |
| **moderate** | Parent choice significantly shifts probabilities | L2 ensemble_strategy to L3 calibration_method |
| **weak** | Parent choice slightly adjusts probabilities | L1 compliance_posture to L5 sbom_generation |

### 4.4 Edge Taxonomy

The network has approximately 80 edges, categorized as:

| Edge Type | Count | Example |
|---|---|---|
| L1 to L2 (strategy to architecture) | ~15 | `project_purpose` to `observability_depth` |
| L2 to L3 (architecture to technology) | ~30 | `ensemble_strategy` to `calibration_method` |
| L3 to L4 (technology to infrastructure) | ~15 | `experiment_tracker` to `container_strategy` |
| L4 to L5 (infrastructure to operations) | ~10 | `gitops_engine` to `retraining_trigger` |
| Skip connections (L1 to L3/L4/L5) | ~10 | `monai_alignment` to `model_export_format` |

### 4.5 Posterior Computation

Given observed evidence (resolved decisions), the posterior probability of an unresolved
decision is computed by combining:

1. **Prior** P(option) from the decision node
2. **Conditional updates** P(option | parent_option) for each resolved parent
3. **Archetype modulation** P_archetype(option) if a team profile is active
4. **Domain overlay** adjustment factor from the active domain

For simplicity, we use **noisy-OR combination** rather than full Bayesian inference:
the posterior is computed as the weighted geometric mean of the prior and all applicable
conditional rows, normalized to sum to 1.0. This avoids the computational complexity of
exact inference in large Bayesian networks while preserving the directional influence
of parent decisions.

---

## 5. Archetype Profiles

### 5.1 Solo Researcher

**File:** `archetypes/solo-researcher.archetype.yaml`

The default archetype representing the current project context. A single researcher
building a portfolio project with emphasis on learning and tool exposure.

| Characteristic | Value |
|---|---|
| Team size | 1 |
| Budget | < $100/month cloud spend |
| GPU access | Single consumer GPU (RTX 3080/4090) or Colab Pro |
| Primary constraint | Time (one person doing everything) |
| Tool preference | Maximize breadth over depth |
| Deployment target | Docker Compose on local machine |
| Compliance need | SaMD-principled (demonstrate, not certify) |

**Key probability shifts vs. default priors:**
- `container_strategy`: boosts `docker_compose_profiles` to 0.70 (no K8s overhead)
- `gpu_compute`: boosts `local_gpu` to 0.50 (minimize cloud costs)
- `retraining_trigger`: boosts `manual_only` to 0.45 (no auto-triggers needed)
- `federated_learning`: boosts `not_applicable` to 0.55 (solo, no federation)

### 5.2 Lab Group

**File:** `archetypes/lab-group.archetype.yaml`

A small academic research group (3-8 people) with shared compute infrastructure and
collaborative annotation workflows.

| Characteristic | Value |
|---|---|
| Team size | 3-8 |
| Budget | University compute cluster + some cloud |
| GPU access | Shared DGX or multi-GPU cluster |
| Primary constraint | Coordination (multi-annotator, shared experiments) |
| Tool preference | Balance reproducibility with collaboration |
| Deployment target | Docker Compose to shared server, optional K8s |
| Compliance need | Research reproducibility, not regulatory |

**Key probability shifts vs. default priors:**
- `label_quality_tool`: boosts `label_studio_agreement` to 0.50 (multi-annotator)
- `experiment_tracker`: boosts `mlflow_plus_wandb` to 0.35 (team visibility)
- `data_versioning`: boosts `dvc_s3` to 0.40 (shared remote)
- `retraining_trigger`: boosts `scheduled_periodic` to 0.35 (nightly re-runs)

### 5.3 Clinical Deployment

**File:** `archetypes/clinical-deployment.archetype.yaml`

A clinical AI team building toward regulatory submission. Compliance is non-negotiable,
and every design decision must be defensible to auditors.

| Characteristic | Value |
|---|---|
| Team size | 5-20 (including regulatory, QA) |
| Budget | $5K-50K/month infrastructure |
| GPU access | Cloud GPU instances with SLA |
| Primary constraint | Regulatory compliance (IEC 62304, ISO 13485) |
| Tool preference | Stability, auditability, vendor support |
| Deployment target | Kubernetes with full observability |
| Compliance need | Full IEC 62304 + ISO 14971 |

**Key probability shifts vs. default priors:**
- `compliance_posture`: boosts `full_iec_62304` to 0.65
- `container_strategy`: boosts `k8s_kustomize` to 0.50
- `audit_trail`: boosts `structured_json_otel` to 0.65
- `sbom_generation`: boosts `syft_uv_lock` to 0.60
- `model_governance`: boosts `custom_approval_flow` to 0.45
- `air_gap_strategy`: boosts `offline_docker_bundle` to 0.55
- `federated_learning`: boosts `monai_fl` to 0.45

---

## 6. Scenario Composition

### 6.1 What is a Scenario?

A scenario is a **fully resolved path** through the decision network: every decision node
is collapsed to exactly one option. The scenario is valid if and only if:

1. All 52 decisions are resolved to exactly one option
2. No hard constraints are violated (e.g., choosing `k8s_kustomize` but `iac_tool: none`)
3. The joint probability is above a minimum threshold (0.0001)

### 6.2 Joint Probability

The joint probability of a scenario is the product of all selected option probabilities
(using posterior probabilities that account for conditional dependencies):

```
P(scenario) = Product_i [ P(option_i | parents_i, archetype) ]
```

In practice, joint probabilities for 52-node networks are vanishingly small in absolute
terms. The meaningful comparison is **relative joint probability** between scenarios:
which path through the network is most consistent with the evidence?

### 6.3 Active Scenario: Learning-First MVP

**File:** `scenarios/learning-first-mvp.scenario.yaml`

This scenario represents the current active implementation (Phases 0-6 complete).
It resolves approximately 33 of 52 decisions, leaving 19 for future work.

**Key resolved choices:**
- `project_purpose`: `self_learning`
- `monai_alignment`: `monai_plus_external`
- `model_adapter_pattern`: `abc_protocol`
- `primary_3d_model`: `vista3d` (primary), `swinunetr` and `segresnet` as implemented alternatives
- `config_architecture`: `dual_config` (Hydra-zen + Dynaconf)
- `experiment_tracker`: `mlflow_local`
- `container_strategy`: `docker_compose_profiles`
- `ci_cd_platform`: `github_actions_cml`
- `agent_architecture`: `langgraph_stateful`
- `llm_tracing`: `langfuse_self_hosted`

### 6.4 Planned Scenarios

**Research Scaffold** (`scenarios/research-scaffold.scenario.yaml`):
Optimized for manuscript production. Emphasizes reproducibility, ablation coverage,
and benchmark completeness over tool breadth.

**Clinical Production** (`scenarios/clinical-production.scenario.yaml`):
Full compliance path. Resolves all compliance, governance, and air-gap decisions
for a hypothetical regulatory submission.

---

## 7. Domain Overlay System

### 7.1 Purpose

Domain overlays adjust decision probabilities and add domain-specific metadata without
modifying the core decision network. The same 52-node network serves vascular
segmentation, cardiac imaging, neuroimaging, and general medical imaging.

### 7.2 Registry

**File:** `domains/registry.yaml`

Lists all registered domains with their active/inactive status and the overlay file path.

### 7.3 Active Domain: Vascular Segmentation

**File:** `domains/vascular-segmentation/overlay.yaml`

The vascular segmentation overlay adjusts the following decisions:

| Decision | Overlay Effect |
|---|---|
| `loss_function` | Boosts `cldice_soft_skeleton` to 0.50 (topology preservation) |
| `topology_metrics` | Boosts `betti_gudhi` to 0.45 (vessel connectivity) |
| `primary_metrics` | Boosts `cldice_nsd_hd95` to 0.35 (vessel-specific) |
| `augmentation_library` | Boosts `torchio_monai_combined` to 0.60 (anisotropy handling) |
| `calibration_method` | Boosts `mondrian_conformal_mapie` to 0.35 (boundary vs. interior) |
| `primary_3d_model` | No change (VISTA-3D is appropriate for vessels) |

**Domain-specific metadata added by the overlay:**
- Preferred patch size: (128, 128, 32) instead of default (96, 96, 96)
- Anisotropy-aware spacing normalization required
- Foreground-aware intensity normalization (`nonzero=True`)
- Minimum connected component filtering for post-processing

### 7.4 Planned Domains

**Cardiac Imaging** (`domains/cardiac-imaging/overlay.yaml`):
- Boosts temporal augmentations (cardiac motion)
- Prefers `swinunetr` for cardiac structures
- Adds ACDC benchmark metrics

**Neuroimaging** (`domains/neuroimaging/overlay.yaml`):
- Boosts `foundation_model: biomedclip` for brain region classification
- Prefers FreeSurfer-compatible output formats
- Adds Dice-based atlas comparison metrics

**General Medical** (`domains/general-medical/overlay.yaml`):
- Neutral overlay (no strong adjustments)
- Serves as the "backbone" for new domains

### 7.5 Backbone Defaults

**File:** `domains/backbone-defaults.yaml`

Contains the base probability distribution that all overlays modify. If a domain overlay
does not mention a decision, the backbone default applies. This ensures that domain
overlays are sparse (only specifying what differs from the default) rather than
exhaustive.

---

## 8. Maintenance Workflow

### 8.1 PRD-Update Skill

The PRD is maintained using the `prd-update` Claude Code skill at
`.claude/skills/prd-update/`. This skill provides structured protocols for common
maintenance operations:

| Operation | Protocol | Input | Output |
|---|---|---|---|
| `add-decision` | Create a new decision node | Title, level, options, priors | New `.decision.yaml` + updated `_network.yaml` |
| `update-priors` | Adjust probabilities based on evidence | Decision ID, new priors, rationale | Updated `.decision.yaml` |
| `add-option` | Add option to existing decision | Decision ID, option details | Updated `.decision.yaml` |
| `create-scenario` | Compose a new scenario | Name, archetype, decision overrides | New `.scenario.yaml` |
| `ingest-paper` | Extract decisions from a paper | Paper summary or BibTeX | One or more `update-priors` operations |
| `validate` | Check all invariants | None | Pass/fail report |

### 8.2 Evidence Ingestion Protocol

When a new paper, benchmark, or tool release arrives:

1. **Identify affected decisions** — Which of the 52 decisions does this evidence
   bear on? A SAMv4 release affects `foundation_model`; a new MAPIE version affects
   `calibration_method` and `uncertainty_framework`.

2. **Quantify the shift** — How much should priors change? Use the following heuristic:
   - Published benchmark with clear winner: shift 0.10-0.20
   - New tool release (stable API): shift 0.05-0.15
   - Blog post or preprint: shift 0.02-0.05
   - Personal experiment result: shift 0.05-0.10

3. **Update priors** — Adjust the affected option probabilities, ensuring they still
   sum to 1.0.

4. **Update volatility** — If the evidence indicates a classification change (e.g.,
   stable to shifting), update the `volatility` block.

5. **Propagate conditionals** — If the updated decision is a parent, review whether
   conditional tables for children should also change.

6. **Run validation** — Execute `validate` to check all invariants.

### 8.3 Volatility Review Schedule

| Classification | Review Frequency | Typical Decisions |
|---|---|---|
| **stable** | Every 6 months | `model_adapter_pattern`, `ci_cd_platform`, `data_versioning` |
| **shifting** | Every 3 months | `foundation_model`, `llm_provider_strategy`, `xai_voxel_tool` |
| **volatile** | Every 2 weeks | `primary_3d_model` (if new foundation models releasing), `llm_evaluation` |

### 8.4 Resolved Decision Maintenance

Even resolved decisions (already implemented) should be reviewed if new evidence suggests
a better option. A resolved decision with a high-probability alternative (>0.30) in a
non-selected option is a candidate for refactoring. The decision's status can be changed
from `resolved` back to `partial` to signal this.

---

## 9. Validation Rules

### 9.1 Probability Invariants

All of these must hold across the entire PRD:

1. **Prior probability sum**: For every decision node, the sum of all option
   `prior_probability` values must equal 1.0 (tolerance: 0.01).

2. **Conditional table row sum**: For every row in every conditional probability table,
   the `then_probabilities` values must sum to 1.0 (tolerance: 0.01).

3. **Conditional table completeness**: Every conditional table row's `then_probabilities`
   must contain keys for all options in the current decision. No option may be omitted.

4. **Conditional parent coverage**: Every conditional table must have one row per option
   in the parent decision. If the parent has 4 options, the conditional table must have
   4 rows.

5. **Archetype override sum**: For every archetype modifier, the
   `probability_overrides` must sum to 1.0 (tolerance: 0.01) and must cover all
   options in the decision.

### 9.2 Graph Invariants

6. **DAG acyclicity**: The `_network.yaml` graph must contain no directed cycles. A
   topological sort must succeed.

7. **Referential integrity (parent)**: Every `parent_decision_id` in a conditional
   dependency must reference an existing node in `_network.yaml`.

8. **Referential integrity (option)**: Every option ID referenced in conditional tables,
   archetype overrides, scenario selections, and domain overlays must exist in the
   corresponding decision node's `options` list.

9. **Level ordering**: Edges must flow from lower-numbered levels to higher-numbered
   levels (L1 to L2, L2 to L3, etc.) with the exception of explicitly documented
   skip connections. No edge may flow upward (L3 to L1).

### 9.3 Scenario Invariants

10. **Scenario completeness**: A scenario marked as `complete: true` must resolve every
    decision in the network. A `partial` scenario must resolve at least all L1 and L2
    decisions.

11. **Scenario consistency**: A scenario must not violate any hard constraints defined
    in the option `constraints` fields. For example, choosing `k8s_kustomize` for
    `container_strategy` requires `iac_tool` to not be `none`.

12. **Scenario archetype alignment**: If a scenario declares an archetype, the resolved
    options should be consistent with that archetype's probability overrides. A scenario
    claiming `clinical_deployment` archetype but resolving `compliance_posture` to
    `research_only` should trigger a warning.

### 9.4 Domain Overlay Invariants

13. **Overlay sparsity**: A domain overlay should override fewer than 50% of decisions.
    If it overrides more, it may be better modeled as a separate network.

14. **Overlay probability sums**: Any probability overrides in a domain overlay must
    still sum to 1.0 per decision.

15. **Domain registry consistency**: Every overlay file referenced in `registry.yaml`
    must exist on disk. Every overlay file on disk must be listed in `registry.yaml`.

### 9.5 Volatility Invariants

16. **Review date validity**: The `next_review` date must be in the future (at validation
    time). Overdue reviews trigger a warning.

17. **Classification consistency**: A `volatile` decision should have a `next_review`
    date within 4 weeks. A `stable` decision should have a `next_review` date at least
    3 months in the future.

### 9.6 Automated Validation

The `validate` operation in the prd-update skill runs all 17 invariants and produces
a report:

```
PRD Validation Report (2026-02-23)
===================================
Probability sums:     52/52 PASS
Conditional tables:   78/80 PASS (2 incomplete: foundation_model, topology_metrics)
Archetype coverage:   3/3 PASS
DAG acyclicity:       PASS
Referential integrity: PASS
Level ordering:       PASS (3 documented skip connections)
Scenarios:            1/1 complete scenarios valid
Domain overlays:      1/4 active, all valid
Volatility reviews:   3 overdue (next_review < today)

OVERALL: 2 warnings, 0 errors
```

---

## Appendix A: Edge Summary

### A.1 Major Conditional Dependencies

| Parent (L1) | Child | Strength | Effect |
|---|---|---|---|
| `monai_alignment` = monai_native | `model_export_format` | strong | Boosts `monai_bundle` to 0.55 |
| `monai_alignment` = monai_native | `federated_learning` | strong | Boosts `monai_fl` to 0.50 |
| `project_purpose` = self_learning | `observability_depth` | moderate | Boosts `full_stack_otel` to 0.55 |
| `project_purpose` = portfolio | `air_gap_strategy` | moderate | Boosts `offline_docker_bundle` to 0.55 |
| `compliance_posture` = full_iec_62304 | `audit_trail` | strong | Boosts `structured_json_otel` to 0.70 |
| `compliance_posture` = full_iec_62304 | `sbom_generation` | strong | Boosts `syft_uv_lock` to 0.60 |
| `model_philosophy` = ensemble_first | `ensemble_strategy` | strong | Boosts `heterogeneous_multi_model` to 0.55 |

| Parent (L2) | Child | Strength | Effect |
|---|---|---|---|
| `ensemble_strategy` = heterogeneous | `calibration_method` | moderate | Boosts `mondrian_conformal_mapie` to 0.35 |
| `uncertainty_framework` = conformal | `calibration_method` | strong | Boosts `mondrian_conformal_mapie` to 0.50 |
| `serving_architecture` = bentoml | `model_export_format` | moderate | Boosts `onnx_runtime` to 0.50 |
| `agent_architecture` = langgraph | `llm_tracing` | strong | Boosts `langfuse_self_hosted` to 0.60 |
| `agent_architecture` = langgraph | `llm_provider_strategy` | moderate | Boosts `litellm_multi_provider` to 0.55 |
| `xai_strategy` = multi_layer | `xai_voxel_tool` | strong | Boosts `captum_gradcam` to 0.55 |
| `xai_strategy` = multi_layer | `xai_meta_tool` | strong | Boosts `both_shap_quantus` to 0.45 |
| `data_validation_depth` = 12_layer | `dataframe_validation` | moderate | Boosts `pandera_schemas` to 0.55 |
| `data_validation_depth` = 12_layer | `data_profiling` | moderate | Boosts `whylogs` to 0.50 |

### A.2 Skip Connections

| Source | Target | Strength | Rationale |
|---|---|---|---|
| L1 `monai_alignment` | L3 `model_export_format` | strong | MONAI alignment directly determines export format preference |
| L1 `monai_alignment` | L5 `federated_learning` | strong | MONAI FL only makes sense with MONAI alignment |
| L1 `compliance_posture` | L5 `sbom_generation` | moderate | Compliance rigor directly drives supply chain requirements |

---

## Appendix B: Implementation Mapping

### B.1 Decisions Already Resolved in Codebase

The following decisions are implemented in the current codebase (Phases 0-6) and have
effectively collapsed priors (selected option probability approaching 1.0):

| Decision | Resolved To | Phase | Test Coverage |
|---|---|---|---|
| `model_adapter_pattern` | `abc_protocol` | P1 | ADR-0001, 48 adapter tests |
| `config_architecture` | `dual_config` | P0 | ADR-0002, Hypothesis tests |
| `data_validation_depth` | `validation_onion_12_layer` | P0 | ADR-0003 |
| `observability_depth` | `full_stack_otel` | P4 | ADR-0004, 15 tests |
| `primary_3d_model` | `vista3d` (primary) | P1 | SegResNet + SwinUNETR also implemented |
| `ensemble_strategy` | `heterogeneous_multi_model` | P2 | Voting, soup, WeightWatcher |
| `experiment_tracker` | `mlflow_local` | P1 | DuckDB analytics integration |
| `container_strategy` | `docker_compose_profiles` | P0 | 3 profiles (dev/monitoring/full) |
| `ci_cd_platform` | `github_actions_cml` | P0 | 8 workflow files |
| `agent_architecture` | `langgraph_stateful` | P5 | Training + evaluation graphs |
| `llm_tracing` | `langfuse_self_hosted` | P4 | Config in telemetry module |
| `llm_evaluation` | `braintrust_autoevals` | P5 | Scorer configs |
| `serving_architecture` | `bentoml_primary` | P3 | ONNX export + Gradio demo |
| `audit_trail` | `structured_json_otel` | P4 | IEC 62304 lifecycle hooks |
| `drift_monitoring` | `evidently_lib` | P2 | KS test + PSI detection |

### B.2 Decisions with Configuration but No Integration

These decisions have configuration files or dependency declarations but are not yet
functionally integrated into the pipeline:

| Decision | Configured As | Missing Integration |
|---|---|---|
| `label_quality_tool` | `cleanlab` in pyproject.toml | No actual label quality pipeline |
| `data_profiling` | `whylogs` in pyproject.toml | No profiling in data pipeline |
| `xai_voxel_tool` | `captum` in pyproject.toml | No XAI module in src/ |
| `xai_meta_tool` | `shap` + `quantus` in pyproject.toml | No meta-decision explanations |
| `lineage_tracking` | `marquez` in docker-compose.yml | No OpenLineage integration |
| `dashboarding` | `grafana` + `prometheus` in docker-compose.yml | No custom dashboards |

---

*Generated: 2026-02-23 | Based on modernize-minivess-mlops-plan.md (Draft v3) and Phase 0-6 implementation*
