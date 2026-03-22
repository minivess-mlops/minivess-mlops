---
theme: deployment
health_score: 55
doc_count: 5
created: "2026-03-22"
kg_domain: knowledge-graph/domains/architecture.yaml
archive_dir: docs/planning/v0-2_archive/original_docs
status: archived-with-synthesis
---

# Theme: Deployment

BentoML + ONNX Runtime serving, model registry with promotion stages, MONAI Deploy
clinical pathway, interactive segmentation for annotation proofreading, and the deploy
flow as the 4th core Prefect flow in the 5-flow architecture.

---

## Key Scientific Insights

1. **The deployment chain is MLflow to ONNX to BentoML, not PyTorch direct.** The
   serving architecture resolves a common academic ML gap: training in PyTorch but
   serving requires ONNX Runtime for deterministic, framework-agnostic inference.
   The chain is: champion model selected by compound metric (0.5*clDice +
   0.5*normalized_MASD) in MLflow Model Registry, exported to ONNX (opset 17),
   imported into BentoML model store, and served via REST API. The BentoML response
   includes the binary mask, calibrated voxel-level probabilities, and uncertainty
   intervals -- not just a segmentation mask. This full-probability output enables
   downstream consumers (annotation proofreading, clinical decision support) to apply
   their own thresholds.

2. **Model registry with promotion stages bridges research and deployment.** The
   four-stage promotion workflow (DEVELOPMENT to STAGING to PRODUCTION to ARCHIVED)
   maps to the academic publication lifecycle: experimental results (dev), internal
   validation (staging), paper submission / community serving (production), superseded
   models (archived). Promotion requires meeting metric thresholds
   (`PromotionCriteria`) and is audit-logged for reproducibility. This is
   deliberately lightweight compared to enterprise MLOps registries -- no governance
   committee, just metric gates.

3. **Interactive segmentation for proofreading reduces annotation burden by 30%+.**
   The research report evaluates 6 seed papers and 20+ methods for correcting
   AI-generated 3D segmentation masks. Key finding: nnInteractive (Isensee et al.,
   2025) with 3D Slicer integration via SlicerNNInteractive (de Vente et al., 2025)
   is the strongest immediately deployable solution. The prior mask input channel
   directly enables the proofreading workflow (AI init, then interactive correction).
   K-Prism (Guo et al., 2025) provides the strongest quantitative evidence:
   feeding an initial mask as prior reduces NoC90 (number of clicks to reach 90%
   Dice) by over 30% compared to interactive-only annotation.

4. **MONAI Deploy provides the clinical pathway but is not the primary serving
   mechanism.** For academic use (Nature Protocols paper, community serving), BentoML
   + ONNX Runtime is the serving stack. MONAI Deploy (MAP -- MONAI Application
   Package) is the FDA/clinical pathway: DICOM I/O, structured reports, and
   packaging for hospital PACS integration. The two coexist: BentoML for research
   deployment (REST API, Gradio demo, SkyServe spot instances), MONAI Deploy for
   clinical deployment (DICOM SCP, FHIR, IEC 62304 lifecycle stages). Neither
   replaces the other.

5. **Champion selection from factorial experiments requires compound metric ranking.**
   The deploy flow selects the best model from the factorial experiment results
   (4 models x 3 losses x 2 aux_calib = 24 conditions) using the compound metric
   (0.5*clDice + 0.5*normalize(MASD)). The champion can be a single checkpoint,
   a CV-average pyfunc (3 fold checkpoints), or a hierarchical ensemble pyfunc
   (multiple models). Champion metadata (model, loss, aux_calib, post_training,
   fold_strategy, ensemble type, per-metric scores) is tagged in MLflow Model
   Registry for full provenance.

---

## Architectural Decisions Made

| Decision | Winner | Rationale | KG Node |
|----------|--------|-----------|---------|
| Serving framework | BentoML + ONNX Runtime | Framework-agnostic inference, REST API, Gradio demo, SkyServe spot instances | `serving_architecture` |
| ONNX export strategy | Dynamo-first, legacy fallback | Try `torch.onnx.dynamo_export` first, fall back to `torch.onnx.export` | `code-review-report-v0-2-2nd-pass.md` |
| Model registry | MLflow Model Registry + 4-stage promotion | DEVELOPMENT to STAGING to PRODUCTION to ARCHIVED. Metric-gated promotion. | `model-registry-plan.md` |
| Clinical pathway | MONAI Deploy (MAP) | DICOM I/O, structured reports, IEC 62304 lifecycle. Complementary to BentoML. | `monai-deploy-plan.md` |
| Annotation platform | 3D Slicer + SlicerNNInteractive | nnInteractive backend, client-server via FastAPI, prior mask support | `interactive-segmentation-report.md` |
| Champion metric | 0.5*clDice + 0.5*normalize(MASD) | MetricsReloaded-informed. Topology (clDice) + surface distance (MASD) equally weighted. | `pr-d-deploy-flow-plan.xml` |
| BentoML response format | Full: mask + probabilities + uncertainties | Downstream consumers apply their own thresholds. Not just binary mask. | `e2e-testing-user-prompt.md` Q9 |
| Gradio demo | NIfTI upload + slice viewer + overlay | Axial/sagittal/coronal views. ONNX or dummy inference. Academic demo. | `bentoml-and-ui-demo-plan.md` |

---

## Implementation Status

| Document | Status | Key Deliverable | Implementation Evidence |
|----------|--------|-----------------|----------------------|
| `bentoml-and-ui-demo-plan.md` | **Partial** | BentoML + ONNX + Gradio serving stack | `serving/bento_service.py`, `serving/onnx_inference.py`, `serving/gradio_demo.py` exist as scaffolds. Key bugs identified (output.prediction wrong for raw model, Gradio 2D-only). Tasks T1-T4 (deps install, ONNX roundtrip, BentoML fix, Gradio NIfTI) partially addressed. |
| `interactive-segmentation-report.md` | **Reference** | Research report: interactive segmentation for 3D vascular annotation proofreading | 6 seed papers analyzed. nnInteractive + SlicerNNInteractive recommended. PRD `annotation_platform` decision updated. No code implementation (research report only). |
| `model-registry-plan.md` | **Implemented** | Model registry with 4-stage promotion (dev/staging/prod/archived) | `src/minivess/observability/model_registry.py` exists. ModelStage enum, ModelVersion, PromotionCriteria, PromotionResult, ModelRegistry orchestrator. Tests in `test_model_registry.py`. |
| `monai-deploy-plan.md` | **Partial** | MONAI Deploy clinical deployment pathway | `src/minivess/serving/clinical_deploy.py` exists with DeploymentTarget, DICOMConfig, ClinicalDeployConfig, DICOMHandler, MonaiDeployManifest, ClinicalDeploymentPipeline. `monai_deploy_app.py` and `monai_deploy_compat.py` also present. Tests partially written. |
| `pr-d-deploy-flow-plan.xml` | **Partial** | Deploy flow for factorial experiment champion models | `deploy_flow.py` exists with 5 tasks (discover, export, import, generate, promote). `champion_evaluator.py`, `champion_factorial_selection.py`, `champion_metadata.py`, `champion_registry.py` exist. Champion selection from factorial results and BentoML import chain verification planned but not fully validated E2E. |

---

## Cross-References

- **KG domain**: `knowledge-graph/domains/architecture.yaml` -- `serving_architecture: bentoml_primary` (resolved)
- **KG infrastructure domain**: `knowledge-graph/domains/infrastructure.yaml` -- Docker registry (DockerHub for RunPod, GAR for GCP), container_strategy
- **Source implementations**:
  - `src/minivess/serving/` -- 24 files covering BentoML, ONNX, Gradio, clinical deploy, champion selection, model registry
  - `src/minivess/orchestration/flows/deploy_flow.py` -- 5-task deploy flow
  - `src/minivess/serving/champion_evaluator.py` -- Compound metric ranking
  - `src/minivess/serving/champion_factorial_selection.py` -- Factorial-aware champion selection
  - `src/minivess/serving/bento_model_import.py` -- MLflow to ONNX to BentoML import chain
  - `src/minivess/serving/deploy_artifacts.py` -- Bentofile, docker-compose, README generation
  - `src/minivess/observability/model_registry.py` -- 4-stage promotion registry
- **Architecture theme**: Flow topology (deploy is 4th core flow), inter-flow MLflow contract, Hydra config for champion discovery
- **Testing theme**: E2E testing plan Phase 3 (BentoML + Grafana + Evidently), deploy import chain integration tests
- **Training theme**: Post-training plugins (SWA, calibration, conformal) produce candidates for champion selection
- **CLAUDE.md rules**: Rule 17 (never standalone scripts -- deployment goes through Prefect deploy flow), Rule 18 (explicit Docker volume mounts), Rule 19 (STOP protocol)

---

## Constituent Documents

1. `bentoml-and-ui-demo-plan.md` -- BentoML + ONNX + Gradio serving plan (2026-02-24). Issue #36. 4 tasks: install serving deps, ONNX export roundtrip test, fix BentoML service (ONNX not PyTorch), Gradio NIfTI upload with slice viewer. Current scaffold has bugs (output.prediction wrong, Gradio 2D-only). Academic serving use case.

2. `interactive-segmentation-report.md` -- Research report on interactive segmentation for proofreading 3D vascular annotations (2026-02-24). Evaluates nnInteractive (Isensee et al., 2025), SlicerNNInteractive (de Vente et al., 2025), MedSAM-Agent (Liu et al., 2026), K-Prism (Guo et al., 2025), OAIMS (Xu et al., 2025). Recommendation: 3D Slicer + SlicerNNInteractive as primary annotation/proofreading platform. MONAI Label retained for active learning. Key quantitative finding: prior mask reduces clicks by 30%+.

3. `model-registry-plan.md` -- Model registry with promotion stages (Issue #50). 4 stages: DEVELOPMENT, STAGING, PRODUCTION, ARCHIVED. ModelVersion with semantic versioning. PromotionCriteria with metric thresholds. ModelRegistry orchestrator with register_version(), evaluate_promotion(), promote(), get_production_model(). Fully implemented in `src/minivess/observability/model_registry.py`.

4. `monai-deploy-plan.md` -- MONAI Deploy clinical deployment pathway (Issue #47). Extends the BentoML research serving with clinical capabilities: DICOM I/O (validate metadata, create structured reports), MAP manifest generation, ClinicalDeploymentPipeline with validation and manifest creation. 4 deployment targets: RESEARCH, STAGING, CLINICAL, PACS. Partially implemented.

5. `pr-d-deploy-flow-plan.xml` -- Deploy flow for factorial experiments (2026-03-17). PR-D, execution order 4 of 5. Phase 0: champion selection from factorial results (compound metric ranking, CV-average preferred). Phase 1: BentoML import chain verification (MLflow to ONNX to BentoML). Phase 2: deployment artifact generation. Champion metadata tagging with all factorial factors. Existing infrastructure: 5 deploy flow tasks, MLflow pyfunc models, BentoML import utilities.
