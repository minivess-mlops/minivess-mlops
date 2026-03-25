# Cold-Start Prompt: README Enrichment with Verified Hyperlinks

**Branch**: `test/run-debug-gcp-5th-pass`
**Date**: 2026-03-25
**Skill**: `.claude/skills/readme-enrichment/SKILL.md`

## TASK

Execute `/readme-enrichment` on `README.md`. Add verified hyperlinks for all
libraries, papers, models, standards, and issue cross-references. Every URL
MUST be web-verified before insertion (zero hallucination policy).

## WHAT IS ALREADY DONE

- Skill created and registered (`.claude/skills/readme-enrichment/`)
- Bibliography loaded with 9 key papers in `knowledge-graph/bibliography.yaml`
- 44+ literature files available in `/home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/`
- Existing correct links identified (do NOT modify these)

## BIBLIOGRAPHY — PRE-RESOLVED PAPER URLS (from bibliography.yaml)

These are VERIFIED from the canonical bibliography — use directly:

| Model/Method | URL | Source |
|-------------|-----|--------|
| SAM3 | `https://github.com/facebookresearch/sam3` | bibliography.yaml `ravi_2025_sam3` |
| VesselFM | `https://arxiv.org/abs/2411.17386` | bibliography.yaml `wittmann_2024_vesselfm` |
| MambaVesselNet++ | `https://doi.org/10.1145/3757324` | bibliography.yaml `xu_2025_mambavesselnet_plus` |
| MambaVesselNet | `https://doi.org/10.1145/3696409.3700231` | bibliography.yaml `chen_2024_mambavesselnet` |
| clDice | `https://arxiv.org/abs/2003.07311` | bibliography.yaml `shit_2021_cldice` |
| MONAI | `https://arxiv.org/abs/2211.02701` | bibliography.yaml `cardoso_2022_monai` |
| MetricsReloaded | `https://doi.org/10.1038/s41592-023-02151-z` | bibliography.yaml (already linked in README) |
| SkyPilot | `https://www.usenix.org/conference/nsdi23/presentation/yang-zongheng` | bibliography.yaml (already linked) |
| SWAG | `https://arxiv.org/abs/1902.02476` | bibliography.yaml (already linked) |

## TOOL/LIBRARY HOMEPAGE URLS — NEED WEB VERIFICATION

Verify each of these via `WebFetch` before inserting. These are high-confidence
but MUST be confirmed (Rule R1):

| Tool | Expected URL | README Location |
|------|-------------|-----------------|
| PyTorch | `https://pytorch.org/` | Tech Stack table, line 341 |
| MONAI | `https://monai.io/` | Line 14, 39, 341 |
| TorchIO | `https://torchio.readthedocs.io/` | Line 341 |
| Hydra-zen | `https://mit-ll-responsible-ai.github.io/hydra-zen/` | Line 344 |
| Dynaconf | `https://www.dynaconf.com/` | Line 345 |
| Pydantic | `https://docs.pydantic.dev/` | Lines 343, 346 |
| Pandera | `https://pandera.readthedocs.io/` | Line 346 |
| Great Expectations | `https://greatexpectations.io/` | Line 346 |
| MLflow | `https://mlflow.org/` | Line 347 |
| DuckDB | `https://duckdb.org/` | Line 347 |
| Optuna | `https://optuna.org/` | Line 348 |
| BentoML | `https://www.bentoml.com/` | Line 349 |
| ONNX Runtime | `https://onnxruntime.ai/` | Line 349 |
| Gradio | `https://www.gradio.app/` | Line 349 |
| OpenLineage | `https://openlineage.io/` | Lines 26, 43, 350 |
| Marquez | `https://marquezproject.github.io/marquez/` | Lines 163, 350 |
| Evidently | `https://www.evidentlyai.com/` | Lines 44, 351 |
| whylogs | `https://whylogs.readthedocs.io/` | Lines 44, 352 |
| Prometheus | `https://prometheus.io/` | Line 353 |
| Grafana | `https://grafana.com/` | Line 353 |
| Docker Compose | `https://docs.docker.com/compose/` | Line 355 |
| Pulumi | `https://www.pulumi.com/` | Line 355 |
| Hypothesis | `https://hypothesis.readthedocs.io/` | Line 358 |
| gudhi | `https://gudhi.inria.fr/` | Line 359 |
| networkx | `https://networkx.org/` | Line 359 |
| scipy | `https://scipy.org/` | Line 359 |
| Captum | `https://captum.ai/` | Line 360 |
| SHAP | `https://shap.readthedocs.io/` | Line 360 |
| Quantus | `https://github.com/understandable-machine-intelligence-lab/Quantus` | Line 360 |
| Langfuse | `https://langfuse.com/` | Line 361 |
| Braintrust | `https://www.braintrust.dev/` | Line 361 |
| LiteLLM | `https://docs.litellm.ai/` | Line 361 |
| CycloneDX | `https://cyclonedx.org/` | Line 362 |
| CopilotKit | `https://www.copilotkit.ai/` | Line 479 |
| PostHog | `https://posthog.com/` | Line 480 |
| Sentry | `https://sentry.io/` | Line 480 |
| NVIDIA FLARE | `https://nvidia.github.io/NVFlare/` | Line 481 |
| MinIO | `https://min.io/` | Lines 131-132 |
| Pydantic AI | `https://ai.pydantic.dev/` | Lines 67, 343 |

## STANDARDS — NEED WEB VERIFICATION

| Standard | Expected URL |
|----------|-------------|
| IEC 62304 | `https://www.iso.org/standard/71604.html` or Wikipedia |
| FDA SaMD | `https://www.fda.gov/medical-devices/digital-health-center-excellence/software-medical-device-samd` |

## MODELS NOT IN BIBLIOGRAPHY — NEED WEB SEARCH

These models are mentioned in the README but not in bibliography.yaml.
Web search to find their papers:

| Model | What to search | README context |
|-------|---------------|----------------|
| DynUNet | MONAI DynUNet (dynamic UNet) | Line 39 — "DynUNet (CNN baseline)" |
| SAM3 TopoLoRA | SAM3 with topology-aware LoRA | Line 39 — SAM3 variant |
| SAM3 Hybrid | SAM3 hybrid adapter | Line 39 — SAM3 variant |
| CAPE loss | CAPE loss function for segmentation | Line 40 |
| Betti matching loss | Betti matching loss topology | Line 40 |
| Skeleton recall loss | Skeleton recall loss vessel | Line 40 |

NOTE: SAM3 TopoLoRA and Hybrid are our own adapter variants (no separate paper).
DynUNet is from MONAI (no separate paper — link to MONAI docs instead).
CAPE, Betti matching, skeleton recall — search `bibliography.yaml` or web search.

## PLANNED SECTION — ISSUE CROSS-REFERENCES

The "Planned" section (lines 477-483) needs:

1. Tool links for CopilotKit, PostHog, Sentry, NVIDIA FLARE
2. A "Science backlog" bullet with links to all 12 open science issues:
   ```markdown
   - **Science backlog**: calibration-aware ensembles ([#896](url)),
     greedy ensemble selection ([#894](url)), snapshot ensembles ([#895](url)),
     spec curve analysis ([#898](url)), uncertainty-guided eval ([#897](url)),
     topology-critical calibration ([#899](url)), VLM calibration ([#798](url)),
     federated learning ([#842](url)), Syne Tune HPO ([#861](url)),
     AI card stack ([#864](url)), KG provenance ([#938](url))
   ```

## IN PROGRESS SECTION — FIX STALE LINKS

Lines 473-474 have links to the OLD repo name (`minivess-mlops`):
- `https://github.com/petteriTeikari/minivess-mlops/issues/799` → should be `vascadia`
- `https://github.com/petteriTeikari/minivess-mlops/issues/821` → should be `vascadia`

## LINK FORMAT CONVENTION (from Skill R3)

- **Tech Stack table**: `[ToolName](homepage)` in the Tool column
  - Example: `| [PyTorch](https://pytorch.org/) + [MONAI](https://monai.io/) + [TorchIO](https://...) |`
- **Model names in Key Features**: `[ModelName](paper_url)` inline
  - Example: `[MambaVesselNet++](https://doi.org/10.1145/3757324)`
- **Loss functions**: `[clDice](https://arxiv.org/abs/2003.07311)` inline
- **Standards**: `[IEC 62304](https://...)` inline

## EXISTING CORRECT LINKS — DO NOT MODIFY

These are already correct in the README — preserve them:
- Line 24: de Almeida et al. link
- Line 33: Poon et al. dataset DOI
- Line 42: SkyPilot Yang et al.
- Line 46: MetricsReloaded Maier-Hein et al.
- Line 49: SWAG Maddox et al.
- Lines 434-440: ADR table links
- Lines 488-495: Further Reading links

## EXECUTION ORDER

1. **Read README.md** fully (already provided above)
2. **Verify tool URLs**: WebFetch each tool homepage (batch of 5-10 at a time)
3. **Verify paper URLs**: WebFetch bibliography entries for models
4. **Web search** for missing papers (CAPE, Betti matching, skeleton recall, DynUNet docs)
5. **Edit Tech Stack table**: Add `[Tool](url)` links to the Tool column
6. **Edit Key Features**: Add `[Model](url)` links for models and loss functions
7. **Edit Planned section**: Add tool links + science issue backlog
8. **Fix stale links**: Update minivess-mlops → vascadia in In Progress section
9. **Final verification**: Read the edited README and spot-check 10 random links

## BIBLIOGRAPHY SEARCH PATH

If a paper is not in `bibliography.yaml`:
1. Search `sci-llm-writer/biblio/biblio-vascular/` files
2. Web search for the paper
3. Add the citation to `bibliography.yaml` after verification

## FILES TO READ FIRST

```
README.md                                    # Target file
knowledge-graph/bibliography.yaml            # Paper URLs (pre-verified)
.claude/skills/readme-enrichment/SKILL.md    # Skill protocol
```
