# TopoLoRA-SAM3 Planning Report

**Date**: 2026-03-21
**Branch**: `test/local-debug-3flow-execution`
**Plan output**: `docs/planning/post-run-debug-factorial-experiment-report-2nd-pass-local-fixes.xml`

---

## Paper Reference

**TopoLoRA-SAM**: Khazem et al. (2025). "Topology-Aware Low-Rank Adaptation for
Retinal Vessel Segmentation."
- Paper: https://arxiv.org/html/2601.02273v1
- Code: https://github.com/salimkhazem/Seglab

---

## User Prompts and Q&A (Verbatim)

### Round 1: Issue Scope

**Q1**: Issue #878 says H1 (--no-serve-artifacts) is ALREADY in Dockerfile.mlflow-gcp
and smoke_test_gcp.yaml. The fix is switching train_factorial.yaml from file-based
/app/mlruns back to the Cloud Run URI. Should we just finish the 10% or deeper rearchitecture?

**User answer**: "Full checkpoint rearchitecture"
(H1+H6 + atomic writes + SHA256 verification + spot resume wiring)

---

**Q2**: Issue #879 (LoRAConv2d) is P2 and you explicitly deferred it on 2026-03-20.
SAM3 TopoLoRA works with Linear-only LoRA. Should we still address this before GCP?

**User answer**: "TopoLoRA-SAM is coming from this paper
https://arxiv.org/html/2601.02273v1 and from https://github.com/salimkhazem/Seglab
and the implementation has to be exactly like that! This should be made even more
crystal clear in the CLAUDE.md, kg, everywhere and if this is not that the WHOLE
PLATFORM REPO so far is about developing a MLOps Infrastructure and creating our own
SOTA implementations. Everything as it is in the literature!"

---

**Q3**: I couldn't see the pasted SkyPilot text you referenced.

**User answer**: "I have no idea what you are asking? Can you reformulate this better"

---

**Q4**: Should this plan ONLY cover fixing #878/#879 before GCP, or include the local
3-flow debug execution?

**User answer**: "The 3-flow debug was already run on a different run, read the report:
docs/planning/run-debug-factorial-experiment-report-2nd-pass-local.md"

---

### Round 2: Architecture Fidelity

**Q5**: The TopoLoRA-SAM paper uses SAM 1 (ViT-B, 93.7M params), NOT SAM3 (ViT-32L,
648M params). Our repo adapts the TopoLoRA technique to SAM3. Should we match the
paper EXACTLY (switch to SAM1) or keep our SAM3 adaptation?

**User answer**: "Keep SAM3, match technique, but with the only tweak that as our one
factor in this experiment is to vary the loss, we should make the choice of loss
flexible! SO all the loss choices (3) and any other from MONAI that we later implement
should be easily used, so there can be some switch / if/elif/else type of loss selector
there"

---

**Q6**: The paper uses clDice loss. Should we verify our clDice matches the paper's
differentiable version?

**User answer**: "See my previous answer, so have a look what SegLab used, and if they
used some exotic loss that we did not yet have, then we have to add that to our losses
in this factorial design so we can have a look how this TopoLoRA worked with the default
loss compared to all the other losses, and possibly contribute some new scientific
knowledge to the field, right?"

---

### Round 3: Checkpoint Depth

**Q7**: For #878 you chose 'Full checkpoint rearchitecture'. How deep — just persistence
(H1+H6) or also reliability features (atomic, SHA256, spot resume)?

**User answer**: "Everything including spot resume"

---

## Key Decisions Made

| Decision | Rationale |
|----------|-----------|
| Keep SAM3 ViT-32L as backbone | Deliberate adaptation of TopoLoRA technique to SAM3 |
| Add Spatial Adapter (Conv_DW + Conv) | Missing from our implementation, paper has it |
| LoRA on FFN only (no Conv2d) | Paper explicitly targets FFN only |
| Config-driven loss selection | All 3 factorial losses + future MONAI losses must work |
| Full checkpoint rearchitecture | H1+H6+atomic+SHA256+spot resume before GCP |
| Relabel #879 | From "LoRAConv2d" to "Spatial Adapter + paper fidelity" |

---

## Paper Architecture (From Diagram)

```
Input (H×W×C)
    ↓
SAM Image Encoder (ViT, FROZEN ❄️)
    ├── Transformer Block (frozen) + LoRA (FFN, trainable 🔥)
    ├── Transformer Block (frozen) + LoRA (FFN, trainable 🔥)
    ├── ...
    └── Transformer Block (frozen) + LoRA (FFN, trainable 🔥)
    ↓
Image Embeddings (H/16 × W/16)
    ↓
Spatial Adapter (trainable 🔥)
    Conv_DW(3×3) + Conv(1×1) + BN + GELU + residual
    ↓
Decoder (trainable 🔥)
    ↓
Output Mask
```

**Trainable parameters** (~5% of total):
- LoRA A/B matrices on FFN: ~2.4M (paper r=16)
- Spatial Adapter: ~66K
- Mask Decoder: ~2.4M

---

## Gap Analysis: Our Code vs Paper

| Component | Paper (SegLab) | Our Implementation | Gap |
|-----------|---------------|-------------------|-----|
| Encoder | SAM1 ViT-B (frozen) | SAM3 ViT-32L (frozen) | DELIBERATE adaptation |
| LoRA targets | FFN (mlp.lin1/lin2) | FFN (Linear layers) | ✅ Correct |
| LoRA rank | r=16 (optimal from ablation) | r=16 (default) | ✅ Correct |
| Spatial Adapter | Conv_DW(3x3)+Conv(1x1)+BN+GELU+residual | **MISSING** | ❌ Must add |
| Decoder | SAM Mask Decoder (trainable) | Sam3MaskDecoder (trainable) | ✅ Correct |
| Loss | BCE+Dice+0.5*clDice | cbdice_cldice | ⚠️ Similar but not identical |
| clDice | Soft skeletonization (10 iter min/max pool) | MONAI SoftclDiceLoss | ⚠️ Verify equivalence |
| Conv2d LoRA | NOT applied | NOT applied (skipped) | ✅ Correct |

---

## SegLab Loss Analysis

SegLab uses:
- BCE loss (weight 1.0)
- Dice loss (weight 1.0)
- clDice with soft skeletonization (weight 0.5)
- Boundary loss with Laplacian edge detector (weight 0.0 = disabled by default)

**No exotic losses** — all map to our existing loss registry. The compound
`dice_ce_cldice` is the closest match to the paper's default.

---

## Files Referenced

| File | Purpose |
|------|---------|
| `src/minivess/adapters/sam3_topolora.py` | Our TopoLoRA adapter (needs Spatial Adapter) |
| `src/minivess/pipeline/loss_functions.py` | Loss registry (18 losses) |
| `deployment/skypilot/train_factorial.yaml` | SkyPilot YAML (needs URI + mounts fix) |
| `src/minivess/orchestration/flows/train_flow.py` | Training flow (needs atomic + resume) |
| `docs/planning/run-debug-factorial-experiment-report-2nd-pass-local.md` | Local debug results |
