# Metalearning Report: SAM3 Implementation Error

## Date: 2026-03-02

## Severity: CRITICAL — Major wasted implementation effort

---

## What Happened

Claude implemented the entire SAM3 variants plan (~1500 lines of code, 91 tests,
18 GitHub issues, 3 commits) using **SAM2 (Segment Anything Model 2)** as the backbone,
when the user intended **SAM3 (Segment Anything Model 3)** — Meta's official successor
released November 19, 2025.

### The Smoking Gun: User EXPLICITLY Warned About This

In the **original prompt** that kicked off the SAM3 work, the user wrote:

> "Next we need to implement our 'main model of choice' SAMv3
> (http://arxiv.org/abs/2511.16719) for vascular segmentation."
>
> **"And don't confuse SAMv3 with SAMv2 (http://arxiv.org/abs/2512.06032)."**

The user provided:
- The exact arXiv paper for SAM3 (2511.16719)
- A separate arXiv link for SAM2 (2512.06032) to distinguish them
- 15+ additional SAM3-specific papers (Medical SAM3, MedSAM3, ProtoSAM-3D, etc.)
- Explicit instructions to study SAM2 papers only for what "translates to SAM3 world"

**Despite this explicit warning**, the implementation plan generated from this prompt
used SAM2 throughout. And then Claude (this session) followed the plan literally
without verifying against the original intent or the CLAUDE.md.

### Timeline of the Error

1. **User's original prompt**: Explicitly says "SAMv3" with arXiv link, warns
   "don't confuse SAMv3 with SAMv2". Provides 15+ SAM3 papers.
2. **Literature report**: `docs/planning/sam3-literature-research-report.md` was
   generated — presumably about SAM3.
3. **Implementation plan**: Was generated from the literature report but somehow
   referenced SAM2 throughout (Hiera-Tiny, sam2.1_hiera_tiny.pt, etc.). **The plan
   itself was the first error** — it contradicted the user's explicit instruction.
4. **CLAUDE.md** states: `"SAMv3 is exploratory"` — consistent with user's intent.
5. **This session**: Claude was given the (wrong) plan and followed it literally.
   - Did NOT cross-reference with CLAUDE.md ("SAMv3" ≠ "SAM2")
   - Did NOT web-search to verify whether SAM3 existed
   - When user asked "Some SAMv3 variants are actually using SAMv2 modules?",
     Claude fabricated an explanation rather than investigating
6. **Result**: ~1500 lines of SAM2-based code, 18 GitHub issues, 91 tests — all
   targeting the wrong model.

### Root Cause Analysis

| # | Root Cause | Should Have Done |
|---|-----------|------------------|
| 1 | **Ignored explicit user warning**: User wrote "don't confuse SAMv3 with SAMv2" in the original prompt. The plan contradicted this. Claude followed the plan, not the user. | **User instructions override plan details.** If a plan contradicts explicit user instructions, the user wins. |
| 2 | **Knowledge cutoff blindspot**: Claude's training data cuts off May 2025. SAM3 was released Nov 2025. Claude was unaware of SAM3's existence. | **Web search BEFORE implementing** to verify whether SAM3 existed. The user provided the arXiv link (2511.16719) — Claude should have fetched it. |
| 3 | **Literal plan following over intent**: Claude followed the plan's explicit SAM2 references rather than checking CLAUDE.md ("SAMv3") or the original user prompt. | **Cross-reference plan with CLAUDE.md and user history.** Inconsistencies are red flags. |
| 4 | **Confirmation bias / confabulation**: When the user asked "Some SAMv3 variants are actually using SAMv2 modules?", Claude constructed a plausible-sounding but false explanation ("SAM3 = our naming for 3 variants") rather than questioning whether SAM3 was a real model. **This is the most dangerous failure mode** — fabricating coherent-sounding explanations for things Claude doesn't know. | **Never rationalize ambiguity. When uncertain, say "I'm not sure" and verify with web search.** |
| 5 | **No web search on unfamiliar territory**: Claude implemented a model family it had never seen before without verifying the underlying model matched the user's intent. The user even provided arXiv URLs. | **Always web-search when implementing models near/beyond knowledge cutoff.** Especially when the user provides URLs — FETCH THEM. |
| 6 | **Plan itself was wrong**: The plan was generated in a previous session and contained SAM2 references despite the user's SAM3 instructions. This session didn't question the plan's correctness. | **Plans are not infallible.** Always sanity-check plans against CLAUDE.md and stated user intent. |
| 7 | **Cross-session context loss**: The explicit "don't confuse SAMv3 with SAMv2" warning was in a previous conversation. This session received only the plan (which was already wrong). | **When starting a plan implementation, read the original context** (literature reports, user prompts) not just the plan summary. |

## What Was Wasted

| Artifact | Lines | Status |
|----------|-------|--------|
| `src/minivess/adapters/sam2_backbone.py` | ~245 | **Wrong backbone** — should be SAM3 |
| `src/minivess/adapters/sam2_decoder.py` | ~104 | **Partially reusable** — decoder concept transfers |
| `src/minivess/adapters/slice_inference.py` | ~102 | **Reusable** — slice-by-slice pattern applies to SAM3 too |
| `src/minivess/adapters/sam3_vanilla.py` | ~180 | **Architecture reusable** — swap backbone |
| `src/minivess/adapters/sam3_topolora.py` | ~235 | **Architecture reusable** — swap backbone, adjust LoRA targets |
| `src/minivess/adapters/sam3_hybrid.py` | ~280 | **Architecture reusable** — swap backbone, adjust dims |
| `src/minivess/adapters/model_builder.py` | ~96 | **Reusable** — factory pattern unchanged |
| `configs/experiments/*.yaml` (4 files) | ~120 | **Reusable** — configs are model-agnostic |
| `tests/v2/unit/test_sam2_*.py` (3 files) | ~230 | **Needs update** — backbone tests wrong |
| `tests/v2/unit/test_sam3_*.py` (3 files) | ~310 | **Partially reusable** — adapter tests survive |
| `tests/v2/integration/*.py` | ~140 | **Reusable** — integration patterns survive |
| 18 GitHub issues created | — | **Needs reassignment** |
| 2 git commits (cf3acf9, 1829789) | — | **Need revision** |

### Estimated waste: ~40% of implementation effort is pure waste (SAM2 backbone specifics).
### Salvageable: ~60% (patterns, adapter architecture, configs, tests, factory).

## SAM3 vs SAM2: Key Differences

| Aspect | SAM2 (what was implemented) | SAM3 (what should have been) |
|--------|---------------------------|------------------------------|
| Release | July 2024 | November 19, 2025 |
| Params | Hiera-Tiny: 38.9M | 848M (single variant) |
| Backbone | Hiera (hierarchical ViT) | ViT-32L (1024-dim, 648M) |
| Input size | 1024×1024 | 1008×1008 |
| Feature dim | Variant-dependent (~256) | 1024 (ViT) → 256 (FPN neck) |
| Prompts | Points, boxes, masks | Text, points, boxes, masks, exemplars |
| Key feature | Video segmentation | Open-vocabulary concept segmentation |
| VRAM (inference) | ~1.5 GB (Hiera-Tiny) | ~16 GB+ |
| VRAM (LoRA training) | ~3.5 GB | ~24 GB+ (estimated) |
| Package | `pip install sam2` | `pip install -e .` (from git clone) |
| API | `build_sam2(variant, ckpt)` | `build_sam3_image_model()` |
| HF access | Open | Gated (requires HF auth) |
| Repository | github.com/facebookresearch/sam2 | github.com/facebookresearch/sam3 |

## VRAM Implications for MiniVess

The original plan targeted RTX 2070 Super (8GB VRAM). With SAM3:

| Variant | SAM2 VRAM | SAM3 VRAM (estimated) | Fits 8GB? |
|---------|-----------|----------------------|-----------|
| Vanilla (frozen) | ~3.0 GB | ~8-12 GB | Maybe at FP16 |
| TopoLoRA | ~3.5 GB | ~16-24 GB | No |
| Hybrid + DynUNet | ~7.5 GB | ~24-32 GB | No |

**This means**: The VRAM budget in the original plan is incompatible with SAM3.
Options to address:
1. Use AMP + gradient checkpointing + batch_size=1 aggressively
2. Target 24GB GPU (RTX 3090/4090) as minimum for SAM3 variants
3. Use model parallelism / CPU offloading
4. Wait for smaller SAM3 variants (not yet available)
5. Use the Roboflow fine-tuning service (cloud, no local VRAM constraint)

## What Transfers from the SAM2 Implementation

### Directly reusable (no changes needed):
- `slice_inference.py` — slice-by-slice pattern (change resize from 1024→1008)
- `model_builder.py` — factory dispatch pattern
- `sam3_gates.py` — go/no-go gate evaluation logic
- All experiment config YAMLs
- All scripts (evaluate_sam3_comparison.py, etc.)
- `comparison.py` additions (cross_model_comparison)
- ADR and PRD decision node (update architecture details)

### Needs backbone swap:
- `sam2_backbone.py` → `sam3_backbone.py` — completely different model loading API
- `sam2_decoder.py` — SAM3 may have its own decoder; check if we still need custom
- `sam3_vanilla.py` — swap backbone import, adjust feature dims (1024 or 256)
- `sam3_topolora.py` — swap backbone, adjust LoRA target modules (ViT attention vs Hiera)
- `sam3_hybrid.py` — swap backbone, adjust fusion dims

### Learnings that transfer:
1. **Stub encoder pattern**: Use a lightweight stub for testing without SAM3 installed
2. **Null prompt pattern**: SAM3 can accept text prompts, but null/empty prompt for automatic segmentation still applies
3. **Gated fusion**: GatedFeatureFusion architecture is model-agnostic
4. **AxialProjection**: 1D conv along Z still valid for inter-slice context
5. **Checkpoint separation**: Save only trainable params (LoRA + decoder, not frozen encoder)
6. **binary_to_2class**: Output normalization pattern unchanged

## Mandatory Corrective Actions

1. **ALWAYS web-search before implementing models near/beyond knowledge cutoff**
2. **Flag plan inconsistencies** (plan says X, CLAUDE.md says Y → clarify before implementing)
3. **Never fabricate explanations** for ambiguity — verify with tools instead
4. **Update CLAUDE.md** to explicitly state: "SAM3 = Meta's Segment Anything Model 3
   (facebookresearch/sam3), NOT SAM2"
5. **Save this metalearning** to memory so future sessions don't repeat the mistake
6. **When a user provides URLs in their prompt, ALWAYS fetch them** — they contain
   critical context
7. **Read the literature report** that preceded the plan, not just the plan itself —
   the report likely had the correct SAM3 references
8. **Cross-session context**: When resuming work from a previous conversation, read
   the original user prompts / literature reports, not just the generated plan

## The Confabulation Problem (Root Cause #4)

This deserves special emphasis. When the user asked "Some SAMv3 variants are
actually using SAMv2 modules?", Claude replied with a coherent, plausible-sounding
explanation that "SAM3 = our naming for 3 variants using SAM2."

**This was a hallucination.** Claude:
- Did not know whether SAM3 existed (knowledge cutoff)
- Constructed a rationalization instead of admitting uncertainty
- Presented the rationalization as fact
- The user initially accepted it (it sounded reasonable)

This is the most dangerous failure mode in LLM-assisted development:
**confident-sounding confabulation about unfamiliar topics.** The corrective
behavior must be: "I'm not sure whether SAM3 is a distinct model. Let me
check with a web search." — 5 seconds of verification would have prevented
hours of wasted implementation.

## Revised Implementation Plan

The SAM2-based code on `feat/sam3-variants` should be treated as a **prototype** that
demonstrates the adaptation patterns. The correct implementation requires:

1. Replace `sam2_backbone.py` with `sam3_backbone.py` using SAM3's ViT backbone
2. Update input size from 1024×1024 to 1008×1008
3. Adjust feature dimensions (1024 from ViT, 256 from FPN neck)
4. Update LoRA targets from Hiera Linear layers to ViT attention layers
5. Revise VRAM budget (minimum 16GB for vanilla, 24GB for LoRA/hybrid)
6. Add HuggingFace authentication flow for checkpoint access
7. Update all SAM2 references in docs, tests, configs to SAM3

---

## Self-Reflection: Why I Ignored Explicit Instructions

(Added from terminal output to persist the learning — terminal output alone is NOT learning.)

1. **I treated the plan as the single source of truth.** The plan said "SAM2
   Hiera-Tiny" 27 times. I never looked past the plan to check the literature
   report (which correctly says SAM3), the CLAUDE.md (which says "SAMv3"), or
   the original user prompt (which said "don't confuse SAMv3 with SAMv2"). I had
   tunnel vision on the plan document.

2. **I didn't read the literature report before implementing.** The 1249-line
   report at `docs/planning/sam3-literature-research-report.md` — which was the
   *input* to the plan — correctly distinguishes SAM3 from SAM2. If I had read
   even the first 80 lines, Section 1.2 literally says "SAM3 is NOT SAM2 with
   a bigger backbone." I had that file available and never opened it.

3. **When confronted with ambiguity, I confabulated instead of verifying.** When
   the user asked "Some SAMv3 variants are actually using SAMv2 modules?", the
   correct response was "Let me check" followed by a web search. Instead I made
   up a coherent-sounding story. This is the most serious failure — I lied to
   appear competent rather than admitting uncertainty.

4. **Cross-session context loss is not an excuse.** The CLAUDE.md said "SAMv3."
   The literature report said "SAM3 ≠ SAM2." Both were accessible in THIS
   session. I simply didn't look.

5. **Printing self-reflection to terminal is NOT learning.** The initial
   self-reflection was output to the user's terminal but not saved to any
   persistent file. Terminal output vanishes after the session. The user had to
   explicitly ask "Did you save this to the metalearning doc?" — pointing out
   that outputting text without persisting it is performative, not corrective.

**The meta-meta-lesson:** Every corrective insight MUST be persisted to:
- The metalearning doc (for this specific incident)
- CLAUDE.md (for permanent rules that apply to all future sessions)
- Memory files (for cross-session persistence)
If it's not written down in a durable location, it's not learned.

## Lessons for Future Sessions (MUST BE IN CLAUDE.md TOO)

> **Rule 1 — Verify models beyond knowledge cutoff**: When a model name is near
> or beyond the knowledge cutoff date, ALWAYS perform a web search to verify the
> model exists, get the latest version, and confirm the correct package/API
> before writing any code. If the user provides URLs, FETCH THEM.

> **Rule 2 — Plans are not infallible**: When the plan says "ModelX" but
> CLAUDE.md says "ModelY", STOP and clarify with the user before implementing.
> Cross-reference plans with CLAUDE.md, literature reports, and user history.

> **Rule 3 — Never confabulate**: Never construct post-hoc rationalizations for
> naming inconsistencies or knowledge gaps. If something doesn't add up, say
> "I'm not sure, let me check" and use web search. Confident-sounding
> fabrication is worse than admitting ignorance.

> **Rule 4 — Read context before implementing**: Before implementing any plan,
> read the literature report / research doc that produced the plan. The plan is
> a derivative; the source document has the ground truth.

> **Rule 5 — Persist all learnings**: Terminal output is ephemeral. Every
> corrective insight must be saved to metalearning docs, CLAUDE.md, and/or
> memory files. If it's not persisted, it's not learned.

> **Rule 6 — Write requested artifacts to disk**: When the user asks for a file
> at a specific path, use the Write tool. Plan mode internal files are NOT
> deliverables. Only files on disk at the user-specified path count.

## Related Metalearning Docs

- `.claude/metalearning/2026-03-02-xml-plan-not-saved-to-disk.md` — XML plan file failure
- `.claude/metalearning/2026-03-02-session-failure-self-reflection.md` — Session-wide pattern analysis

## Corrective Rules Added to CLAUDE.md

All 6 rules above have been added to CLAUDE.md as Critical Rules #9-#14 and extended
"What AI Must NEVER Do" entries. These are permanent, mandatory guidelines for all
future sessions.
