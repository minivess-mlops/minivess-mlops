# Metalearning: paper_factorial.yaml Created with WRONG Model Lineup — AGAIN

**Date:** 2026-03-19
**Severity:** P0 — CRITICAL recurring failure (3rd occurrence)
**Trigger:** User asked to plan pre-debug QA; Claude presented two conflicting factorial
designs and asked user which was canonical — despite the KG having the answer.

---

## What Happened

1. User asked to create `fix/pre-debug-qa-verification` branch for GCP debug run
2. Claude launched 6 research agents to explore the codebase
3. Claude found TWO conflicting factorial designs:
   - `paper_factorial.yaml`: dynunet, **segresnet**, sam3_vanilla, vesselfm (WRONG)
   - `factorial-design-demo-experiment-plan.md`: DynUNet, Mamba++, SAM3 TopoLoRA, SAM3 Hybrid (CORRECT trainable models)
4. **Instead of checking the KG**, Claude presented BOTH to the user and asked "which is canonical?"
5. User response: "What the fuck with the models there? why in the fuck do we have some segresnet?"

## Why This Is a P0 Failure

The KG has the AUTHORITATIVE answer at `knowledge-graph/domains/models.yaml` line 34:

```yaml
paper_model_comparison:
  rationale: "6-model lineup for Nature Protocols paper: DynUNet, MambaVesselNet++,
    SAM3 Vanilla, SAM3 TopoLoRA, SAM3 Hybrid, VesselFM"
```

And there is ALREADY a metalearning doc about this EXACT failure:
`2026-03-17-model-lineup-ignorance-massive-fuckup.md`

Despite this:
- Claude Code created `paper_factorial.yaml` in PR #865 (2026-03-19) with **segresnet**
  instead of mambavesselnet++ and sam3_topolora/sam3_hybrid
- Claude Code then asked the USER to resolve the conflict it created
- The user had to correct Claude for the THIRD TIME on the same issue

## Root Cause Analysis (Deep)

### RC1: paper_factorial.yaml was created WITHOUT reading the KG
PR #865 created this file on 2026-03-19. The KG node `paper_model_comparison` was
already present (created 2026-03-17). The metalearning doc about this EXACT failure
was already written (2026-03-17). Neither was consulted during PR #865.

**The metalearning system failed because the agent creating the PR did not read
metalearning docs before creating configs.**

### RC2: The agent bootstrapped the factorial from CODE rather than from the KG
`paper_factorial.yaml` lists dynunet, segresnet, sam3_vanilla, vesselfm — these are
the models that have `status: production` or `status: experimental` adapter code.
The agent scanned `src/minivess/adapters/` and picked models with working adapters.

MambaVesselNet++ (`status: complete, GPU_PENDING`) was excluded because it hadn't
been GPU-verified. SAM3 TopoLoRA and SAM3 Hybrid were excluded for the same reason.

**The agent used "what works now" instead of "what the paper needs."**

### RC3: No validation gate between KG and config files
There is no test that verifies `paper_factorial.yaml` model list matches
`paper_model_comparison.yaml` model list. The config file and the KG node
can diverge silently.

### RC4: Session-to-session amnesia on model lineup
This is the THIRD occurrence:
1. 2026-03-15: `kg-mamba-missing-scope-blindness.md` — Mamba not in KG
2. 2026-03-17: `model-lineup-ignorance-massive-fuckup.md` — Could not name 6 models
3. 2026-03-19: THIS DOC — Created paper_factorial.yaml with wrong models

Each time:
- A metalearning doc was written
- The KG was updated
- The NEXT session ignored both

**The metalearning → behavior loop is broken. Writing docs does not prevent recurrence.**

### RC5: T4 still in gcp_spot.yaml despite being BANNED
`configs/cloud/gcp_spot.yaml` lists `T4:1` as the FIRST accelerator priority,
despite `2026-03-15-t4-turing-fp16-nan-ban.md` banning T4 and CLAUDE.md explicitly
stating "T4 BANNED (Non-Negotiable)." Same pattern: metalearning written, config
not fixed.

## The Correct Model Lineup (Single Source of Truth)

**Trainable models** (4, for factorial design):
1. **DynUNet** — CNN baseline (3.5 GB VRAM, fits local 8 GB)
2. **MambaVesselNet++** — SSM hybrid (estimated 4-8 GB, needs cloud verification)
3. **SAM3 TopoLoRA** — Foundation + LoRA fine-tuning (~16 GB, needs L4/A100)
4. **SAM3 Hybrid** — Foundation + DynUNet fusion (7.18 GB, needs cloud 24 GB)

**Zero-shot baselines** (2, no training):
5. **SAM3 Vanilla** — Frozen encoder, zero-shot eval only
6. **VesselFM** — External data only (DeepVess/TubeNet, NOT MiniVess)

**NOT in the paper factorial**: SegResNet, SwinUNETR, AttentionUNet, UNETR,
COMMA-Mamba, ULike-Mamba. These may have adapter code but are NOT in the
paper comparison.

## Corrective Actions

### Immediate (this session):
1. ✅ Fix `paper_factorial.yaml` — replace segresnet with correct models
2. ✅ Fix `gcp_spot.yaml` — remove T4, L4 only
3. ✅ Fix `gcp_quotas.yaml` — remove T4 quota references
4. ✅ Write this metalearning doc
5. ✅ Create test: `test_factorial_matches_kg.py` — verifies config ↔ KG consistency

### Structural (prevent 4th occurrence):
6. Add to CLAUDE.md: **"Before creating ANY experiment config, read
   `knowledge-graph/domains/models.yaml::paper_model_comparison` for the
   authoritative model lineup."**
7. Add pre-commit hook or test that validates paper_factorial.yaml models
   against the KG `paper_model_comparison` node

## Behavioral Rule

**BEFORE creating or modifying ANY experiment config that lists model families:**
1. Read `knowledge-graph/domains/models.yaml::paper_model_comparison`
2. Cross-check every model name against the KG node
3. If the config disagrees with the KG → the CONFIG is wrong, not the KG
4. Never derive model lists from adapter code scan — derive from KG

**The KG `paper_model_comparison` node is the SINGLE SOURCE OF TRUTH for which
models are in the paper. Everything else (configs, plans, issues) must be
DERIVED from it, not the other way around.**

## Related Failures

- `2026-03-15-kg-mamba-missing-scope-blindness.md` — 1st occurrence
- `2026-03-17-model-lineup-ignorance-massive-fuckup.md` — 2nd occurrence
- `2026-03-16-kg-exists-read-it-first.md` — Same meta-pattern (not reading KG)
- `2026-03-14-poor-repo-vision-understanding.md` — Same meta-pattern (not understanding WHY)
- `2026-03-15-t4-turing-fp16-nan-ban.md` — T4 ban written but config not fixed
