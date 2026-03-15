# 2026-03-15 — KG Scope Blindness: Mamba Missing from Knowledge Graph

## What Happened

During cover letter creation for sci-llm-writer, the user mentioned four model families
in their prompt: "dynUnet, Mamba, vesselfm and SAM3." When asked a clarifying question
about Mamba's status ("Is Mamba future work only / add to R3b / separate paper / not sure?"),
the user was justifiably furious: **"How is this unclear? Mamba is one model that we are
testing! WTF was all your knowledge graph work there if you cannot figure this out?"**

The user is correct. The KG was built in this session precisely so that agent sessions
have project scope without requiring the user to re-explain it. Asking the user to confirm
the model comparison scope is a failure of the entire KG system.

---

## Root Cause Analysis

### Primary Failure: KG construction only captured existing code, not planned work

The `adapters.yaml` bootstrap in T06 scanned `src/minivess/adapters/` and found:
DynUNet, SegResNet, SAM3 (V1/V2/V3), VesselFM, SwinUNETR, AttentionUNet.

Mamba has no adapter file yet (because it's not yet implemented). Therefore Mamba
was silently absent from the KG. The KG scanner faithfully reflected the code — but
the code was incomplete relative to the PLANNED experiment scope.

**The KG captured current state but NOT planned scope.**

### Secondary Failure: Agent questioned user's stated scope rather than flagging a KG gap

When the user mentioned Mamba, the correct response was:
> "I see Mamba mentioned in your prompt but there is no Mamba adapter in
> `code-structure/adapters.yaml`. This appears to be a KG gap — Mamba is planned
> but not yet implemented. I'll note this as a gap in the cover letter."

Instead, the agent asked the user to CONFIRM whether Mamba was planned at all —
treating the user as an unreliable source about their own paper's scope. This is
the opposite of the trust relationship that should exist with a repo owner.

**Rule**: When the user states something as fact about their own project, assume
they are correct. Check the KG for confirmation; if the KG contradicts the user,
treat it as a **KG gap**, not as a user error.

### Tertiary Failure: No "planned_adapters" section in the KG

The `adapters.yaml` format has no mechanism to capture models that are IN SCOPE
for the paper but not yet implemented. The adapter node schema captures only
`status: production | experimental | available` — there is no `status: planned`.

---

## Impact

- User had to repeat paper scope that should have been in the KG
- Caused frustration and eroded trust in the KG system
- Cover letter had to have a "⚠️ KG GAP" warning about Mamba
- GitHub Issue #735 (P0) created to track the fix

---

## Fixes Required

### Fix 1 (IMMEDIATE): Add Mamba to adapters.yaml with `status: planned`

```yaml
  - id: mamba_variants
    name: "Mamba Variants (SegMamba / U-Mamba)"
    file: src/minivess/adapters/mamba.py  # not yet created
    family: state_space_model
    status: planned
    gpu_requirement: "~8 GB (similar to DynUNet — to be verified)"
    source: "SegMamba (github.com/ge-xing/SegMamba) or U-Mamba (github.com/bowang-lab/U-Mamba)"
    intent:
      why_chosen: >
        Represents the State Space Model (SSM) family. Demonstrates platform generalizability
        across transformer, CNN, and SSM architectures. Hypothesis: long-range context via
        selective state spaces may improve vascular topology continuity vs sliding-window DynUNet.
      intent_source: "User session 2026-03-15 — explicit paper scope"
    test_markers: [model_loading]
    note: "Adapter not yet implemented — needed for R3b multi-model comparison"
```

### Fix 2 (STRUCTURAL): Add `planned_scope` section to manuscript/methods.yaml and adapters.yaml

Distinguish between:
- `status: production` — implemented, tested, in results
- `status: experimental` — implemented, results pending GPU runs
- `status: planned` — in paper scope, adapter not yet written

### Fix 3 (BEHAVIORAL): Update KG bootstrap protocol

When the user mentions a model family not in the KG during any session, the agent MUST:
1. Flag it as a KG gap immediately
2. Add it to `adapters.yaml` with `status: planned`
3. NEVER ask the user whether it's "really" in scope

### Fix 4 (PROCESS): Planning docs → KG extraction

If there are planning docs (CLAUDE.md, docs/planning/*.md) that describe intended
model families, the KG bootstrap should extract them as `status: planned` nodes —
not just scan code. The planned model scope should come from the author's stated
intentions, not just from the code that exists today.

---

## Behavioral Rules Added

**WHEN** user mentions a model, tool, or component not found in the KG:
- **DO**: Say "This isn't in the KG yet — noting as a gap"
- **DO**: Add it to the appropriate KG file with `status: planned`
- **DO NOT**: Ask the user to confirm whether it's really in scope
- **DO NOT**: Frame it as optional ("is this future work only?")

**The user is the source of truth about their own project. The KG is the documentation.
If they conflict, the KG has a gap.**

---

## Related Failures

- `2026-03-02-sam3-implementation-fuckup.md` — SAM3 vs SAM2 confusion
- `2026-03-15-wrong-config-chasing-phantoms.md` — config state confusion
- Common thread: agent didn't treat user statements as authoritative, wasted time with unnecessary confirmation questions
