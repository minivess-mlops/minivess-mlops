# 2026-03-14 — Poor Understanding of Repo Vision: Cloud GPU Purpose

## Discovery

Claude Code repeatedly planned to run DynUNet (3.5 GB VRAM) on RunPod cloud GPUs
as a "plumbing test" — despite the fact that DynUNet runs perfectly on the local
RTX 2070 Super (8 GB). This appeared in every RunPod plan (v1 through v4.0 initial),
wasting plan complexity, issue tracking overhead, and would have wasted real GPU
credits if executed.

The ENTIRE reason cloud GPU exists in this project is for models that OOM locally:
- **SAM3 hybrid**: 7.18 GiB @ patch 64³ — OOMs on 8 GB GPU
- **VesselFM**: ~10 GB estimated — OOMs on 8 GB GPU

This is not obscure knowledge. It is the foundational motivation for the cloud GPU
work. Yet Claude failed to internalize it across 9 plans and 12 metalearning docs.

## Root Cause Analysis

### 1. No permanent record of "why cloud GPU exists"

The CLAUDE.md file (43 KB, 699 lines) documents extensively how to run things but
never states the WHY of cloud GPU in one clear sentence:

> "Cloud GPU is ONLY for models that OOM on the local 8 GB RTX 2070 Super.
> If a model fits locally, it MUST run locally. Never burn cloud credits on
> models that run fine on the dev machine."

This sentence does not exist anywhere in the repo. It should.

### 2. Generic "plumbing test" pattern applied without thinking

The "run a small model first to verify infrastructure" pattern is reasonable in
general. But it requires asking: "does this model NEED the infrastructure being
tested?" DynUNet does not need a 24 GB cloud GPU. The plumbing test should use
the SMALLEST model that actually requires cloud — which is SAM3 hybrid itself.

### 3. CLAUDE.md overload causes information dilution

At 43 KB / 699 lines, CLAUDE.md is beyond what any LLM processes effectively.
Critical context (like "cloud GPU = heavy models only") gets lost in the noise
of Docker hardening rules, citation formatting, regex bans, and Prefect flow
architecture. This is exactly what issue #693 (CLAUDE.md refactor) addresses.

### 4. Session amnesia despite memory system

The memory system (`MEMORY.md` + memory files) captures many facts but misses
the core strategic context: what hardware the user has, what it can/can't run,
and therefore what cloud compute is needed for. Each session re-derives this
from scratch instead of reading a clear statement.

## Impact

- 9 planning documents all included a DynUNet cloud phase
- Issue #698 created for "DynUNet E2E plumbing test on RunPod" (now closed won't-do)
- User frustration: "can you focus and explain why would want run any of the non-heavy
  GPU jobs that I can run locally for fuck sake!"
- Trust erosion: if Claude doesn't understand WHY the user needs cloud GPU, what else
  is it missing about the project vision?

## What Should Be in CLAUDE.md (or a memory file)

```
## Cloud GPU Strategy (Non-Negotiable)

Local dev machine: RTX 2070 Super, 8 GB VRAM.

Models that FIT locally (NEVER run on cloud GPU):
- DynUNet: 3.5 GB VRAM — runs locally, always.
- SAM3 Vanilla: 2.9 GB inference, 3.5 GB training — runs locally.
- SegResNet: ~2 GB — runs locally.

Models that REQUIRE cloud GPU (24+ GB VRAM):
- SAM3 Hybrid: 7.18 GiB @ patch 64³ — OOMs on 8 GB GPU.
- VesselFM: ~10 GB estimated — OOMs on 8 GB GPU.
- Any future model > 7 GB VRAM.

Cloud GPU (RunPod) exists ONLY for the second group. NEVER burn credits on
models that run locally. If testing infrastructure, test with the SMALLEST
model that actually requires cloud — go straight to SAM3 hybrid.
```

## Resolution

1. Created this metalearning doc
2. Fixed plan v4.0: removed DynUNet phase, start directly with SAM3 hybrid
3. Closed issue #698 as won't-do
4. Will create P1 issue to add cloud GPU strategy to CLAUDE.md (#693 scope)
5. Will add local hardware specs to memory file

## Broader Pattern: Claude Doesn't Understand "Why"

This failure is an instance of a broader pattern: Claude excels at "how" (write
tests, fix code, create issues, harden configs) but struggles with "why" (why
does this feature exist, why does the user need this, what problem is being solved).

Without the "why", Claude produces technically correct but strategically wrong
outputs: a perfectly hardened DynUNet cloud pipeline that nobody needs.

The fix is not more metalearning docs — it's ensuring the "why" is in the
permanent context (CLAUDE.md, memory files) in a form that can't be missed.

## Cross-References

- `.claude/metalearning/2026-03-14-false-blockers-devex-failure.md` — same session
- `docs/planning/runpod-dev-verification-plan-for-realz-maybe.xml` — fixed plan
- Issue #693: CLAUDE.md refactor (progressive disclosure)
- Issue #698: DynUNet E2E on RunPod (closed won't-do)
- `memory/user_cloud_accounts.md` — RunPod balance, cloud account state
