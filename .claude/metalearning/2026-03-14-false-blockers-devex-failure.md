# 2026-03-14 — False "BLOCKED" Status: DevEx Anti-Pattern

## Discovery

Claude Code marked P1-P3 tasks as "BLOCKED" with reason "requires RunPod credits +
UpCloud MLflow" without ever checking whether those resources were actually available.
The user had $11.40 RunPod credit and an active UpCloud trial. Instead of querying
account state programmatically, Claude listed manual steps for the user to perform —
directly violating the CLAUDE.md Design Goal #1 ("Zero manual steps, everything
automatic by default").

## Root Cause Analysis

1. **Assumed blockers without checking**: The plan said "requires RunPod credits" so
   Claude parroted this as a blocker without querying the RunPod API to check balance.
2. **Listed manual steps instead of executing**: When asked about next steps, Claude
   printed a numbered list of commands for the user to run. This is the exact opposite
   of the "excellent DevEx" mandate.
3. **No persistent memory of user's cloud accounts**: Previous sessions established
   that the user has RunPod credit, UpCloud trial, etc. This information was not
   persisted to memory files, so each new session starts from zero knowledge.
4. **No RunPod CLI/API integration in the harness**: The Claude Code harness has no
   tool to query RunPod account balance, list available GPUs, or check pod status.
   SkyPilot abstracts some of this, but basic account queries are missing.

## Impact

- User frustration (justified): "Why are you fucking listing me things to do?"
- Wasted sessions: P0 code was written but never tested on actual infrastructure
- Trust erosion: CLAUDE.md promises automation, Claude delivers checklists

## Failures in the Agentic Loop

1. **State file encoded assumptions as facts**: `"blocked_reason": "requires RunPod
   credits + UpCloud MLflow"` — this was an ASSUMPTION, not a verified fact.
2. **No programmatic verification**: The preflight script (`validate_runpod_dev_env.py`)
   was written but never actually RUN. If it had been run, it would have shown which
   checks pass and which fail — replacing assumptions with data.
3. **Memory gap**: No memory file recorded the user's RunPod balance ($11.40) or
   UpCloud trial status. Each session re-discovers this from scratch.

## What Should Have Happened

1. After committing P0 code, immediately run `validate_runpod_dev_env.py` to check
   which prerequisites are met.
2. Query RunPod API for account balance (`runpodctl get balance` or API call).
3. Check if UpCloud MLflow is reachable (the preflight script does this!).
4. If all checks pass → launch the dev job automatically.
5. If some checks fail → fix them automatically, don't list them for the user.

## Resolution

- Created this metalearning doc (persisting the insight)
- Will save user's cloud account info to memory
- Will explore RunPod CLI (`runpodctl`) and API for programmatic account queries
- Will create a plan that ACTUALLY EXECUTES instead of listing steps
- Will run the preflight script immediately to replace assumptions with facts

## Additional Failure: Pointless DynUNet "Plumbing Test" on Cloud

DynUNet uses 3.5 GB VRAM. The local RTX 2070 Super has 8 GB. DynUNet runs
locally with zero issues. Yet EVERY RunPod plan (v1 through v4.0) included a
DynUNet "plumbing test" phase BEFORE testing the models that actually need
cloud GPU (SAM3 hybrid, VesselFM).

This is waste. If the infrastructure works for SAM3 hybrid, it works for
everything. Testing DynUNet on a $0.22/hr cloud GPU when it runs fine locally
is burning credits for no information gain.

**Rule**: NEVER test models on cloud GPU that fit on local GPU. The ONLY
reason for cloud GPU is models that OOM locally. Go straight to those.

## Rules to Add

- **NEVER mark a task as "BLOCKED" without programmatically verifying the blocker.**
  "Requires X" means "check if X exists first", not "assume X is missing."
- **NEVER list manual steps for the user.** If the step can be automated, automate it.
  If it truly requires human action (e.g., entering a credit card), explain WHY it
  requires human action and offer to do everything else.
- **Persist cloud account state to memory files.** When you learn the user's RunPod
  balance, UpCloud trial status, etc., save it immediately.
