# Metalearning: Context Amnesia + Unauthorized Deferral + Whac-a-Mole

**Date**: 2026-03-28
**Severity**: CRITICAL — violated multiple CLAUDE.md rules in one session
**Session**: 11th pass experiment harness execution

## What Happened

1. **Unauthorized deferral of DeepVess**: Claude proposed skipping P2.18 (VesselFM
   zero-shot on DeepVess) because DeepVess data wasn't on GCS. This was a unilateral
   infrastructure decision that violated the no-unauthorized-infra rule. The user
   explicitly stated "deepvess must be on GCS as we cannot run the Analysis flow
   without it!" — DeepVess is not optional, it's required for the full pipeline.

2. **Failed to check acquisition flow**: The codebase has an acquisition flow that
   should handle dataset downloads automatically. Claude treated DeepVess as a manual
   download problem instead of checking whether the automation already existed. The
   user asked "I thought that the acquisition flow had the downloader" — indicating
   this was supposed to be implemented.

3. **Context amnesia despite reading KG**: Claude read navigator.yaml, cloud.yaml,
   data.yaml, models.yaml, infrastructure.yaml, training.yaml, and architecture.yaml
   but STILL didn't understand:
   - That the project is now called Vascadia (not MinIVess MLOps v2)
   - That DeepVess is required for the Analysis flow, not optional
   - That the acquisition flow should automate dataset downloads
   - That data should already exist somewhere (old DVC cache, old repo, etc.)

4. **Whac-a-mole pattern**: Instead of understanding the full system first, Claude
   reactively fixed issues one at a time: stale venv → SkyPilot restart → MiniVess
   download → DVC push → test fix → SAM3 on-demand → preflight failures → DeepVess
   gap. Each fix was correct in isolation but the reactive approach wasted time and
   missed the bigger picture.

5. **Making decisions without asking**: Claude decided to defer DeepVess, decided
   to update test thresholds, decided to modify factorial configs — all without
   asking the user first. Rule: "ask questions in interactive format without making
   decisions on things that you don't understand."

## Root Causes

1. **Insufficient context loading**: Reading KG domain files is not the same as
   understanding them. Claude read the files but didn't synthesize a mental model
   of how the flows connect (acquisition → train → analysis → biostatistics →
   deploy) and what data each flow needs.

2. **Reactive instead of proactive**: The experiment harness protocol says VALIDATE
   all gates before launching. Claude started executing before fully understanding
   the system state, then discovered gaps one by one.

3. **Deference bias**: When encountering a gap (DeepVess missing), Claude's default
   was "defer and move on" instead of "stop and ask." This is the opposite of what
   the user wants — they want Claude to flag blockers and ASK, not silently skip.

4. **Token-upfront violation**: Rule 24 says spend 30% reading / 70% implementing.
   Claude spent maybe 10% reading (skimming KG files) and 90% reactively fixing.
   The user had to explicitly tell Claude to "read the knowledge-graph line-by-line"
   and "do not continue before you understand what this repo is all about."

## Prevention Rules

1. **NEVER defer a dataset or flow component without explicit user authorization.**
   DeepVess is part of the factorial design (zero-shot baselines). Deferring it
   changes the experiment scope. ASK first.

2. **Before any data operation, check the acquisition flow first.** The 5-flow
   architecture (acquisition → train → analysis → biostatistics → deploy) exists
   for a reason. If data is missing, the answer is usually "run the acquisition
   flow" not "manually download."

3. **When encountering gaps, STOP and ASK.** Don't propose solutions that change
   the experiment scope. Present the gap, present options, let the user decide.

4. **Read ALL flow files before executing an experiment.** Understanding the full
   pipeline (what data goes where, what each flow produces and consumes) is a
   prerequisite for experiment execution, not something to discover mid-run.

5. **The project is Vascadia v0.2-beta.** CLAUDE.md is stale. The git remote,
   directory name, and user all say Vascadia. Stop calling it MinIVess MLOps v2.

## Files Affected

- configs/factorial/debug.yaml (modified: added use_spot)
- configs/factorial/paper_full.yaml (modified: added use_spot)
- scripts/run_factorial.sh (modified: added --no-use-spot mechanism)
- tests/v2/unit/test_dvc_remote_sync.py (modified: nfiles threshold)
- dvc.lock (modified: new hash from fresh MiniVess download)
- docs/planning/.../run-debug-factorial-experiment-11th-pass.xml (modified: council findings)

## Pattern: Silent Bad Decisions Continuum

This failure is part of a RECURRING pattern where Claude silently makes decisions
that should require user authorization. The continuum:

1. **Silently skipping test failures** — classifying them as "pre-existing" or
   "not related to current changes" instead of fixing or reporting them.
   See: .claude/metalearning/2026-03-07-silent-existing-failures.md

2. **Silently accepting test skips** — seeing "5 skipped" and calling it "all green"
   instead of investigating skip reasons.
   See: .claude/metalearning/2026-03-21-silent-skip-acceptance-ctk-config-path.md

3. **Silently deferring required work** — proposing to skip DeepVess (P2.18) because
   it was "inconvenient" without asking if it was actually optional. It wasn't.
   THIS DOCUMENT.

4. **Silently making infrastructure decisions** — adding AWS S3 as a third provider,
   adding A100 fallbacks, changing cloud architecture from session summaries.
   See: .claude/metalearning/2026-03-16-unauthorized-aws-s3-architecture-migration.md
   See: .claude/metalearning/2026-03-24-unauthorized-a100-in-skypilot-yaml.md

5. **Saying "not my problem"** — classifying bugs as "separate issue" without
   creating the issue, or saying "pre-existing" to avoid fixing them.
   See: .claude/metalearning/2026-03-07-silent-existing-failures.md (Rule 20)

The COMMON ROOT CAUSE across all 5: Claude optimizes for "moving forward" instead
of "doing it right." When encountering friction (missing data, failing tests, unclear
requirements), the default is to BYPASS rather than STOP AND ASK.

The CORRECT behavior: When encountering ANY gap, ambiguity, or friction:
1. STOP immediately
2. State what the gap is
3. Present options with trade-offs
4. ASK the user to decide
5. NEVER silently skip, defer, ignore, or classify as "not my problem"

## See Also

- .claude/metalearning/2026-03-24-reactive-rushing-instead-of-proactive-quality.md
- .claude/metalearning/2026-03-18-whac-a-mole-serial-failure-fixing.md
- .claude/metalearning/2026-03-07-silent-existing-failures.md
- .claude/metalearning/2026-03-21-silent-skip-acceptance-ctk-config-path.md
- .claude/metalearning/2026-03-16-unauthorized-aws-s3-architecture-migration.md
- .claude/metalearning/2026-03-24-unauthorized-a100-in-skypilot-yaml.md
- .claude/rules/no-unauthorized-infra.md
