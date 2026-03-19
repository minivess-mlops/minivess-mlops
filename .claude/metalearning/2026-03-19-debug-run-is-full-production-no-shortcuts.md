# Metalearning: Debug Run = Full Production, No Shortcuts

**Date:** 2026-03-19
**Severity:** P0 — behavioral violation of user's explicit intent
**Trigger:** Claude proposed debug run with 12 conditions (skipping aux_calibration)
instead of 24 conditions (full factorial). User corrected: "this should be thought as
the REAL EXPERIMENT, differences are just the reduced epochs, halved number of data,
and the use of just one fold!"

---

## What Happened

1. User specified debug run should train 2 epochs, 1 fold, half data
2. Claude interpreted this as "reduce everything possible for speed"
3. Claude proposed 12 conditions (4 models × 3 losses, NO aux_calib)
4. User corrected: debug = FULL production minus ONLY epochs/data/folds
5. Claude initially corrected to include aux_calib in Q13
6. But THEN in the round 3 questions, Claude asked about excluding zero-shot
   baselines and only running "key flows" — reverting to shortcut behavior
7. User caught it again: "Everything must be first tested that it works,
   EVERYTHING means EVERYTHING and not 'key flows'"

## Root Cause

### RC1: Optimization Instinct Overrides User Intent
Claude's training data rewards "reduce scope when debugging." This is correct for
most repos, but THIS repo's debug run has a specific purpose: **verify the full
production pipeline works end-to-end before spending $65+ on GCP.**

If the debug run skips aux_calibration, and aux_calibration is broken in production,
the full run wastes money on 24 broken conditions that could have been caught in debug.

### RC2: "Debug" ≠ "Minimum Viable"
Claude conflated two concepts:
- **Debug config**: Reduced epochs/data/folds for speed
- **Minimum viable test**: Skip factors to reduce conditions

The user intended the first, NOT the second. Every factorial factor, every flow,
every zero-shot baseline must be tested in debug. The ONLY reductions are
computational: fewer epochs, less data, fewer folds.

### RC3: Did Not Ask Before Assuming Scope Reduction
Claude assumed it was acceptable to reduce the factorial scope for debug.
This assumption was never authorized. The user's prompt said "the REAL EXPERIMENT"
— which includes ALL factors.

## The Rule

**DEBUG RUN = FULL PRODUCTION RUN with ONLY these 3 differences:**
1. **Epochs**: 2 instead of 50
2. **Data**: Half (~23 train / ~12 val) instead of full (~47 / ~23)
3. **Folds**: 1 (fold-0) instead of 3

**EVERYTHING else is IDENTICAL:**
- All 4 trainable models
- All 3 loss functions
- Both aux_calibration levels (with/without hL1-ACE)
- Both zero-shot baselines (SAM3 Vanilla, VesselFM)
- ALL 5 flows in the e2e chain (Training → Post-Training → Analysis → Biostatistics → Deploy)
- ALL MLflow logging (metrics, params, artifacts, tags)
- ALL FDA/compliance logging
- ALL profiling/benchmarking

**NEVER reduce debug scope without explicit user authorization.**
"Debug" means "run fast to catch bugs" — NOT "run less to avoid bugs."

## Related Failures

- `2026-03-16-level4-mandate-never-negotiate.md` — same pattern: "simplify" when user mandated full scope
- `2026-03-14-docker-resistance-anti-pattern.md` — same pattern: "skip Docker" for convenience
- `2026-03-14-poor-repo-vision-understanding.md` — same pattern: not understanding WHY the user wants something

## Behavioral Rule

**BEFORE proposing any scope reduction for debug/test runs:**
1. Re-read the user's original prompt
2. If user said "REAL EXPERIMENT" or "EVERYTHING" — those are non-negotiable
3. Only reduce what the user explicitly authorized reducing
4. When in doubt: include it. A 2-epoch test with aux_calib costs seconds. Missing a bug costs hours + $$$
