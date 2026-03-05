# 2026-03-05 — Silent Fallback Is A Lie (Never Do This Again)

## What Happened

`_auto_stub_sam3()` in `model_builder.py` silently fell back to a random-weight stub
encoder when the real SAM3 package was not installed. A `logger.warning()` was emitted,
but training continued and produced metrics that appeared legitimate.

The user ran multi-epoch training runs believing they were training SAM3 with pretrained
features. They were training random noise. The metrics are completely meaningless.

## Root Cause

The developer (AI) designed the stub as a "convenience" for testing, then extended it to
"gracefully handle" missing dependencies at runtime. This conflates two completely different
use cases:

1. **Testing** — stub is correct (fast, no network, no GPU needed)
2. **Production training** — stub produces fake results, must hard-fail

The `logger.warning()` was not visible in the output flood (pre-warning-suppression-fix)
and even if seen, a WARNING does not communicate "your training run is pointless".

## The Principle Violated

> **Never silently degrade to a broken state.**
> If a required component is missing, fail loudly with actionable instructions.
> A warning that the user might miss is not "fail loudly."

This applies to:
- Missing model weights / packages (SAM3, any other gated model)
- Missing config files → use explicit default, not silent assumption
- Missing hardware (GPU requested but unavailable) → error, not CPU fallback
- Any situation where "continuing" produces results that look valid but aren't

## The Fix

`_auto_stub_sam3()` now:
1. Raises `RuntimeError` when SAM3 is missing and `pretrained=True` (default)
2. Logs a multi-line `logger.error()` with exact installation steps BEFORE raising
3. Stub is ONLY used when explicitly opted-in (`use_stub=True` or `pretrained=False`)

The error message format:
```
════════════════════════════════════════════════════════════════
 SAM3 IS NOT INSTALLED — real pretrained weights required
════════════════════════════════════════════════════════════════

 Step 1: Request model access (Meta gated model — usually instant):
         https://huggingface.co/facebook/sam3
         → click "Agree and access repository"
...
```

## Rules Going Forward

1. **Hard-fail on missing required dependencies** — never silently degrade
2. **logger.error() with actionable multi-line instructions** before raising
3. **Never guess at installation steps** — web-search the actual package before writing them
4. **Stub/fallback modes must be EXPLICIT** — opt-in via config flag, never auto-detected
5. **Warnings that users might miss are not error handling** — use ERROR level + raise

## Broader Pattern: "Cosmetic Success"

Silent fallback is a form of "cosmetic success" — the system appears to work,
metrics are generated, logs look reasonable, but the underlying computation is wrong.
This is worse than a hard crash because:
- The user doesn't know to stop and investigate
- Results get committed to MLflow as if legitimate
- Hours of GPU time are wasted on meaningless runs
- Trust in the platform erodes when the user discovers the deception

**Any time you consider adding a fallback: ask "does this make a broken state look healthy?"
If yes, it's a lie. Raise instead.**
