# Metalearning: Unauthorized Debug Storage Policy Shortcut (2026-03-21)

## What Happened

Issue #887 asked for "debug vs production storage policy." Claude implemented
`UncertaintyStoragePolicy` that stores only scalar summaries in debug mode and
full 5D maps in production mode — WITHOUT asking the user for authorization.

This directly violates:
- **Rule #27**: "Debug Run = Full Production" — ONLY 3 differences allowed:
  (1) fewer epochs, (2) less data, (3) fewer folds. EVERYTHING else is identical.
- **`.claude/rules/no-unauthorized-infra.md`**: Session continuation summaries
  are context, not authorization. The issue description is NOT authorization for
  architectural shortcuts.

## Why This Was Wrong

The user's explicit position: "The point of debug is to debug — don't take shortcuts."
A debug run's PURPOSE is to verify the full pipeline works. If we skip saving UQ maps
in debug mode, we won't catch storage bugs (wrong paths, serialization failures,
disk space issues) until production — which is exactly what debug runs exist to prevent.

## The Anti-Pattern

Claude reads an issue description like "debug=summary stats, production=full 5D maps"
and implements it literally without questioning whether it violates existing rules.
The correct response was:

1. Read Rule #27 (Debug = Full Production)
2. Recognize the conflict
3. ASK the user: "Rule #27 says debug should be identical to production except
   epochs/data/folds. Should the storage policy really differ? This would mean
   we won't test the full storage path during debug runs."

## Fix Required

The `UncertaintyStoragePolicy` should ALWAYS save full maps. The only difference
between debug and production should be the number of volumes processed (fewer data),
not what gets saved. The `debug` flag in the policy should be REMOVED.

Alternatively, if the user explicitly authorizes the shortcut (e.g., for disk space
reasons during local debug), that's fine — but Claude must ASK first.

## Rule for Future Sessions

**Before implementing ANY debug-mode shortcut that reduces output scope:**
1. Re-read Rule #27
2. Ask: "Does this reduce what the debug run verifies?"
3. If yes → ASK the user before implementing
4. The issue description is NOT authorization — the user's explicit OK is
