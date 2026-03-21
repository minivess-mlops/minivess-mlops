# Metalearning: Silent Skip Acceptance — CTK Config Path (2026-03-21)

## What Happened

Issue #884 asked to fix a `pytest.skip` for CTK config.toml. The test was skipping with
"CTK config.toml not found — skipping hook check" on the dev machine.

Instead of investigating WHY the config wasn't found (nvidia-ctk v1.18.2 was installed),
Claude accepted the skip as "hardware-gated" and added logic tests around it. This is
the EXACT anti-pattern documented in `.claude/metalearning/2026-03-07-silent-existing-failures.md`.

## Root Cause

The test hardcoded `/etc/nvidia-container-toolkit/config.toml` but the actual config
was at `/etc/nvidia-container-runtime/config.toml`. A 5-second `find` command would
have revealed this. Instead, Claude:

1. Read the test and saw "CTK config.toml not found"
2. Classified the skip as "acceptable hardware-gated"
3. Added parser logic tests that work without the config file
4. Moved on without investigating whether CTK was actually installed

This violated:
- **Rule #20**: Zero Tolerance for Observed Failures
- **Rule #28**: Zero Silent Skips — "every SKIPPED test is a bug hiding as a skip"
- **Rule #12**: Never Confabulate — Claude assumed CTK wasn't installed without checking

## The Fix

```python
# WRONG — hardcoded single path
config_path = Path("/etc/nvidia-container-toolkit/config.toml")

# RIGHT — check both known paths
_CTK_CONFIG_PATHS = [
    Path("/etc/nvidia-container-toolkit/config.toml"),
    Path("/etc/nvidia-container-runtime/config.toml"),
]
```

Result: 5 passed, 0 skipped (was: 4 passed, 1 skipped).

## Anti-Pattern: "Acceptable Skip" Classification

When Claude encounters a test skip, the reflex is to classify it:
- "Hardware-gated → acceptable"
- "Module not installed → needs install"
- "Cloud credentials → auto-skip OK"

The problem: this classification happens WITHOUT investigation. The correct
protocol is:

1. **WHY** is it skipping? Run the diagnostic command.
2. **CAN** we fix it? Check if the tool/file/resource exists at a different path.
3. **SHOULD** we fix it? Only skip if truly impossible (no GPU, no cloud creds).

"It's hardware-gated" is a conclusion, not an investigation. The investigation
MUST happen FIRST.

## Rule for Future Sessions

**Before classifying ANY skip as "acceptable":**
1. Run `which <tool>`, `dpkg -l | grep <tool>`, or `find /etc -name <config>`
2. If the tool IS installed, the skip is a bug — fix the test
3. If the tool is NOT installed, ask the user if we should install it
4. Only then classify as "acceptable" (with evidence logged)

"Pre-existing" and "acceptable skip" are both BANNED phrases without evidence.
