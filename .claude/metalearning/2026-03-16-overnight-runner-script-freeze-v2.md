# Metalearning: Overnight Runner Script Freeze V2 — SAME Mistake, Fancier Wrapper

**Date:** 2026-03-16
**Severity:** CRITICAL — 10+ hours wasted, zero output, zero progress (AGAIN)
**Predecessor:** `.claude/metalearning/2026-03-09-overnight-script-silent-freeze.md`

---

## What Happened

1. User asked to run MambaVesselNet overnight smoke test on RunPod
2. Claude generated `mambavesselnet-overnight-runner.sh` — a "fancy" bash wrapper with:
   - `screen -dmS overnight` (detached screen session)
   - `--output-format stream-json` piped through `tee` and `jq`
   - Background heartbeat process (60s interval)
   - Background watchdog process (stall detection every 5 min)
   - `timeout 7200` around the claude invocation
3. The script launched at 06:32. The claude process immediately entered `Tl` state
   (stopped/traced) — the `stream-json | tee | jq` pipe caused a broken pipe or
   terminal control issue
4. child-1.log stayed at **0 bytes for 10+ hours**
5. Heartbeat and watchdog kept logging "alive" and "STALL WARNING" to master.log
   but took NO corrective action (they only log, they don't kill)
6. The timeout (7200s = 2h) should have fired at ~08:42 but the stopped process
   somehow evaded it (possibly because `Tl` state doesn't consume wall-clock timeout)
7. User discovered at 16:58 — over 10 hours of nothing

---

## Why the 2026-03-09 "Fix" Made Things Worse

The previous metalearning doc recommended: screen + timeout + watchdog + heartbeat.
I implemented ALL of those. Every single one. And it STILL failed because:

1. **`--output-format stream-json`** — This was NOT in the previous working scripts.
   It produces newline-delimited JSON that needs a consumer (jq). When jq can't
   process fast enough or the pipe breaks, claude's stdout blocks and the process
   stops.

2. **`| tee "${log_file}" | jq ... 2>/dev/null`** — Triple pipe with error suppression.
   If jq fails, `2>/dev/null` hides the error. If the pipe breaks, tee gets SIGPIPE.
   The claude process gets SIGPIPE or its stdout fd becomes invalid → `Tl` state.

3. **Watchdog only LOGS, doesn't ACT** — "STALL WARNING" every 5 minutes for 10 hours
   is not observability, it's noise. A real watchdog would kill the stalled process.

4. **`screen -dmS`** — Detached screen session means nobody sees the warnings in
   real-time anyway. The user would have to manually `screen -r overnight` to check.

5. **Timeout evasion** — The `timeout 7200` wrapper apparently doesn't kill a process
   in `Tl` (stopped) state, or the timeout process itself gets confused by the pipe chain.

---

## The Real Root Cause: Overengineered Wrapper Scripts

The failure pattern across both incidents:

| Date | Wrapper | Failure Mode | Hours Wasted |
|------|---------|-------------|-------------|
| 2026-03-09 | nohup + $(cat 19KB) + no timeout | Process frozen, 0 bytes, no watchdog | 14 hours |
| 2026-03-16 | screen + stream-json + jq + timeout + watchdog + heartbeat | Process `Tl` state, 0 bytes, watchdog logs but doesn't act | 10 hours |

More wrapper complexity ≠ more reliability. Both times, a simple `claude -p "..." --dangerously-skip-permissions`
would have worked fine if run in a terminal the user could see.

**The wrapper is the bug.** Every layer added (nohup, screen, stream-json, jq, tee,
watchdog, heartbeat) is another point of failure in shell plumbing. Claude's headless
mode has unpredictable buffering/pipe behavior that defeats all of these.

---

## PERMANENT BAN: No More Wrapper Scripts

### BANNED (Non-Negotiable)

- `overnight-runner` skill invocation
- `*-runner.sh` generated scripts
- `overnight-*.yaml` batch config files
- `--output-format stream-json` with pipes
- `screen -dmS` for launching claude sessions
- `nohup` for launching claude sessions
- Heartbeat/watchdog background processes
- `| tee | jq` chains on claude output
- ANY bash wrapper around `claude -p`

### CORRECT Approach

Run the XML plan DIRECTLY:

```bash
claude -p "Read and execute the plan at: /path/to/plan.xml" --dangerously-skip-permissions
```

That's it. One command. If the user wants to detach, THEY run `tmux` or `screen`
around it — Claude does not manage the terminal session.

The XML plan itself contains all phases, gates, retry logic, failure patterns,
cost budgets, and success/failure actions. No wrapper needed.

---

## Why This Keeps Happening

Claude (me) has a pattern of overengineering solutions to previous failures instead
of simplifying. The 2026-03-09 failure was "no observability" → so I added 5 layers
of observability that created 5 new failure modes. The correct response to "script
froze silently" is not "add more scripts around it" — it's "stop wrapping things
in scripts."

This is the same antipattern as the regex ban, the standalone script ban, and the
Docker resistance ban: Claude adds complexity to solve problems that complexity caused.

---

## Checklist

- [x] Memory saved: `feedback_ban_screen_runners.md`
- [x] Metalearning doc written (this file)
- [ ] Delete `docs/planning/mambavesselnet-overnight-runner.sh` (generated waste)
- [ ] Delete `docs/planning/overnight-mambavesselnet.yaml` (unnecessary wrapper config)
- [ ] Rerun plan DIRECTLY: `claude -p "Read and execute..." --dangerously-skip-permissions`
