# METALEARNING: CLAUDECODE Nested Session Silent Failure + Observability Collapse

**Date**: 2026-03-09
**Severity**: CRITICAL — 7+ minutes of zero observable output, user rightly furious
**Root cause**: Two compounding failures — silent subprocess exit + time-delayed watchdog

---

## What Happened

1. User started `overnight-master-flow0-annotation-dashboard.sh` from inside a Claude
   Code terminal session.
2. Claude Code sets `CLAUDECODE` env var in its shell environment.
3. Child processes (including the bash script) inherit this env var.
4. Inside the script, `claude --dangerously-skip-permissions -p "..."` launched.
5. The `claude` CLI detects `CLAUDECODE` in the environment — its "nested session guard"
   refuses to launch and exits immediately.
6. **Exit was silent**: no stderr message reached the terminal. The 0-byte log was the
   only signal.
7. The watchdog was designed to check at 10 minutes. The heartbeat fires every 60s but
   only proves the master script is alive — not that the child did anything useful.
8. The cron monitor checked every 5 minutes.
9. **Net result: 7+ minutes passed with zero detection of a clean-exit subprocess failure.**

---

## The Two Failures

### Failure 1: CLAUDECODE Silent Exit

The `claude` CLI exits with a non-zero code when `CLAUDECODE` is set, but produces no
output that reaches the tee pipeline. The detection signal (non-zero exit) existed — it
was just never checked immediately.

**Fix**: `unset CLAUDECODE` in the subshell before invoking claude.

```bash
(
  cd "${REPO_ROOT}"
  unset CLAUDECODE          # ← THIS LINE IS MANDATORY
  timeout "${CHILD_TIMEOUT_SEC}" claude \
    --dangerously-skip-permissions \
    ...
)
```

This fix is in `overnight-master-flow0-annotation-dashboard.sh` as of 2026-03-09.
It MUST be applied to every overnight runner script.

### Failure 2: Observability Architecture Was Time-Based, Not Event-Based

The watchdog checked log file size on a timer (10-minute first check). This is
fundamentally wrong for detecting fast-exit failures.

A subprocess that exits in 1 second with 0 bytes and exit code 1 looks identical to
a subprocess that hangs silently for 10 minutes — until the timer fires.

**The correct design**: check exit code immediately after subprocess exits, BEFORE
any time-based watchdog logic. If claude exits in < 30 seconds with 0 bytes, that
is ALWAYS a hard error.

---

## What Should Have Been Done: Immediate Exit Detection

The `||` error capture pattern `|| exit_code=$?` captures the final exit code, but
the code then waits for watchdog/heartbeat cleanup before reporting. The error was
not surfaced until after those background processes were killed.

**Correct pattern**: Check for fast-fail immediately after the subshell returns.

```bash
local start_time exit_code=0
start_time=$(date +%s)

(
  cd "${REPO_ROOT}"
  unset CLAUDECODE
  timeout "${CHILD_TIMEOUT_SEC}" claude ...
) 2>&1 \
  | tee "${log_file}" \
  | jq -rj '...' \
  || exit_code=$?

local elapsed=$(( $(date +%s) - start_time ))
local log_bytes=$(wc -c < "${log_file}" 2>/dev/null || echo 0)

# IMMEDIATE fast-fail detection — catches CLAUDECODE and other silent exits
if [ "${elapsed}" -lt 30 ] && [ "${log_bytes}" -lt 100 ]; then
  log "FAST-FAIL DETECTED: child-${child_num} exited in ${elapsed}s with ${log_bytes} bytes"
  log "  Likely causes:"
  log "    1. CLAUDECODE env var set (nested session guard) — unset CLAUDECODE"
  log "    2. claude CLI not found or auth failure"
  log "    3. Plan file missing"
  log "  Check stderr: the error message may be in ${log_file}"
  log "  Captured stderr (if any):"
  head -20 "${log_file}" | tee -a "${LOG_DIR}/master.log" || true
  FAILED_CHILDREN+=("${child_num}:FAST_FAIL(${exit_code}/${elapsed}s/${log_bytes}b):${description}")
  return 1
fi
```

---

## Required Checks at Script Startup

Add to the prerequisite block of every overnight runner:

```bash
# CLAUDECODE guard — fail early if running inside Claude Code terminal
if [ -n "${CLAUDECODE:-}" ]; then
  log "FATAL: CLAUDECODE env var is set — you are running inside a Claude Code session."
  log "  This script CANNOT be run from within Claude Code's terminal."
  log "  Run from a plain terminal: gnome-terminal, screen, tmux, or SSH."
  log "  Or unset CLAUDECODE manually: unset CLAUDECODE && ./this-script.sh"
  exit 1
fi
```

This catches the problem before any child runs, not 10 minutes into child-1.

---

## Stderr Capture Pattern

The current pipeline `2>&1 | tee | jq` merges stderr into stdout and pipes through jq.
If `claude` exits with an error message, jq will try to parse it as JSON, fail silently,
and not display it.

**Fix**: Capture stderr separately or check the raw log file for non-JSON lines.

```bash
# Better: separate stderr capture
(
  cd "${REPO_ROOT}"
  unset CLAUDECODE
  timeout "${CHILD_TIMEOUT_SEC}" claude \
    --dangerously-skip-permissions \
    --output-format stream-json \
    --verbose \
    --include-partial-messages \
    -p "..." \
    2>"${log_file}.stderr"
) \
  | tee "${log_file}" \
  | jq -rj '...' \
  || exit_code=$?

# Always show stderr if non-empty
if [ -s "${log_file}.stderr" ]; then
  log "STDERR from child-${child_num}:"
  cat "${log_file}.stderr" | tee -a "${LOG_DIR}/master.log"
fi
```

---

## Vision Failure: High-Observability Plan Was Not Implemented

The user explicitly requested "full observability and realizing when things are not
progressing." The 10-minute watchdog is not full observability — it is a slow last resort.

Full observability means:
1. **Preflight**: Check `CLAUDECODE` before ANY child runs. Fail fast with clear error.
2. **Launch**: Immediately after subprocess exit, check elapsed time + log bytes.
3. **Stderr**: Always capture and display stderr from claude subprocess.
4. **Exit code**: A non-zero exit from claude is ALWAYS a hard error — log it immediately.
5. **Continuous**: The cron monitor (every 5 min) is for human reassurance, not detection.
   Primary detection must be event-driven (subprocess exit), not time-driven.

---

## Lessons

1. **`unset CLAUDECODE` is MANDATORY** in every overnight runner script, in the subshell,
   before the claude invocation. This is now in the template.

2. **Fast-fail detection is MANDATORY**: check elapsed + bytes immediately after subprocess
   returns. < 30s with < 100 bytes = hard error.

3. **CLAUDECODE preflight check is MANDATORY**: if `$CLAUDECODE` is set in the master
   script environment, STOP with a clear error before running any child.

4. **Stderr must be captured and displayed**, not silently discarded through jq.

5. **The watchdog is a last resort**, not primary detection. Treat it as a backup, not
   the main observability mechanism.

6. **Never let a 0-byte result go unreported for more than the first check interval.**
   The first check should fire within 60 seconds of subprocess exit, not 10 minutes.

---

## Affected Files (All need `unset CLAUDECODE` + fast-fail + CLAUDECODE preflight)

- `docs/planning/overnight-master-flow0-annotation-dashboard.sh` ← fixed 2026-03-09
- `docs/planning/overnight-master-plan.sh` ← PENDING
- `docs/planning/overnight-prefect-docker-monai.sh` ← PENDING
- `docs/planning/overnight-script-consolidation.sh` ← PENDING (if exists)
- `docs/planning/batch-script-executor-monai-and-scripts.sh` ← PENDING
- `.claude/skills/overnight-runner/templates/batch-runner.sh` ← template, fix here = fix everywhere

---

## Overengineering Post-Mortem: The Heartbeat/Watchdog Complexity Was Wrong

The `overnight-master-flow0-annotation-dashboard.sh` was rewritten to add heartbeat
background processes, a watchdog auto-kill, FAILED_CHILDREN[] arrays, start_heartbeat()
and start_watchdog() functions, STALL_KILL_MIN controls, SKIP_END support, and a
complex run_child() abstraction. Compare to the simpler `overnight-master-plan.sh`
which is just a for loop.

### What the overengineered version got RIGHT

- `--output-format stream-json --verbose --include-partial-messages` → real-time output
- `timeout N` → hard-kill frozen sessions
- Pass plan FILE PATH not `$(cat plan.xml)` → avoids 14h buffering freeze
- `tee logfile` → save session for replay
- `jq` filter → human-readable terminal output
- `SKIP_TO` → resume from a specific child

### What the overengineered version got WRONG

- **Background heartbeat processes** introduced new failure modes. When the main
  subshell exits immediately (0 bytes, fast exit), the heartbeat PID is still alive
  and continues firing. Result: "heartbeat running 10m... 11m... 12m" printed in
  3 seconds because multiple accumulated heartbeats fire simultaneously when the
  wait() call returns. This is MORE confusing than the simple loop.

- **Watchdog fires at 10 minutes** — the very problem we were trying to solve. The
  simple pattern (streaming output to terminal) is its own observability. If claude
  is running, you see tokens. If it's not, the terminal is silent and you know
  immediately.

- **Background process management adds race conditions** — heartbeat_pid and
  watchdog_pid are killed after the subshell exits, but the kill/wait sequence can
  hang or produce spurious errors.

- **Multiple claude processes** — the ps aux output showed 4 simultaneous claude
  processes with significant CPU. This was caused by: (a) multiple invocations of
  the master script, (b) heartbeat/watchdog PIDs keeping background state alive
  after child exits. The simple loop has exactly 1 claude process at a time.

- **Zero observability improvement** — the heartbeat proved the MASTER SCRIPT was
  alive, not that the CHILD CLAUDE was producing output. The user was right to be
  furious: all that machinery, and still 0 useful signal for 24+ minutes.

### The actual fix

The simple for-loop pattern IS the observability:

```bash
for plan in "${PLANS[@]}"; do
  timeout "${CHILD_TIMEOUT_SEC}" claude \
    --dangerously-skip-permissions \
    --output-format stream-json \
    --verbose \
    --include-partial-messages \
    -p "Read and execute: ${plan}" \
    2>&1 | tee "${log}" \
         | jq -rj 'select(.type=="stream_event" and (.event.delta.type?=="text_delta")) | .event.delta.text'
done
```

- If claude is running → you see tokens streaming in the terminal → you know it's alive
- If claude is silent → terminal is silent → you know something is wrong within seconds
- If claude exits fast → the jq pipeline closes → the for loop moves on → you see the next child start
- No background processes, no race conditions, no PID management

### Lesson: Resist infrastructure complexity for shell scripts

The overnight runner is a shell script, not a production service. The "infrastructure"
(heartbeat, watchdog, PID management) that seems like robustness is actually fragility.
The simple version that was already working (`overnight-master-plan.sh`) needed only
three changes from the broken stdbuf version:
1. `--output-format stream-json --verbose --include-partial-messages`
2. Pass plan file PATH not `$(cat contents)`
3. `timeout N`

Those three changes fix 100% of the observed failures. The rest was overengineering.

## See Also

- `.claude/metalearning/2026-03-09-overnight-script-silent-freeze.md` — 14h silent freeze
  from `$(cat plan.xml)` buffering pattern
- `.claude/scripting-research.md` — comprehensive headless claude patterns research
