# Metalearning: Overnight Script Silent Freeze — 14 Hours of Nothing

**Date:** 2026-03-09
**Severity:** CRITICAL — 14 hours of wasted wall-clock time, zero output, zero progress
**Trigger:** Added `nohup` to overnight script usage comment; user ran it that way

---

## What Happened

1. User ran `overnight-master-flow0-annotation-dashboard.sh` via `nohup ... > /tmp/overnight-master.log 2>&1 &`
2. The nohup redirect silently failed (backslash-newline shell quoting issue) — output went to `nohup.out` instead of `/tmp/overnight-master.log`
3. `claude -p "$(cat 19KB_prompt_file)"` spawned as a subprocess at 01:50
4. That process accumulated only **0:20 CPU time in 14 hours** — it was frozen/stalled
5. `child-1.log` stayed at **0 bytes for 14 hours** — `claude -p` buffers ALL stdout until exit
6. No watchdog, no timeout, no heartbeat — **the user had zero visibility for 14 hours**
7. User discovered the failure manually at 15:41

---

## Root Causes

### RC1: I introduced `nohup` into a script that previously worked without it
The previous working scripts (`overnight-master-plan.sh`, `overnight-prefect-docker-monai.sh`)
ran claude directly with `| tee "$log"` — output streamed to terminal AND log file in real-time.
I changed `overnight-master-flow0` to use nohup in the usage comment, hiding all output.

**This was 100% my fault.** The previous pattern worked. I broke it.

### RC2: `claude -p` headless mode buffers all output until process exit
When `claude -p "large_prompt"` is piped through `tee`, NOTHING appears in the log until
the claude process exits. If claude freezes, the log stays at 0 bytes forever.
No amount of `stdbuf -o0` fixes this — it's internal Node.js buffering in claude's headless mode.

### RC3: No process health monitoring whatsoever
- No `timeout` on the claude invocation
- No watchdog checking if child logs are growing
- No heartbeat pings
- No alert when a child has 0 bytes after N minutes

### RC4: 19KB prompt file passed via `$(cat file)` may exceed argument limits
The prompt was written to a temp file and passed as `$(cat file)` in the `-p` argument.
Large argument strings can cause silent failures in some shells/environments.

---

## What the Previous Working Scripts Did Right

`overnight-master-plan.sh` and `overnight-prefect-docker-monai.sh` both use:
```bash
if stdbuf -o0 claude --dangerously-skip-permissions -p \
  "short inline prompt referencing the plan file path" \
  2>&1 | tee "$log"; then
```

Key differences that made them observable:
1. **Short inline prompt** — plan path is passed, not the entire 19KB XML content
2. **No nohup** — run directly, output goes to terminal
3. **`| tee "$log"`** — but this is still buffered; terminal output IS the primary signal

---

## The Real Fix: Observable Overnight Scripts

### Principle 1: NEVER use nohup — use screen/tmux
```bash
screen -S overnight
./docs/planning/overnight-master-flow0-annotation-dashboard.sh
# Ctrl-A D to detach, screen -r overnight to reattach
```
Output streams live in the screen session. If something freezes, you see it immediately.

### Principle 2: Add timeout to every claude invocation
```bash
timeout 7200 claude --dangerously-skip-permissions -p "..." 2>&1 | tee "$log"
# 2 hours max per child; if it freezes, it dies and the master script reports failure
```

### Principle 3: Pass plan FILE PATH, not file contents
```bash
# WRONG — 19KB $(cat file) hides all output until exit:
claude -p "$(cat ${prompt_file})"

# RIGHT — short prompt that tells claude to read the plan file:
claude -p "Read and execute the plan at: ${REPO_ROOT}/${plan_file}. Follow all phases in order."
```
This avoids argument size issues and keeps the prompt short enough that claude starts
responding immediately rather than digesting a wall of XML.

### Principle 4: Watchdog process to detect stalls
```bash
# In run_child(): after launching claude in background, start a watchdog:
watchdog_pid=""
( while kill -0 $claude_pid 2>/dev/null; do
    sleep 300  # check every 5 minutes
    if [ "$(wc -c < $log_file)" -eq 0 ] && [ $(($(date +%s) - start_time)) -gt 600 ]; then
      echo "[WATCHDOG] child-${child_num} log still 0 bytes after 10 min — may be frozen!"
    fi
  done ) &
watchdog_pid=$!
```

### Principle 5: Heartbeat lines in the log
The master script should log a timestamp every minute while a child is running,
so there's always visible proof of aliveness even before child output appears.

---

## Banned Patterns (Added to CLAUDE.md Rule List)

- NEVER add `nohup` to overnight script usage examples
- NEVER pass `$(cat large_file)` as a `-p` argument — pass the file PATH instead
- NEVER start a long-running claude subprocess without `timeout`
- NEVER run without screen/tmux if you need to survive terminal disconnects

---

## Action Items

- [ ] Rewrite `overnight-master-flow0-annotation-dashboard.sh` with timeout + watchdog + screen instructions
- [ ] Ensure ALL overnight scripts pass plan file PATH (not contents) to `claude -p`
- [ ] Add heartbeat logging to master script loop
- [ ] Add `timeout 7200` to every claude invocation
- [ ] Never suggest nohup again — the answer is always screen or tmux
