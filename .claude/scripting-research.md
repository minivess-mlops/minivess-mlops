# Claude Code Scripting: Best Practices Research

> Research compiled 2026-03-09 for:
> - MinIVess MLOps overnight-runner skill design
> - Slides: "Agentic Coding: from Vibes to Production"
>
> Sources: official Claude Code docs, Steve Kinney course, Paige Niedringhaus blog,
> YK's 32 tips, AojdevStudio gist, Amber Jain headless article, Claude docs (headless, common-workflows)

---

## Part 1: The Headless Mode Primer

### What `-p` actually does

`claude -p "prompt"` runs Claude non-interactively and exits when done. Called "headless mode"
historically; now officially called the **Agent SDK CLI**. Everything works the same.

```bash
claude -p "Fix the bug in auth.py"                     # simplest form
claude -p "Summarize" --output-format json              # structured output
claude -p "Analyze" --output-format stream-json         # streaming tokens
claude -p "Fix tests" --allowedTools "Bash,Read,Edit"   # auto-approve tools
claude -p "Continue review" --continue                  # resume last session
claude -p "Continue review" --resume "$session_id"      # resume specific session
```

### Output formats — the most important flag to understand

| Format | Behavior | Best for |
|--------|----------|---------|
| `text` (default) | Plain text, **fully buffered until exit** | Simple piping |
| `json` | JSON envelope, **fully buffered until exit** | Structured results |
| `stream-json` | Newline-delimited JSON events, **streams as tokens are generated** | Observability, overnight scripts |

**Critical insight:** `text` and `json` produce ZERO output until the entire session ends.
This is why `child-1.log` stayed at 0 bytes for 14 hours. `stream-json` streams in real-time.

### The fix for the 14-hour silent freeze

```bash
# WRONG — buffers everything until exit, shows nothing if frozen:
claude -p "run plan" 2>&1 | tee child-1.log

# RIGHT — streams every token in real-time, visible immediately:
claude -p "run plan" \
  --output-format stream-json \
  --verbose \
  --include-partial-messages \
  2>&1 | tee child-1.log
```

To show just the text while also saving full JSON:
```bash
claude -p "run plan" \
  --output-format stream-json \
  --verbose \
  --include-partial-messages \
  2>&1 | tee child-1.log | jq -rj '
    select(.type=="stream_event")
    | select(.event.delta.type?=="text_delta")
    | .event.delta.text'
```

---

## Part 2: Prompt Passing Patterns

### Four patterns, in order of correctness

**Pattern A: Inline (good for short prompts)**
```bash
claude -p "Read and execute the plan at: /repo/docs/planning/child-01.xml"
```
- Claude reads the file itself → output starts immediately
- Short prompt = no buffering delay
- **Preferred for overnight scripts** (no temp file, no size limit issues)

**Pattern B: File path reference (good for structured plans)**
```bash
claude -p "Read the plan at: ${PLAN_FILE} and execute all phases"
```
- Tells Claude where the plan is — Claude's Read tool fetches it
- Clean separation of prompt and plan content

**Pattern C: stdin pipe (good for processing data)**
```bash
cat error.txt | claude -p "Explain this build error"
cat code.py   | claude -p "Review this for bugs" --output-format json | jq '.result'
```
- Stdin + `-p` work together: stdin is data, `-p` is instruction

**Pattern D: Large prompt via `$(cat file)` — AVOID**
```bash
# BANNED: 19KB XML as $(cat) argument → buffers all output until exit
claude -p "$(cat 19kb-plan-file.txt)"
```
- Shell argument limit: ARG_MAX ≈ 2MB on Linux, but that's not the problem
- Problem: claude must parse the entire argument before starting → delays first output
- Mitigation if you must: split into smaller sessions

---

## Part 3: Tool Control in Automated Scripts

### `--allowedTools` — the key to unattended runs

```bash
# Allow all editing tools — suitable for overnight TDD runs:
claude -p "Execute plan..." \
  --dangerously-skip-permissions

# Precise tool allowlist (principle of least privilege):
claude -p "Review and commit..." \
  --allowedTools "Read,Grep,Glob,Bash(git status *),Bash(git add *),Bash(git commit *)"
```

The `--dangerously-skip-permissions` flag skips ALL permission checks. Appropriate for:
- Overnight scripts where you've reviewed the plan
- CI/CD environments
- Docker-isolated runs

The `--allowedTools` approach is more surgical but requires knowing every tool the plan needs.

### Permission modes

```bash
claude --permission-mode plan     # read-only exploration, no edits
claude --permission-mode default  # interactive approval (not useful for headless)
```

---

## Part 4: Session Continuity and State

### Sessions are persistent and resumable

```bash
# Capture session ID for later:
session_id=$(claude -p "Start review" --output-format json | jq -r '.session_id')

# Resume that specific session:
claude -p "Now fix the issues found" --resume "$session_id"

# Continue most recent session (no ID needed):
claude -p "Continue from where we left off" --continue
```

**Use case for overnight scripts:** If a child plan is long, split it into phases and use
`--continue` between phases to maintain context without restarting from scratch.

### Multi-session hand-off pattern (from AojdevStudio)

The "four-wave" model for complex implementations:
1. **Wave 1**: Fresh session, planning and PRD analysis
2. **Wave 2**: Fresh session, TDD test writing
3. **Wave 3**: Fresh session, implementation to make tests pass
4. **Wave 4**: Fresh session, quality review

Handoff document between waves contains: progress, attempted solutions, remaining tasks.
Each wave runs in a **fresh session** (no context bleed) with a **handoff prompt**.

This maps directly to the overnight-runner pattern:
```
Child 1 = Wave 1 (acquisition flow)
Child 2 = Wave 2 (annotation flow)  — starts from clean main branch
Child 3 = Wave 3 (dashboard/QA)     — starts from clean main branch
```

---

## Part 5: Observability Patterns

### What to monitor in long-running headless sessions

**Stream-json filtering for human-readable output:**
```bash
claude -p "plan" --output-format stream-json --verbose --include-partial-messages \
  | tee full-session.log \
  | jq -rj 'select(.type=="stream_event" and .event.delta.type?=="text_delta") | .event.delta.text'
```

This gives you:
1. Full JSON log saved to `full-session.log` for forensics
2. Human-readable token stream in the terminal in real-time

**Structured output for downstream parsing:**
```bash
result=$(claude -p "analyze code" --output-format json)
success=$(echo "$result" | jq '.is_error')
text=$(echo "$result" | jq -r '.result')
session=$(echo "$result" | jq -r '.session_id')
```

### Notification hooks — get pinged when Claude needs attention

Configure in `.claude/settings.json`:
```json
{
  "hooks": {
    "Notification": [{
      "matcher": "permission_prompt",
      "hooks": [{"type": "command", "command": "notify-send 'Claude Code' 'Needs attention'"}]
    }]
  }
}
```

Matchers: `permission_prompt`, `idle_prompt`, `auth_success`, `elicitation_dialog`

### Context window tracking

Status line via `context-bar.sh` shows: model, directory, git branch, token usage %.
```json
{ "statusLine": "~/.claude/scripts/context-bar.sh" }
```
Configure in `~/.claude/settings.json`. Shows `██░░░░░░░░ 18% of 200k tokens`.

**Critical for overnight scripts:** sessions accumulate context (10-20 GB Node.js heap
over hundreds of tool calls). Use `/compact` when context gets large. Use fresh sessions
(separate `claude -p` invocations) for independent plan chunks.

---

## Part 6: Isolation and Parallelism

### Git worktrees for parallel execution

```bash
# Built-in worktree flag:
claude --worktree feature-auth    # creates .claude/worktrees/feature-auth/
claude -w bugfix-123              # shorthand

# Manual worktrees:
git worktree add ../repo-feature-a -b feature-a
cd ../repo-feature-a && claude
```

Worktrees = full repo copies with isolated branches. Each agent works independently.
No shared filesystem, no git conflicts, independent node_modules, independent Python envs.

**Multi-agent overnight strategy:**
```
Worktree A: child-01 acquisition  ─┐
Worktree B: child-02 annotation   ─┤ run in parallel
Worktree C: child-03 dashboard    ─┘
```
vs. **sequential strategy** (current):
```
main → child-01 → merge to main → child-02 → merge to main → child-03
```

Sequential is safer for plans with dependencies. Parallel is faster but requires
plans that do not depend on each other's code changes.

### Docker isolation

Each overnight run in its own Docker container = zero environment pollution.
```bash
docker run --rm -v "$PWD:/repo" -w /repo minivess-base:latest \
  claude -p "Execute plan..." --dangerously-skip-permissions
```

---

## Part 7: Common Failure Modes and Mitigations

### Failure 1: Silent freeze (zero output for hours)
**Cause:** `text` or `json` output format + claude blocked on something
**Mitigation:** Always use `--output-format stream-json --verbose --include-partial-messages`
**Detection:** watchdog process checks if log size has grown in last N minutes
**Recovery:** `timeout N` hard-kills frozen session; `SKIP_TO=N` resumes from next child

### Failure 2: Context window exhaustion mid-session
**Cause:** Very large plan (19KB XML) + many tool calls accumulate to 200k tokens
**Mitigation:**
- Pass plan FILE PATH, not contents (saves ~18KB of context upfront)
- Split large plans into smaller children
- Use `/compact` explicitly in the prompt: "After each phase, compact context"

### Failure 3: nohup redirect failure
**Cause:** Shell quoting with backslash-newline + nohup interaction
**Mitigation:** NEVER use nohup. Use `screen -S name` or `tmux new -s name`.
**Why screen > nohup:**
- All output streams live to the terminal
- Detach/reattach without losing output
- Can kill/restart sessions easily
- No redirect quoting issues

### Failure 4: Plan file contents passed as argument
**Cause:** `claude -p "$(cat large-plan.xml)"` — 19KB as shell argument
**Mitigation:** `claude -p "Read the plan at: ${PLAN_ABS_PATH} and execute it"`
**Why:** Shell arguments have size limits; more importantly, claude can start
streaming output before reading the file, vs. parsing 19KB before first token

### Failure 5: No timeout on child sessions
**Cause:** Frozen claude runs indefinitely
**Mitigation:** `timeout "${CHILD_TIMEOUT_SEC}" claude -p "..."`
Exit code 124 means timeout — report TIMEOUT not FAILED

### Failure 6: No per-child progress indicators
**Cause:** Child sessions produce no output until exit (with default text format)
**Mitigation:** Heartbeat loop in master script + watchdog on log file size
**Better mitigation:** `--output-format stream-json` makes this unnecessary

---

## Part 8: The Hybrid Script+AI Architecture

The most robust overnight runner pattern separates concerns:

```
┌─────────────────────────────────────────────┐
│  BASH MASTER SCRIPT                         │
│  • Sequencing, retry, timeout               │
│  • Logging, heartbeat, watchdog             │
│  • Git branch management                    │
│  • PR lifecycle (create, wait, merge)       │
│  • State persistence (SKIP_TO, COMPLETED)   │
└──────────────────┬──────────────────────────┘
                   │ spawns N sequential
                   ▼
┌─────────────────────────────────────────────┐
│  CLAUDE HEADLESS SESSION (per child)        │
│  • Reads XML plan file via Read tool        │
│  • Executes TDD cycle (RED → GREEN → VERIFY)│
│  • Makes all code decisions                 │
│  • Commits + pushes branch                  │
│  • Creates PR via gh                        │
└─────────────────────────────────────────────┘
```

Bash does: **what**, **when**, **how long**, **where output goes**
Claude does: **the actual work**

This separation means:
- Bash failure = fix the script (5 min)
- Claude failure = resume from SKIP_TO (1 command)
- Context exhaustion = split the plan (longer but doable)

---

## Part 9: Advanced Flags Reference

```bash
# Complete reference for scripting use cases:

claude -p "prompt"                                    # basic headless
claude -p "prompt" --dangerously-skip-permissions     # no permission prompts
claude -p "prompt" --allowedTools "Read,Edit,Bash"    # explicit tool allowlist
claude -p "prompt" --output-format text               # plain text (buffered)
claude -p "prompt" --output-format json               # JSON envelope (buffered)
claude -p "prompt" --output-format stream-json        # streaming JSON (real-time)
claude -p "prompt" --verbose                          # include tool calls in output
claude -p "prompt" --include-partial-messages         # stream partial tokens
claude -p "prompt" --continue                         # continue last session
claude -p "prompt" --resume "$session_id"             # resume specific session
claude -p "prompt" --append-system-prompt "..."       # add to system prompt
claude -p "prompt" --permission-mode plan             # read-only mode
claude -p "prompt" --json-schema '{...}'              # enforce output schema
claude --worktree feature-branch                      # isolated worktree
claude --from-pr 123                                  # resume PR-linked session

# Environment variables:
MAX_THINKING_TOKENS=8000 claude -p "..."              # limit thinking budget
CLAUDE_CODE_EFFORT_LEVEL=high claude -p "..."         # thinking depth
```

---

## Part 10: Slide-Ready Talking Points

For "Agentic Coding: from Vibes to Production" slides:

### Slide: "Headless Claude is a Unix citizen"
- Accepts stdin, produces stdout, respectable exit codes
- `--output-format stream-json` = UNIX pipe compatible
- Chain with `jq`, `tee`, `timeout`, `watch`
- Works in CI/CD, cron, overnight scripts

### Slide: "The buffering trap"
- Default text output = ZERO bytes until session exits
- 14-hour silent freeze with no error = invisible failure
- Fix: `--output-format stream-json --verbose --include-partial-messages`
- Then: filter with `jq` for human-readable + save full JSON for forensics

### Slide: "Session state is your crash recovery"
- `--output-format json` returns `session_id`
- Store it → `--resume "$session_id"` restores full context
- For multi-night work: `SKIP_TO=2 ./overnight.sh` continues from failed child
- Alternative: Git history IS your state — each committed phase = resumable checkpoint

### Slide: "Prompt passing anti-patterns"
- Never `$(cat 19KB.xml)` as `-p` argument
- Say `"Read the plan at: /path/to/plan.xml"` — let claude read it
- Claude's Read tool = zero argument size issues + output starts immediately

### Slide: "Screen > nohup for overnight work"
- nohup hides output in nohup.out (hard to monitor)
- Redirect failure mode: `> file 2>&1` quoting pitfalls
- screen/tmux: live terminal + survive disconnect + easy kill/restart
- `screen -S overnight && ./script.sh && Ctrl-A D`

### Slide: "Bash orchestrates, Claude implements"
- Bash: sequencing, timeout, logging, git, PR lifecycle
- Claude: code decisions, TDD cycles, implementation
- Never mix concerns: don't put git commands in Claude's prompt if bash can do them
- State handoff via committed artifacts, not Claude context

### Slide: "Observable = debuggable"
- Heartbeat lines every 60s from master script
- Watchdog: 0 bytes after 10 min = STALL WARNING
- `timeout 7200` = guaranteed cleanup of frozen sessions
- Full stream-json log = forensic replay of what claude actually did

---

## References

- [Claude Code docs — headless mode](https://code.claude.com/docs/en/headless)
- [Claude Code docs — common workflows](https://code.claude.com/docs/en/common-workflows)
- [Steve Kinney — Claude Code and bash scripts](https://stevekinney.com/courses/ai-development/claude-code-and-bash-scripts)
- [Paige Niedringhaus — Getting the most out of Claude Code](https://www.paigeniedringhaus.com/blog/getting-the-most-out-of-claude-code/)
- [YK Dojo — 32 Claude Code Tips](https://agenticcoding.substack.com/p/32-claude-code-tips-from-basics-to)
- [AojdevStudio — Hybrid Script + AI TDD Workflow](https://gist.github.com/AojdevStudio/abaab51251b921eb6c000552983c0f48)
- [Amber Jain — Scripting Claude Code for headless tasks](https://amberja.in/scripting-claude-code-for-headless-and-unattended-tasks.html)
- [ykdojo/claude-code-tips — context-bar.sh](https://github.com/ykdojo/claude-code-tips/blob/main/scripts/README.md)
