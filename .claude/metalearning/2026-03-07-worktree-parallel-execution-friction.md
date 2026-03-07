# Metalearning: Git Worktree Parallel Execution Friction

## Classification: Tooling Limitation Analysis (NOT a fuckup — a design exploration)

## What Happened

Attempted to run the TDD skill (self-learning-iterative-coder) inside a manually-created
git worktree at `.claude/worktrees/biostats-double-check/`. Multiple problems:

1. **Background tasks don't capture output** — `Bash` tool commands piped to files or
   running in background produce empty output when CWD is a worktree subdirectory.
   The background task system writes to `/tmp/claude-1000/-home-petteri-Dropbox-...`
   which resolves based on the main repo path, not the worktree path.

2. **VSCode interference** — The user has VSCode open on the main repo. Any `git checkout`
   in the main repo affects the VSCode UI. The worktree avoids this but introduces the
   execution problems above.

3. **Memory file traversal** — Claude Code loads CLAUDE.md from both the worktree AND the
   parent repo, causing security warnings and duplicate context.

4. **Hook failures** — Stop hooks and other lifecycle hooks may fail in worktree contexts
   because transcript paths don't resolve correctly.

5. **`uv` virtual environment** — The `.venv` in the worktree may not be properly linked,
   causing dependency resolution issues (the `hypothesis` import failure was likely this).

## The Fundamental Problem

Claude Code was designed for **single-directory, single-branch operation**. The worktree
pattern forces it into an alien context where paths, hooks, background tasks, and memory
traversal all have edge cases. The friction is not one big problem but a death-by-a-thousand-cuts
accumulation of small issues that waste tokens diagnosing infrastructure instead of doing work.

## Alternative Approaches Explored

### Option 1: Claude Code Agent Teams (Built-in, Experimental)

**How it works:** A lead Claude session spawns teammate Claude instances, each in its own
git worktree. Teammates communicate via shared task list and messaging. Lead coordinates.
Enable with `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`.

**Pros:**
- Native Anthropic feature — will improve over time
- Built-in worktree isolation per teammate
- Shared task list with dependency tracking (tasks block/unblock automatically)
- Teammates can challenge each other's approaches (adversarial debugging)
- Split-pane display (tmux/iTerm2) for visibility
- CLAUDE.md loaded per worktree automatically
- Hooks (`TeammateIdle`, `TaskCompleted`) for quality gates

**Cons:**
- **Experimental** — known limitations: no session resumption, task status lag, slow shutdown
- 4-15x more tokens than single session (each teammate is a full Claude instance)
- One team per session, no nested teams
- Lead can't be transferred
- Split panes require tmux or iTerm2 (not VSCode terminal)
- Best for tasks with clear parallel boundaries — less useful for sequential TDD phases
- No ability to set per-teammate permissions at spawn

**Fit for MinIVess biostatistics plan:**
MODERATE. The 20-task plan is mostly sequential (Phase 0 -> 2 -> 3 -> ...). Within phases,
some parallelism exists (e.g., pairwise + variance in Phase 3). Agent teams shine for
truly parallel work (3 reviewers on a PR, 5 competing debugging hypotheses). For a
sequential TDD loop, a single session is more cost-effective.

HOWEVER: Could use agent teams for the reviewer phase (3 agents reviewing statistical
methodology, Docker compliance, and testing strategy simultaneously). This matches the
"research and review" use case perfectly.

### Option 2: Conductor (Mac App, Worktree-Based)

**How it works:** Mac desktop app that manages multiple Claude Code sessions, each in its
own git worktree. Dashboard shows all agents and their status. Integrated diff viewer
and merge tooling.

**Pros:**
- Dedicated UI for managing parallel Claude Code sessions
- Each session = one git worktree (proven isolation model)
- Dashboard: see who's working on what at a glance
- Integrated code review and merge workflow
- Supports slash commands and MCP integration
- "Make running a small swarm as easy as running one"

**Cons:**
- Mac only (no Linux support — problematic for this Ubuntu dev machine)
- External tool dependency (not built into Claude Code)
- Still uses worktrees under the hood (same git limitations)
- No container isolation (worktrees share filesystem, node_modules, .venv, etc.)
- Pricing unknown (may become paid)
- Less mature than Claude Code's built-in agent teams

**Fit for MinIVess:** LOW. Mac-only is a hard blocker (this project runs on Ubuntu).

### Option 3: Sculptor by Imbue (Docker Container-Based)

**How it works:** Desktop app that runs each Claude Code session in its own Docker container.
"Pairing Mode" syncs container changes back to local repo for IDE integration and testing.
Free during beta.

**Pros:**
- **True container isolation** — each agent in its own Docker, not just worktree
- Pairing Mode: sync container work to local repo for instant preview/testing
- No git worktree edge cases (containers are fully isolated filesystems)
- Multiple instances can run without port conflicts or shared resource issues
- Can run multiple NextJS apps in parallel (containers have independent network stacks)
- Safety: agents can't destroy host filesystem
- More isolation than worktrees — different .venv, different node_modules, different everything

**Cons:**
- Mac (Apple Silicon) and Linux only — Linux support exists (good for this project)
- Requires Docker (overhead of container creation per session)
- Container sync latency (Pairing Mode adds round-trip time)
- Free in beta but pricing TBD
- Imbue is a startup — longevity risk
- Learning curve for container-based workflow
- May conflict with the project's own Docker-per-flow architecture (Docker-in-Docker?)

**Fit for MinIVess:** HIGH for isolation quality, but Docker-in-Docker complexity is a
concern. The biostatistics flow already requires Docker containers. Running Sculptor's
Docker container that runs our Docker container is an extra layer. However, for the
pure-Python statistical code (Phases 2-7), Sculptor containers would work well.

### Option 4: Manual Worktrees (What We Tried)

**How it works:** `git worktree add` creates a separate directory, Claude Code runs in it
via explicit path passing. All tools reference the worktree path.

**Pros:**
- Zero external dependencies
- Works on any platform
- Full control over branch state
- No token overhead (single session)

**Cons:**
- Background task output doesn't resolve correctly
- Memory file traversal warnings
- Hook failures in worktree contexts
- VSCode still needs separate window for worktree
- Manual — no coordination UI, no dashboard
- The death-by-a-thousand-cuts friction documented above

**Fit for MinIVess:** LOW for multi-branch parallel work. OK for simple "keep main clean
while working on a feature" — but the execution problems make it unreliable for automated
TDD loops.

### Option 5: Simple Sequential (No Parallelism)

**How it works:** Work on one branch at a time. Commit, push, switch branches when needed.
Accept that branch switching loses Claude context.

**Pros:**
- Zero complexity
- Zero extra tooling
- All Claude Code features work as designed
- Most reliable approach

**Cons:**
- Can't work on multiple branches simultaneously
- Context loss on branch switch (re-reading CLAUDE.md, re-understanding codebase)
- Blocking: can't start biostatistics work while enforcement PR is in review

**Fit for MinIVess:** ADEQUATE for most work. The "two branches at once" need is actually
rare. Most of the time, sequential is fine. The frustration with branch switching is real
but finite (20 min context reload per switch, which is the cost of the screenshot the user showed).

## Recommendation

### Short-term (this session): Simple Sequential
Just work on `chore/biostats-double-check` in the main repo. The enforcement PR (#461) is
already merged to main. There's no active parallel branch. The worktree was solving a
problem that no longer exists.

### Medium-term (this week): Enable Agent Teams for Review Phases
Set `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` and use agent teams for the review/research
phases (e.g., "spawn 3 reviewers to validate statistical methodology, Docker compliance,
and test coverage in parallel"). This matches agent teams' sweet spot (parallel exploration)
without fighting its weak spot (sequential implementation).

### Long-term (evaluate): Sculptor for Container Isolation
Sculptor's container model is philosophically aligned with MinIVess's Docker-per-flow
architecture. Worth evaluating once the biostatistics flow is implemented and we have a
concrete Docker container to test with. The Linux support is promising.

### Avoid: Manual Worktrees for TDD Loops
The friction is too high for automated loops. Worktrees are fine for quick "check something
on another branch" tasks, but not for sustained development.

## Key Insight

The real question isn't "how do I run two branches at once?" but "do I actually need to?"
In this case, the enforcement PR was already merged. The biostatistics work has no parallel
dependency. The perceived need for parallel branches was an artifact of the conversation
flow, not a real workflow requirement.

When genuine parallelism IS needed (e.g., 3 reviewers validating a statistical plan),
Claude Code's built-in agent teams are the right tool. For everything else, sequential
is fine.

## Sources

- [Claude Code Agent Teams Docs](https://code.claude.com/docs/en/agent-teams)
- [Conductor Docs](https://docs.conductor.build/)
- [Sculptor by Imbue](https://imbue.com/sculptor/)
- [ccswarm: Multi-agent orchestration with Git worktree isolation](https://github.com/nwiizo/ccswarm)
- [Claude Code Common Workflows (Worktrees)](https://code.claude.com/docs/en/common-workflows)
- [Claude Code memory traversal in worktrees (Issue #16600)](https://github.com/anthropics/claude-code/issues/16600)
- [Stop hook fails in worktrees](https://github.com/thedotmack/claude-mem/issues/1276)
- [Claude --worktree bug (Issue #27044)](https://github.com/anthropics/claude-code/issues/27044)
