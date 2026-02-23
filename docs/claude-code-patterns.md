# Claude Code Patterns — Real-World Examples from MinIVess MLOps v2

> **Purpose:** Document advanced Claude Code patterns as they are used during this project.
> These examples feed into the Agentic Coding course slides (tier-2 and tier-3).
> Updated incrementally as new patterns are demonstrated.

---

## Pattern 1: Parallel Sub-agents for Independent Tasks (Module 8)

**Context:** Phase 0 foundation setup — 4 independent files need to be created simultaneously.

**What we did:** Launched 4 Task agents in a single message, each with `run_in_background: true`:

```
Agent 1: pyproject.toml (uv + PEP 621)       → reads plan Section 5.1
Agent 2: docker-compose.yml (12 services)     → reads plan Section 5.2 + 16.1 C4
Agent 3: src/minivess/ package skeleton       → reads CLAUDE.md directory structure
Agent 4: .pre-commit-config.yaml + justfile   → reads CLAUDE.md dev tooling
```

**Key insight:** Each agent gets its own 200K context window. The orchestrating session stays lean — it only receives summaries, not the full file contents each agent reads. This is critical for large codebases.

**Real speedup:** 4 agents completed in ~45 seconds total vs. ~3 minutes sequential.

**Code pattern (from actual session):**
```python
# Launch 4 independent agents in ONE tool call block
Task(description="Create new pyproject.toml with uv",
     prompt="Read the plan at docs/modernize-minivess-mlops-plan.md Section 5.1...",
     subagent_type="Bash", run_in_background=True)

Task(description="Create docker-compose.yml with profiles",
     prompt="Read the plan Section 5.2, 16.1 C4, 17.5...",
     subagent_type="Bash", run_in_background=True)

Task(description="Create src/minivess package structure",
     prompt="Read CLAUDE.md for target directory structure...",
     subagent_type="Bash", run_in_background=True)

Task(description="Create pre-commit config and justfile",
     prompt="Read CLAUDE.md for project rules...",
     subagent_type="Bash", run_in_background=True)
```

**Anti-pattern avoided:** Don't launch dependent tasks in parallel. pyproject.toml doesn't depend on docker-compose.yml, so parallel is safe. But "write tests" depends on "create package structure" — those must be sequential.

---

## Pattern 2: Plan-Driven Development with Reviewer Convergence (Module 6/7)

**Context:** Expanding the tech stack required evaluating 30+ tools across 3 dimensions (MLOps fitness, data quality for 3D imaging, eval/XAI appropriateness).

**What we did:**
1. **Plan mode** — Explored the codebase and drafted a comprehensive modernization plan
2. **3 reviewer agents** — Each with a different expert persona evaluated the plan independently
3. **Convergence** — Identified agreements, disagreements, and resolutions
4. **Implementation** — Only after reviewer consensus did we write code

**Key disagreement resolved:** Great Expectations vs Pandera
- MLOps reviewer: "Use GE, replace Pandera"
- Data quality reviewer: "GE can't validate 3D NIfTI volumes — keep Pandera"
- Resolution: Use BOTH — GE for tabular metadata, Pandera for dataset schemas, custom for 3D data

**Real CLAUDE.md excerpt demonstrating the output:**
```markdown
| **Data Validation** | Pydantic v2 (schema) + Pandera (DataFrame) + Great Expectations (batch quality) |
```

---

## Pattern 3: Phase Tracker for Context Management (Module 4)

**Context:** Implementing a 6-phase project that will span many sessions and context window compressions.

**What we did:** Created `.claude/phase-tracker.md` — a persistent file that tracks:
- Current phase and task status
- What's completed vs. pending
- Dependencies between tasks

**Why this matters:** When context compresses (auto or manual `/clear`), the agent re-reads the tracker file and continues from where it left off. The plan document is on disk, the tracker state is on disk — the agent's "memory" survives context window limits.

```markdown
# Phase Execution Tracker
## Current Phase: 0 — Foundation
### Phase 0 Tasks
- [x] P0.1: Initialize project with `uv init`, set up pyproject.toml
- [x] P0.2: Create docker-compose.yml with profiles
- [x] P0.3: Create src/minivess package skeleton
- [x] P0.4: Pre-commit + justfile
- [ ] P0.5: Pydantic v2 config models
- [ ] P0.6: Hydra-zen + Dynaconf configs
...
```

---

## Pattern 4: CLAUDE.md as a Living Contract (Module 3/4)

**Context:** MinIVess v2 has strict rules (uv only, TDD mandatory, future annotations everywhere) that must be enforced across all sessions.

**What we did:** The CLAUDE.md file serves as:
1. **Quick Reference** — Tool stack at a glance (13 rows)
2. **Critical Rules** — 7 non-negotiable constraints
3. **Workflow Definition** — TDD phases with linked protocol files
4. **Negative Constraints** — "What AI Must NEVER Do" section

**Key design choice:** The Quick Reference table grew from 4 to 13 rows as tools were added. This isn't bloat — each row is a signal to the agent about which tool to reach for in a given situation.

---

## Pattern 5: Self-Learning Iterative Coder Skill (Module 6/7)

**Context:** TDD mandate requires a specific workflow that's easy for agents to skip steps in.

**What we did:** Created a custom skill at `.claude/skills/self-learning-iterative-coder/SKILL.md` with:
- Protocol files for each phase (red, green, verify, fix, checkpoint, convergence)
- Activation checklist that must run before multi-task implementations
- State tracking across the RED→GREEN→VERIFY cycle

**This is a skill that constrains the agent's behavior** — it cannot just write implementation code directly. It must first write a failing test, then implement, then verify.

---

## Pattern 6: Documentation-as-You-Go for Pattern Extraction

**Context:** This project serves dual purpose — MLOps implementation AND Claude Code course material.

**What we did:** Created THIS document (`docs/claude-code-patterns.md`) to capture patterns in real-time as they're demonstrated. After each major milestone, patterns are extracted into slide content.

**The meta-pattern:** The best way to document advanced techniques is to use them on a real project and write down what happened — not to fabricate examples.

---

*Last updated: 2026-02-23 — Phase 0 Batch 1 (parallel sub-agents)*
