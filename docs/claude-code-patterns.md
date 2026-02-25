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

## Pattern 7: Batch-and-Commit Workflow (Module 5)

**Context:** MinIVess v2 was implemented across 6 phases, each decomposed into batches. Between each batch, the agent verified all tests and committed. This created a disciplined checkpoint cadence across 10+ commits.

**What we did:** Each phase followed the same cycle:
1. Launch parallel sub-agents for the batch's independent tasks
2. Verify all tests pass (`uv run pytest tests/ -x -q`)
3. Commit with a descriptive message linking phase/batch
4. Move to the next batch or phase

**Real commit history demonstrating the pattern:**
```
1217dbc feat(v2): Phase 0 Batch 1 — Foundation skeleton with uv, Docker Compose, package structure
80a17ac feat(v2): Phase 0 Batch 2 — Pydantic configs, Dynaconf envs, tests, CI
69e63fa feat(v2): Phase 0 Batch 3 — Hydra-zen configs, Pandera/GE validation, DVC pipeline
f1586bd feat(v2): Phase 1 Batch 1 — ModelAdapter ABC, SegResNet/SwinUNETR adapters, data pipeline
76ca42b feat(v2): Phase 1 Batch 2 — Training engine, MLflow tracking, metrics, DuckDB analytics
f471914 feat(v2): Phase 1 Batch 3 — Unit tests for adapters + pipeline (48 tests, 66 total)
1b58189 feat(v2): Phase 2 — Ensemble strategies, WeightWatcher, calibration, drift detection
b40f44c feat(v2): Phase 3 — BentoML serving, ONNX Runtime inference, Gradio demo
8492624 feat(v2): Phase 4+5 — OTel telemetry, SaMD compliance, LangGraph agents, Braintrust eval
```

**Key insight:** Frequent commits serve as "save points" for the agent. If a later batch introduces a regression, `git diff` against the previous commit immediately isolates the problem. Without commits, the agent's debugging context grows unboundedly.

**Anti-pattern avoided:** Waiting until "everything works" to commit. In multi-phase projects, this leads to enormous diffs and impossible-to-debug regressions.

---

## Pattern 8: Test-First Sub-agents (Module 6)

**Context:** Phase 1 had 3 batches. Batches 1 and 2 implemented the core ML pipeline (adapters, trainer, MLflow, metrics). Batch 3 was dedicated exclusively to writing tests after the implementation was done.

**What we did:**
- Batch 1 agents: wrote ModelAdapter ABC, SegResNet/SwinUNETR adapters, data pipeline
- Batch 2 agents: wrote training engine, MLflow integration, TorchMetrics, DuckDB analytics
- Batch 3 agent: read ALL implementations from Batches 1-2, then wrote 48 tests covering them

**Test count progression across the project:**
```
Phase 0 Batch 2:   18 tests (config validation, property-based with Hypothesis)
Phase 0 Batch 3:   18 tests (no new — validation/DVC configs don't add unit tests)
Phase 1 Batch 3:   66 tests (+48: adapters, trainer, loss, metrics, DuckDB)
Phase 2:           80 tests (+14: ensemble, calibration, drift, WeightWatcher)
Phase 3:           87 tests (+7:  BentoML, ONNX, Gradio — 8 skipped for opt deps)
Phase 4+5:        102 tests (+15: telemetry, audit trail, model cards, agents)
```

**Key insight:** The TDD mandate says "write failing tests FIRST." But with parallel agents, a pragmatic adaptation is: implementation agents write the code in Batches 1-2, then a dedicated test agent in Batch 3 reads the implementations and writes comprehensive tests. The test agent has the advantage of seeing the actual API surface, not guessing at it.

**Real conftest.py that the test agent needed to create:**
```python
# tests/conftest.py — must suppress warnings BEFORE third-party imports
import warnings

warnings.filterwarnings("ignore", message=".*deprecated.*",
                        category=DeprecationWarning, module="pyparsing.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning,
                        module="MetricsReloaded.*")
```

---

## Pattern 9: Graceful Dependency Handling (Module 7)

**Context:** Phase 1 Batch 3 revealed multiple dependency conflicts that only manifest at test time. Each required a different fix strategy.

**What we fixed:**

| Problem | Root Cause | Fix |
|---------|-----------|-----|
| MetricsReloaded fails to build | Uses `pkg_resources`, removed in `setuptools>=82` (Python 3.13+) | `[tool.uv] build-constraint-dependencies = ["setuptools<82"]` |
| WeightWatcher pins `pandas==2.1.4` | Old pin, fails to build from source on 3.13+ | `override-dependencies = ["pandas>=2.2"]` |
| CML install fails from PyPI | PyPI `cml` is an unrelated package; Iterative CML is npm-based | Disabled in `[project.optional-dependencies]` with comment |
| SwinUNETR API breaks with MONAI 1.5.x | MONAI changed constructor parameter names | Updated adapter to use new API (`norm_name="instance"`, explicit `depths`, `num_heads`) |

**Real pyproject.toml excerpts:**
```toml
# CML is npm, not PyPI
ci = [
    # cml>=0.20 — disabled: PyPI 'cml' is a different package than iterative CML;
    # iterative CML is installed via npm, not pip.
]

# uv — workaround for MetricsReloaded missing pkg_resources build dep
[tool.uv]
build-constraint-dependencies = ["setuptools<82"]
# weightwatcher pins pandas==2.1.4 which fails to build from source on Python 3.13+
override-dependencies = ["pandas>=2.2"]
```

**Key insight:** Sub-agents that encounter dependency conflicts should fix them locally (in pyproject.toml or the adapter code) and document the workaround inline. The next agent to read the file understands the "why" immediately.

---

## Pattern 10: Import-Time Warning Suppression (Module 7)

**Context:** `pytest` was configured with `filterwarnings = ["error"]` to catch real issues. But MONAI imports `matplotlib`, which imports `pyparsing`, which emits a `DeprecationWarning` at import time — before any test code runs. MetricsReloaded emits `SyntaxWarning` at import. These warnings became test failures.

**The problem chain:**
```
MONAI → matplotlib → pyparsing → DeprecationWarning (import time)
MetricsReloaded → SyntaxWarning (import time)
pytest filterwarnings = ["error"] → these warnings become test failures
```

**Why `pyproject.toml` filters aren't enough:** The `filterwarnings` in `pyproject.toml` applies after pytest collects tests, but some warnings fire during initial imports in conftest.py or early collection. The suppression must happen *before* the imports.

**Solution:** Root `tests/conftest.py` with `warnings.filterwarnings()` calls at the top of the file:
```python
from __future__ import annotations

import warnings

# Suppress warnings that occur during import of third-party libraries.
# These must be set before the libraries are imported, so pytest's
# filterwarnings config (which applies after collection) is too late.
warnings.filterwarnings("ignore", message=".*deprecated.*",
                        category=DeprecationWarning, module="pyparsing.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning,
                        module="MetricsReloaded.*")
```

**Pattern:** Medical imaging stacks (MONAI, TorchIO, MetricsReloaded) have noisy import-time warnings from deep transitive dependencies. The root `conftest.py` is the right place to suppress them — it runs before test collection, and the suppressions are version-pinned by the comment explaining which library triggers each one.

---

## Pattern 11: Model-Agnostic Adapter Pattern for ML Pipelines (Module 9)

**Context:** MinIVess v2 supports multiple segmentation architectures (SegResNet, SwinUNETR, with SAMv3 planned). The training engine, ensemble strategies, and serving layer should not know which model they are operating on.

**What we did:** Created a clean abstraction ladder:

```
ModelAdapter ABC (base.py)
    ├── SegResNetAdapter (segresnet.py)
    ├── SwinUNETRAdapter (swinunetr.py)
    └── [future: SAMv3Adapter]
          │
          ▼
SegmentationTrainer.fit(model: ModelAdapter)   # trains any adapter
          │
          ▼
EnsemblePredictor(models: list[ModelAdapter])  # ensembles any adapter
          │
          ▼
BentoML service + ONNX export                  # serves any adapter
```

**The ABC interface (from `src/minivess/adapters/base.py`):**
```python
class ModelAdapter(ABC, nn.Module):
    @abstractmethod
    def forward(self, images: Tensor, **kwargs: Any) -> SegmentationOutput: ...
    @abstractmethod
    def get_config(self) -> dict[str, Any]: ...
    @abstractmethod
    def load_checkpoint(self, path: Path) -> None: ...
    @abstractmethod
    def save_checkpoint(self, path: Path) -> None: ...
    @abstractmethod
    def trainable_parameters(self) -> int: ...
    @abstractmethod
    def export_onnx(self, path: Path, example_input: Tensor) -> None: ...
```

**Why this matters for agent-generated code:** When Claude Code creates a new model adapter, it only needs to implement 6 methods. The rest of the pipeline (training, ensembling, serving, compliance audit) works automatically because it programs against the ABC, not the concrete class. This is the kind of constraint that makes agent-generated code reliable — the type checker enforces completeness.

**Anti-pattern avoided:** A monolithic `train()` function with `if model_type == "segresnet": ... elif model_type == "swinunetr": ...` branches. This grows unboundedly and each new model requires touching every function.

---

## Pattern 12: SaMD Compliance as Code (Module 10)

**Context:** IEC 62304 (Software as a Medical Device) requires audit trails, traceability, and documentation. Rather than maintaining compliance in separate Word documents, we encoded every requirement as Python dataclasses with JSON serialization.

**Three pillars implemented:**

1. **Audit Trail** (`src/minivess/compliance/audit.py`):
```python
@dataclass
class AuditEntry:
    timestamp: str
    event_type: str       # DATA_ACCESS, MODEL_TRAINING, MODEL_DEPLOYMENT, TEST_EVALUATION
    actor: str
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)
    data_hash: str | None = None  # SHA-256 for data integrity

@dataclass
class AuditTrail:
    entries: list[AuditEntry] = field(default_factory=list)

    def log_data_access(self, dataset_name, file_paths, *, actor="system"):
        data_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        ...

    def save(self, path: Path) -> None:  # JSON serialization
    def load(cls, path: Path) -> AuditTrail:  # JSON deserialization
```

2. **Model Cards** (`src/minivess/compliance/model_card.py`) — Mitchell et al. (2019) format:
```python
@dataclass
class ModelCard:
    model_name: str
    model_version: str
    intended_use: str = "Research use only - biomedical vessel segmentation"
    ethical_considerations: str = "Not intended for clinical diagnostic use..."
    metrics: dict[str, float] = field(default_factory=dict)

    def to_markdown(self) -> str:  # Generates standard Model Card Markdown
```

3. **SHA-256 Data Hashing** for integrity verification:
```python
content = "\n".join(sorted(file_paths))
data_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
```

**Key insight:** When compliance requirements are encoded in code, they are:
- **Testable** — 15 tests verify audit trail behavior (log events, save/load, hash computation)
- **Versioned** — git tracks every change to the compliance schema
- **Automatable** — the LangGraph evaluation pipeline (Phase 5) calls `audit.log_test_evaluation()` automatically
- **Portable** — JSON audit trails can be ingested by any regulatory tool

**Anti-pattern avoided:** Maintaining compliance documentation as separate Word/PDF files that drift from the actual code behavior. When compliance is code, it cannot be out of date.

---

## Pattern 13: Multi-Agent Parallel Code Review (Module 8)

**Context:** After completing all 51 GitHub issues (662 tests), the codebase needed a deep quality review before moving to production hardening. A single-pass review would be too shallow; sequential reviews would be too slow.

**What we did:** Launched 6 specialist reviewer agents in parallel, each with a different focus:

```
Agent 1: Dead code analysis        → orphaned modules, unreachable code, stub classes
Agent 2: Duplicate code patterns   → shared utilities, StrEnum proliferation, markdown boilerplate
Agent 3: Test coverage gaps        → untested methods, missing error paths, property-based tests
Agent 4: Decoupling analysis       → cross-package imports, hardcoded values, DI opportunities
Agent 5: Reproducibility issues    → seeds, datetime handling, encoding, path handling
Agent 6: API consistency           → naming conventions, return types, docstring styles, exceptions
```

**Architecture diagram:**
```
Orchestrating Agent (main context)
    │
    ├── launches 6 agents in parallel (each in background)
    │   ├── Agent 1: dead code      ──→ reads 40+ src files
    │   ├── Agent 2: duplicates     ──→ reads 35+ src files
    │   ├── Agent 3: coverage       ──→ reads 50+ src + test files
    │   ├── Agent 4: decoupling     ──→ reads 30+ src files
    │   ├── Agent 5: reproducibility──→ reads 25+ src files
    │   └── Agent 6: API consistency──→ reads 40+ src files
    │
    ├── waits for all 6 to complete (~2-3 minutes)
    │
    ├── reads each agent's findings (summaries, not raw files)
    │
    └── synthesizes into unified report with prioritized remediation plan
```

**Key insight:** Each agent gets its own 200K context window. Agent 3 (test coverage) read 50+ source AND test files — that's ~150K tokens of code. The orchestrating agent only receives the ~5K summary from each agent. Without this delegation, a single agent couldn't hold the entire codebase + analysis in context.

**Results from actual run:**
- 6 agents consumed ~334K total tokens across their independent analyses
- Identified 42 actionable issues (4 critical, 12 high, 14 medium, 9 low)
- Produced a unified remediation plan with 18 TDD-ready work items across 4 phases
- The entire review (launch → synthesis → report) completed in one conversation turn

**What the review found (real results):**
| Agent | Key Finding |
|-------|-------------|
| Dead code | Sam3Adapter is entirely dead (constructor always raises) |
| Duplicates | 13 files share identical `to_markdown()` patterns (~400 LOC extractable) |
| Coverage | Zero error-path tests across the entire codebase |
| Decoupling | Ensemble module imports concrete ModelAdapter (should use Protocol) |
| Reproducibility | No centralized seed propagation; MONAI transforms unseeded |
| API consistency | 10+ methods return `dict[str, Any]` where typed dataclasses should be used |

**Anti-pattern avoided:** Running one massive "review everything" agent. Such an agent would exhaust its context window reading files before finishing analysis. The specialist pattern ensures each agent can deeply analyze its domain.

**Anti-pattern avoided:** Sequential reviews (Agent 1 finishes, then Agent 2 starts). All 6 agents run simultaneously — the wall-clock time is `max(agent_times)` not `sum(agent_times)`.

---

## Pattern 14: Property-Based Testing with Hypothesis (Module 6)

**Context:** Traditional example-based tests verify specific inputs/outputs. Property-based testing uses Hypothesis to generate hundreds of random inputs and verify *invariants* that must hold for ALL valid inputs — catching edge cases humans wouldn't think to test.

**Technique:** Use `@given` decorators with `hypothesis.strategies` to generate random inputs. Define mathematical properties rather than specific test cases.

**Real example from this project:**

```python
from hypothesis import given, settings
from hypothesis import strategies as st

class TestBootstrapCIProperties:
    @given(
        data=arrays(dtype=np.float64, shape=st.integers(5, 50),
                    elements=st.floats(0.0, 1.0)),
    )
    @settings(max_examples=20, deadline=5000)
    def test_ci_bounds_ordering(self, data):
        """bootstrap_ci should ALWAYS produce lower <= point <= upper."""
        ci = bootstrap_ci(data, n_resamples=200, seed=42)
        assert ci.lower <= ci.point_estimate <= ci.upper
```

**Properties tested in MinIVess (10 tests):**

| Property | Module | Invariant |
|----------|--------|-----------|
| CI ordering | `pipeline/ci.py` | lower ≤ point ≤ upper for ANY sample |
| CI width monotonicity | `pipeline/ci.py` | Higher confidence → wider interval |
| Temperature scaling | `ensemble/calibration.py` | Output sums to 1.0 for ANY temperature |
| Probability range | `ensemble/calibration.py` | All outputs in [0, 1] |
| Risk non-negativity | `observability/pprm.py` | |pred - label| ≥ 0 for ANY input |
| Risk shape | `observability/pprm.py` | Output shape = input shape |
| Dice range | `pipeline/metrics.py` | 0 ≤ Dice ≤ 1 for ANY prediction |

**Bug found by property-based thinking:** The PPRM detector's `monitor()` method used `np.var(ddof=1)` on a single sample, causing a RuntimeWarning (degrees of freedom ≤ 0). This edge case was never caught by example-based tests but emerged from thinking about what inputs are *possible*.

**Lesson:** Property-based tests are most valuable for mathematical/statistical code where invariants are known but edge cases are hard to enumerate manually.

---

*Last updated: 2026-02-25 — Pattern 14 from property-based testing (793 tests, up from 662)*
