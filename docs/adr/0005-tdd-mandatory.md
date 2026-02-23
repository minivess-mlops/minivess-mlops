# ADR-0005: Mandatory Test-Driven Development for SaMD

## Status

Accepted

## Date

2026-02-23

## Context

MinIVess MLOps v2 targets SaMD-principled (Software as a Medical Device) development, following IEC 62304 software lifecycle and ISO 13485 quality management system patterns. While the platform is used for preclinical research and is not seeking regulatory submission, adopting these practices from Day 1 serves two purposes:

1. **Portfolio demonstration**: Proving the ability to operate under SaMD discipline to potential employers and collaborators in the regulated medical device and defense sectors.
2. **Future readiness**: If the pipeline is later applied to clinical data or integrated into a regulated product, the test infrastructure and audit trails are already in place.

The v0.1-alpha codebase had 14 test files but no systematic testing discipline. Tests were written after implementation, coverage was uneven, and there was no enforcement of test-first development.

IEC 62304 requires documented evidence that software units are verified before integration. Test-driven development provides this evidence structurally: the failing test is the specification, and the passing test is the verification evidence.

## Decision

All implementation in the v2 codebase must follow a strict TDD workflow, enforced through the `self-learning-iterative-coder` skill (`.claude/skills/self-learning-iterative-coder/SKILL.md`):

1. **RED**: Write a failing test that specifies the desired behavior.
2. **GREEN**: Write the minimum implementation code to make the test pass.
3. **VERIFY**: Run the full test suite, linter (ruff), and type checker (mypy).
4. **FIX**: If any gate fails, apply targeted fixes without expanding scope.
5. **CHECKPOINT**: Commit the passing state with a descriptive message.
6. **CONVERGE**: All gates green -- proceed to the next task.

This workflow is non-negotiable for all code changes: features, bug fixes, and refactors.

Additional testing practices:

- **Hypothesis property-based testing** for configuration validation edge cases.
- **Test markers** (`@pytest.mark.slow`, `@pytest.mark.gpu`, `@pytest.mark.integration`) for selective execution.
- **Deepchecks Vision suites** as validation tests for data integrity and train/test distribution.
- **Three-gate CI**: `pytest` + `ruff check` + `mypy` must all pass before any commit.

## Consequences

**Positive:**

- Every module in the codebase has corresponding test coverage, currently at 102+ tests across 6 test files.
- The three-gate verification (tests + lint + typecheck) catches regressions before they reach the repository.
- Test files serve as living documentation of expected behavior, which is especially valuable for the ModelAdapter protocol and validation layers.
- IEC 62304 verification evidence is produced as a natural byproduct of the development workflow rather than as an after-the-fact documentation exercise.

**Negative:**

- TDD slows initial development velocity, particularly for exploratory research code.
- Property-based testing (Hypothesis) can produce non-obvious failure cases that require careful analysis.
- Strict enforcement requires developer discipline; the tooling detects violations (no implementation without tests) but cannot prevent them entirely.

**Neutral:**

- Tests for optional dependencies (Deepchecks Vision, WeightWatcher, ONNX Runtime) use `pytest.importorskip()` and are skipped when packages are not installed, maintaining a fast default test suite.
- The v0.1-alpha test files in `tests/unit/`, `tests/integration/`, and `tests/data/` are preserved but not executed by the v2 test suite. The v2 tests reside in `tests/v2/`.
