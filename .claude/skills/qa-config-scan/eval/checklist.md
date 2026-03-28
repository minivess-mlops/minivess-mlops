# QA Config Scan Evaluation Checklist

6 binary YES/NO criteria for evaluating a scan run. All 6 must be YES for
a passing scan. 3 structural criteria verify the scan mechanics work
correctly. 3 behavioral criteria verify the scan produces useful results.

## Structural Criteria

### S1: All scan directories exist and contain Python files?

- [ ] YES / NO

Every directory in the scan target list (`src/minivess/orchestration/flows/`,
`src/minivess/pipeline/`) must exist and contain at least one `.py` file.
A scan that silently processes zero files is worse than a scan that fails.

**Verification**: `test_scan_directories_exist` and `test_scanned_files_are_nonempty`
in `test_no_hardcoded_params_in_flows.py`.

### S2: AST parsing succeeds on all scanned files?

- [ ] YES / NO

Every Python file in the scan scope must parse without `SyntaxError`.
Files that fail to parse are silently skipped, which hides potential
violations. Report any parse failures as scan infrastructure issues.

**Verification**: Count files that raise `SyntaxError` during `ast.parse()`.
Must be zero.

### S3: Phase 1 COLLECT produces structured output?

- [ ] YES / NO

The deterministic collection phase must produce a well-formed result dict
with all expected keys: `hardcoded_params`, `version_pin_drift`,
`cross_file_inconsistency`, `boundary_violations`, `files_scanned`,
`timestamp`. Missing keys indicate a broken collection pipeline.

**Verification**: Check that `collect_result` dict has all 6 required keys
and `files_scanned > 0`.

## Behavioral Criteria

### B1: Zero false positives in CRITICAL findings?

- [ ] YES / NO

Every CRITICAL-severity finding must be a genuine violation of CLAUDE.md
rules (Rule 22, 29, or 31). False positives at CRITICAL level erode trust
and cause developers to ignore the scanner.

**Verification**: Manually review each CRITICAL finding. Config definition
files (`src/minivess/config/`), test files, display/layout values, and
infrastructure constants (ports, timeouts) must NOT appear as CRITICAL.

### B2: Known violations are detected?

- [ ] YES / NO

The scan must detect at least the violation classes that have existing
guard tests:
- Hardcoded alpha=0.05 (if present) -- detected by `test_no_hardcoded_alpha.py`
- Version pin drift (if present) -- detected by `test_version_pin_consistency.py`
- env fallback values (if present) -- detected by `test_env_single_source.py`

A scan that misses violations caught by existing tests is less useful than
the tests alone.

**Verification**: Compare scan findings against existing guard test results.
The scan should be a superset.

### B3: Suggested guard tests are implementable?

- [ ] YES / NO

Every suggested new pytest test in the Phase 4 REPORT must be:
1. Specific enough to implement (exact file path, test class name, check description)
2. Non-redundant with existing guard tests
3. Deterministic (no LLM judgment required at test runtime)
4. Fast enough for the staging tier (< 5 seconds per test)

**Verification**: Review each suggested test. Could a developer implement it
from the description alone without further clarification?
