# Activation Checklist: create-literature-report

Run ALL checks before starting Phase 0. If any BLOCKING check fails, STOP.

## 1. Topic Validation [BLOCKING]

- [ ] Topic is specific enough to generate focused search queries
- [ ] Topic is broad enough to find ≥ target_paper_count papers
- [ ] Topic is stated as a research question or domain, not a single paper

## 2. Seed Papers [BLOCKING]

- [ ] At least 5 seed papers provided (minimum for cross-domain synthesis)
- [ ] Each seed has: author, year, title, URL
- [ ] Seed URLs are reachable (spot-check 3 random seeds via WebFetch)

## 3. Output Path [BLOCKING]

- [ ] `output_path` directory exists and is writable
- [ ] No existing file at `output_path` (or user confirms overwrite)

## 4. State File Check

- [ ] Check `state/literature-report-{topic_slug}-state.json`
  - If exists: RESUME from last checkpoint (load state, skip completed phases)
  - If not exists: FRESH START (Phase 0)

## 5. Tools Available [BLOCKING]

- [ ] WebFetch tool available (required for Phase 3 verification)
- [ ] WebSearch tool available (required for Phase 1 gather)
- [ ] Agent tool available (required for Phase 1 and Phase 3 parallelism)
- [ ] `gh` CLI authenticated (required for Phase 5 issue creation, skip if `create_issues=false`)

## 6. Quality Target Confirmed

- [ ] User has confirmed quality_target (default: MINOR_REVISION)
- [ ] User has confirmed whether Phase 4 (REVIEW) should run

## 7. Execution Mode

- [ ] Interactive: Ask alignment questions in Phase 0
- [ ] Autonomous: Use provided parameters, skip alignment questions

## Red Flags (STOP immediately)

- "I remember this paper" → You are about to hallucinate. Use WebSearch instead.
- Seed paper URL returns 404 → Do not proceed with broken seeds.
- Topic is "everything about X" → Too broad. Narrow to a specific question.
- No seed papers at all → Refuse. The skill requires seeds as starting points.
