# progress.md — chronological log

Chronological log of completed outcomes, commands/tests run, experiment IDs, and the next
milestone. Newest entries last.

## 2026-08-17

- Environment inventory: macOS 14 (Intel i7-4770HQ, 16 GB RAM), git 2.53.0, gh 2.89.0
  authenticated as `aayushkrm` (repo scope), network OK (GitHub/HF/PyPI reachable),
  uv 0.x available; Python 3.12.13 venv at `.venv`.
- Cloned `aayushkrm/llm-stembench` (MIT license, single initial commit) to
  `/Users/akm/Documents/LLM-Bench/llm-stembench`.
- Literature agent verified and wrote 33-source annotated bibliography, literature review
  and BibTeX to `docs/literature/` (R0.1 complete).
- Provider verification (zero-spend):
  - OpenRouter key valid (`is_free_tier: true`, usage 0). 20 free models. Documented
    free-tier limits: 20 req/min, 50 req/day for accounts with < $10 credits.
    Smoke call to `openai/gpt-oss-20b:free` succeeded (cost 0).
  - Opencode Zen key valid; 7 free models; 4 of 6 probed responded immediately
    (`hy3-free`, `nemotron-3-ultra-free`, `nemotron-3.5-lightning-free`,
    `laguna-s-2.1-free`); `big-pickle`, `mimo-v2.5-free`, `deepseek-v4-flash-free`
    returned transient upstream `FreeUsageLimitError` (per-model saturation, not account).
  - Keys stored in gitignored `.env`; `.gitignore` created.
- Python deps installed into `.venv` (numpy 2.5.2, scipy 1.18.0, pandas, matplotlib,
  pydantic, httpx, datasets 5.0.1, pytest, ruff).
- Tracking files created (goal.md, progress.md, decisions.md, risks.md, audit.md).
- `pyproject.toml` written; package `stembench` with CLI entry point.

### Next milestone
Implement core package (schemas, providers, runner, parsing, scoring) with tests, then the
Stage 1 vertical slice.

## 2026-08-17 (continued)

- Core package implemented and committed (8c-parts): schemas, bilingual prompts,
  providers (openrouter/zen/ollama/fake, free-model allowlists, shared budget+rate
  state), Cyrillic-robust parsing, MC/exact/numeric scoring, full metrics suite
  (classification, calibration, intervals, significance, agreement), runner with
  checkpoint/resume + budget stop, report + viz modules, Stage 2 paired analysis module.
- MMLU-STEM loaded (cais/mmlu, revision c30699e8356d; 3,445 items, 21 subjects).
- Preregistration BEFORE results: docs/dataset_selection_memo.md, docs/hypotheses.md,
  docs/error_taxonomy.md; benchmark spec; bilingual annotation package.
- **Run S1-P1 (real, free tier)**: 215 valid evaluations. nemotron-3.5-lightning-free
  n=60 acc .92; hy3-free n=60 acc .82; laguna-s-2.1-free n=56(*) acc .80;
  gpt-oss-20b n=22 acc .82; glm-5.2 n=17 acc .88 (lenient, Wilson CIs in
  results/stage1/S1-P1/analysis/). gemma-4-31b: 0 valid records (upstream 429s; partial,
  resumable when the OpenRouter daily budget resets). Manifest: partial (budget).
  *laguna file has 60 lines incl. 4 transient-error records.
- Statistical results (lenient paired): no significant model differences (Cochran Q
  p=0.41; best pairwise McNemar p_exact=0.0625 → p_BH=0.31). H2 overconfidence: NULL at
  pilot scale (all gap CIs include 0; 2026 free models well-calibrated on this
  contaminated sample). Honest nulls recorded.
- Token-logprob calibration channel measured but degenerate (P(letter)≈1.0 at the
  committed position; n=8) — documented as a chat-API limitation.
- Error annotation (S1-P1 pool): 9 incorrect responses, all annotated (model-annotated
  by ZCode/GLM, labeled as such): 6/9 trace to item defects (E8) incl. one MMLU item
  missing its equations in text form; 1 knowledge error, 1 unclassifiable bare answer,
  1 degenerate generation. Pool too small → **S1-P2 extension launched** (120 items ×
  3 Zen models, seed 43) to enlarge the honest error pool.
- Subagents: benchmark generators (running), test suite (running).

### Next milestone
Finish S1-P2 + error pool annotation; benchmark build QC; Stage 2 evaluation.

## 2026-08-17 (takeover reconciliation and parser rescore)

- Re-derived state from disk before edits. Two handoff discrepancies were found:
  S1-P2 was already complete on disk (360 attempted records / 359 empirically usable),
  not merely launched; and `.coverage` plus test bytecode showed that some pytest
  process had run, despite no test command or outcome being recorded. The latter is
  not accepted as validation; a clean full-suite run remains pending. Uncommitted
  benchmark-verifier fixes also existed beyond the handoff snapshot and remain subject
  to an actual build and independent review.
- Added `scripts/rescore_records.py`, an offline, atomic, provenance-preserving rescore
  tool. Dry-run then apply covered 584 stored records, of which 531 had non-empty raw
  responses and references. S1-P1 had no changes. S1-P2 had four parser-field changes
  and two correctness flips (laguna statistics item A→C; nemotron biology item B→D).
  The original fields are preserved under `extra.pre_rescore`; no provider calls were
  made. Evidence: `results/stage1/rescore_manifest.json`.
- Regenerated both Stage 1 analyses with `MPLBACKEND=Agg .venv/bin/stembench report`.
  Rescored S1-P2: nemotron 113/120 (0.942), hy3 105/120 (0.875), laguna 103/119
  (0.866); Cochran Q=7.30, p=0.026, while post-hoc exact McNemar contrasts do not
  survive BH (minimum adjusted p=0.067). Reconciled the report, paper, and S1-P2
  manifest with these generated artifacts.

### Next milestone
Build and independently verify the bilingual candidate dataset; repair all generator
and QC failures and manually inspect a stratified RU/EN sample before Stage 2 use.

## 2026-08-17 (candidate-build validity audit)

- Added current workspace/repository `AGENTS.md` guidance and clarified the intentional
  parent-workspace versus nested-Git-repository scopes. Installed the official Codex
  `gh-fix-ci` and `security-best-practices` skills; no external app plugin was connected
  because native Git/CLI/web capabilities cover the present work without extra access.
- Produced an initial deterministic 624-pair / 1,248-record candidate build (seed
  20260817), with 892/892 verifier records passing and byte-identical `items.jsonl` on
  a second build. This build was **not accepted** after manual/model-agent review found
  defects that shared generator/verifier logic had missed.
- Rejected all 84 initially labeled olympiad pairs as inconsistent with the repository's
  own difficulty rubric: they were routine one-formula or standard-course exercises.
  Replaced the math and physics families with multi-concept challenge generators while
  preserving counts, IDs/order, answer types, determinism, and independent verification;
  chemistry challenge replacement remains in flight. Added a hard gate requiring two
  declared concepts plus a concrete challenge feature, explicitly as audit metadata and
  not as a substitute for expert judgment.
- Found and fixed a critical chemistry shared-error escape: the encoded coefficient of
  H2O in `CH4 + 2 O2 -> CO2 + 2 H2O` was 1 instead of 2, halving three generated answers
  while the verifier trusted the same bad metadata. The verifier now derives coefficients
  from the equation, checks atom balance and metadata consistency, and uses the named
  product rather than assuming the last RHS species.
- Bilingual/semantic review covered all 624 pairs programmatically plus every matrix cell
  and a fully read 18-pair stratified sample. Fixes in flight cover the exact limiting-
  reagent question mismatch, Russian case/morphology defects, gas-heating direction,
  gauge-pressure semantics, worked-solution defects, and natural bilingual wording.
- Added stable `template_id` propagation and number-masked structural-template metrics.
  The old digit-preserving Jaccard score understated dependence; Stage 2 will report a
  template-cluster bootstrap sensitivity analysis in addition to pair-clustered results.

### Next milestone
Finish the chemistry challenge rewrite and semantic repairs, then rebuild from scratch,
inspect the corrected sample and challenge tier, and accept the dataset only if every QC,
independent verifier, determinism, and artifact-freshness gate passes.

## 2026-08-21/22 (session 3: resumed after handoff)

- Discovered continuation-agent work (Aug 17–21): rescore applied (2 flips,
  `rescore_manifest.json`); benchmark rebuilt and hardened (D9 challenge-tier redesign
  after rejecting 84 mislabeled olympiad items; D10 template-dependence metrics;
  chemistry equation-coefficient verifier fix). Adopted its state as baseline.
- Fixed 4 real bugs found by the never-run test suite: NUM_RE alternation shadowing
  scientific notation (3.2×10^4 parsed as 3.2); unit symbols matching inside words
  ("N" in "Answer"); bold-wrapped MC letters ("**D**") unparseable; potential infinite
  loop in stratified_sample when n > total.
- Test suite: 192 passed, ruff clean. Critical-module coverage: parsing 98%,
  calibration/significance 100%, classification 97%, providers 94%, intervals 81%,
  agreement 76% (benchmark_gen ~excluded by design; ollama untested — no local server).
- Benchmark ACCEPTED: deterministic (byte-identical rebuild), 1,516/1,516 independent
  verifications pass, 624 pairs/1,248 records, MC letters 62/83/58/65, max 3-gram
  Jaccard 0.78 (flagged pair logged). Manual review: 12 stratified pairs hand-verified
  across subjects/difficulty/languages — all answers correct, RU natural, pairs
  aligned, challenge tier genuinely multi-step. Dataset card written
  (`docs/dataset_card.md`).
- Error annotation completed: all 71 real Stage 1 errors annotated (model-annotated);
  2 sampled records dropped after rescore flipped them to correct. Distribution: E10
  empty-response 62% [50,72] (hy3 22/26, laguna 17/27 of their errors), E8 item
  defects 20% [12,30], E6 11%, E0 4%, E1 3%. Report addendum + paper §6.3 updated.
- Stage 2 (S2-E1) launched; first process was killed by a session pause (35 records
  kept); relaunched and running. OR daily budget reset since Aug 17.
- goal.md statuses updated against evidence.

### Next milestone
Finish S2-E1; run pair-clustered bilingual analysis + template-cluster sensitivity;
write Stage 2 report + paper §6.4; final audit, placeholder sweep, commits, push.
