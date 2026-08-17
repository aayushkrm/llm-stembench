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
