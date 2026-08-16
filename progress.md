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
