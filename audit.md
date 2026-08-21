# audit.md — Final Requirement-by-Requirement Audit

**Project:** LLM-STEMBench · **Audit date:** 2026-08-22 · **Repo:** `/Users/akm/Documents/LLM-Bench/llm-stembench` (GitHub `aayushkrm/llm-stembench`, branch `main`)

**Overall status: PARTIAL.** Everything achievable under the standing constraints
(zero paid spend, no human participants, no un-authorized publication) is done and
verified; the remaining gaps are irreducibly external (see §Blockers). No requirement
was silently weakened; every deviation is recorded in `decisions.md` (D1–D12).

## The 12 required questions (contract §12)

1. **Verified sources / bibliography.** 33 sources, every entry fetched and checked
   (23 on arXiv/ACL/Euclid; paywalled classics via reliable secondary pages, access
   URLs recorded). `docs/literature/annotated_bibliography.md`,
   `literature_review.md`, `references.bib`. Several commonly miscited attributions
   were corrected during verification.
2. **Pilot dataset and why.** MMLU-STEM (21-subject test split, revision
   `c30699e8356d`): MC format → clean scoring/paired stats/calibration; MIT license;
   comparability; contamination risk acknowledged. MATH deferred with rationale
   (free-tier budgets; free-response path still implemented + exercised in Stage 2).
   Memo dated before any results: `docs/dataset_selection_memo.md`.
3. **Stage 1 models/items/settings/costs.** S1-P1: 215 valid evaluations (nemotron
   n=60, hy3 60, laguna 56, gpt-oss 22, glm 17; gemma 0 — rate-limited), S1-P2: 359
   (three Zen models × 120/119/120). temperature 0, max_tokens 2048, prompt template
   `mc_answer_confidence_v1` (hash in every record). **Total cost: $0** (free-tier
   endpoints; manifests record cost 0). Manifests: `results/stage1/*/manifest.json`.
4. **Where results regenerate.** `stembench report --run results/stage1/<id>`
   (metrics.json, model_table.csv, pairwise_table.csv, figures/);
   `python -m stembench.analysis_stage2 results/stage2/S2-E1` (bilingual analysis +
   figures). Raw records are committed; loaders exclude synthetic `fake__*` files.
5. **Error taxonomy / who annotated.** Taxonomy E0–E10: `docs/error_taxonomy.md`.
   **All 71 real incorrect responses** annotated by the **AI agent (GLM) —
   model-annotated, never described as human**; evidence quotes per item in
   `results/stage1/error_analysis/annotations_merged.jsonl`; distribution with
   Wilson CIs in `error_distribution.json` (+ figure). Shortfall vs the 100–200
   target reported (71 real errors existed; manufacturing more would violate §7.4).
6. **Bilingual sizes / validation / hashes.** 624 pairs = 1,248 records raw **and**
   accepted (post-QC build is the dataset; ≥600/≥500 targets met). Independent
   verification: 1,516/1,516 pass (second code path). Determinism: byte-identical
   rebuild, CI-enforced. SHA-256 `bc74adea…358998` in `DATASET_VERSION`;
   `verification_report.json` carries counts, letter balance, near-dup max (0.78),
   and structural-template metrics.
7. **Experts?** **No.** No human annotated anything; no human κ is reported anywhere.
   The full validation package (bilingual guidelines, blind workflow, adjudication,
   calibration set, agreement tooling) is ready: `docs/annotation/`. The dataset is
   labeled `v0.1.0-candidate` — **this is the primary release blocker**.
8. **Stage 2 models / missing cells.** 8 models ran (S2-E1 final, 2026-08-22; 520
   evaluations, cost $0): **complete at n=100** — Zen nemotron-3.5-lightning (85
   correct), hy3 (82), nemotron-3-ultra (77), laguna-s-2.1 (72) and OpenRouter
   **stealth/ox-alpha (80; `reasoning_effort: max`, uncapped per D11, 0 parse
   failures)**; **budget-capped below target** — gemma-4-31b (n=1), gpt-oss-20b
   (n=17), glm-5.2 (n=2), all three blocked by the shared 50-requests/day OpenRouter
   free cap (50/50 used; resume command in the manifest extends them after reset).
   Headline result: **no EN–RU language gap** (H4/H5 null; pooled +0.005
   [−0.041, +0.050], p=0.82; robust to template clustering).
9. **Traceability of claims.** Every number in `reports/` and `paper/` cites its
   generated artifact; reports regenerate by command (`docs/reproducibility.md`);
   no hand-entered result values (verified by the placeholder sweep at this audit).
10. **Fresh-checkout reproduction.** `pip install -e ".[dev]"` → `pytest` (197 tests,
    offline, green; ruff clean) → dry-run e2e → deterministic benchmark rebuild
    (hash-checked) → `stembench report` regenerates all tables/figures from committed
    records. CI runs the same on push (incl. secret scan + build determinism).
11. **Tests / what could not run.** 197 tests green locally and on GitHub Actions
    (3.10 + 3.12; critical modules: parsing 98%, calibration/significance 100%,
    classification 97%, providers 94%, intervals 81%, agreement 76% coverage). Not
    runnable in CI: live-provider paths (no secrets in CI) and the Ollama adapter
    (no local server). Live API behavior was verified in-session with real runs
    instead. The first CI run failed (bare `pytest` could not import `scripts/`);
    fixed via `pythonpath` ini option (commit 1f59de7) and re-verified green.
12. **Actually pushed/released vs ready.** **Pushed:** repository to `origin main`
    (non-destructive; commit list below). **Ready but NOT published:** HF dataset
    bundle (needs write token), paper submission (no venue authorization), v1.0
    release (needs expert validation). Nothing claims otherwise anywhere.

## Validation evidence (commands run at this audit)

- `pytest tests/ -q` → 197 passed (exit 0); `ruff check src tests scripts` → clean.
- Benchmark rebuild × 2 → byte-identical `items.jsonl`; SHA-256 matches
  `DATASET_VERSION`.
- Stage 1 reports regenerated post-rescore (2 parser-fix flips documented in
  `results/stage1/rescore_manifest.json`).
- Stage 2 analysis regenerated from committed records: bilingual gaps, category
  breakdowns, failure patterns + 5 figures
  (`results/stage2/S2-E1/analysis/stage2_analysis.json`, `analysis/figures/`).
- Manifest-summary bug (D12) found at this audit and fixed: `n_total_evaluated`
  had counted retryable error records as evaluated; runner fixed and all three
  manifests recomputed from raw JSONL (gemma S1-P1 now honestly 0 evaluated;
  no analysis number changed).
- Placeholder/secret sweep: no `TODO/TBD/XXX/PLACEHOLDER` markers in reports/paper;
  no key-shaped strings in tracked files (CI enforces on push).

## Known limitations (honest)

- Sample sizes capped by free tiers; three OpenRouter models at n=1–17 of 100
  (documented per model; resume command extends after daily reset); wide CIs reported.
- Benchmark is template-parameterized; all inferential claims carry template-cluster
  sensitivity analyses (D10); dev split is format-only.
- Difficulty labels are generator-declared audit metadata pending expert review.
- Token-probability calibration measured degenerate via chat APIs (negative result,
  documented in Stage 1 report).
- Error annotations are single model-annotator (exploratory; release-gated).

## Blockers (external) and smallest remaining actions

| Blocker | Smallest external action |
|---|---|
| Human expert validation (κ ≥ 0.75 gate; dataset stays candidate) | Recruit 2–3 independent experts; run `docs/annotation/` workflow + `scripts/aggregate_annotations.py` |
| HF dataset publication | Provide HF write token; run `scripts/publish_hf.py --repo-id <user>/stembench` |
| OpenRouter models' n (gemma 1, gpt-oss 17, glm 2 of 100) | Re-run `stembench run --config configs/stage2_eval.yaml` after each daily reset (resume; no duplicate calls) |
| Paper submission | Explicit venue authorization from the user |

## Git state at audit

Commits on `main`: 9f2685c (initial) → 72828c4 (core pipeline + prereg docs) →
b5bc046 → 0a82950 (Stage 1 results) → 17ac8e7 (tests+benchmark+errors) →
fee503c (ox-alpha) → 3e9bcf0 (honest manifest counts, stage-2 figures, CLI) →
f7162a7 (S2-E1 results, reports, paper, audit) → 1f59de7 (pytest pythonpath CI
fix). Remote: **pushed non-destructively to `origin main`** (`9f2685c..1f59de7`);
first GitHub-Actions CI run green (32522130061: secret-scan + tests on 3.10/3.12
incl. lint, coverage, offline e2e, benchmark determinism). `.env` never tracked
(CI secret scan passing).
