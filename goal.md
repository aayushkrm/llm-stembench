# goal.md — Requirement Traceability Matrix

Maps every requirement from `Project.md` (phases 0–10) and `PROJECT_EXECUTION_PROMPT.md`
(§7–§12) to an acceptance check, evidence location, and status.
Statuses: `not started` | `in progress` | `complete` | `partial` | `blocked`.

| ID | Requirement (source) | Acceptance check | Evidence | Status |
|----|----------------------|------------------|----------|--------|
| R0.1 | Literature study (Ph.0) | ≥20 verified sources, annotated | `docs/literature/annotated_bibliography.md` (33 verified; every entry fetched) | complete |
| R0.2 | Pilot dataset selection with criteria (Ph.0) | Memo, recorded before results | `docs/dataset_selection_memo.md` (dated before first run) | complete |
| R0.3 | Hypothesis registry before evaluation (Prompt §7.1) | Registry w/ confirmatory split | `docs/hypotheses.md` (registered pre-results) | complete |
| R1.1 | Version-pinned dataset loading (§7.2) | Pinned revision + checksum in manifests | `src/stembench/datasets/mmlu_stem.py`; manifests record rev `c30699e8356d` | complete |
| R1.2 | Real pilot on 2–3 models (Ph.1) | ≥2 real models, raw JSONL + manifests | `results/stage1/S1-P1` (5 models, 215 valid) + `S1-P2` (3 models, 359 valid), cost 0 | complete |
| R1.3 | Logits/logprobs when available | Stored when exposed; capability-detected | gpt-oss records carry `logprobs_raw` (20 recs); channel reported degenerate (post-commit) — `analysis/metrics.json` | complete |
| R1.4 | Accuracy/precision/recall (Ph.1) | Metric module + tests | `src/stembench/metrics/classification.py`; `tests/test_metrics_classification.py` | complete |
| R2.1 | ECE, MCE, Brier, NLL, reliability diagrams (Ph.2) | Implemented + tested + figures | `metrics/calibration.py`; `results/stage1/*/analysis/figures/reliability.png` | complete |
| R2.2 | CIs for accuracy (Ph.2) | Wilson + bootstrap, tested | `metrics/intervals.py`; `tests/test_metrics_intervals.py` | complete |
| R2.3 | χ² requirement + paired-correct tests (Ph.2, §7.3) | McNemar/Cochran implemented+used; χ² implemented+tested | `metrics/significance.py`; S1-P2 addendum (Q=7.30, p=0.026; post-hoc BH) | complete |
| R2.4 | Cohen's κ / multi-rater (Ph.2) | Implemented, tested; human κ NOT reported (no humans) | `metrics/agreement.py`; `scripts/aggregate_annotations.py` | complete |
| R3.1 | Error taxonomy ≥10 categories + adjudication (Ph.3) | Taxonomy doc | `docs/error_taxonomy.md` (E0–E10, adjudication order) | complete |
| R3.2 | Classify 100–200 incorrect responses (Ph.3) | Annotated sample, annotator declared | 71 real errors annotated (model-annotated); **shortfall vs 100–200 documented** per contract §7.4 | partial |
| R3.3 | Error distribution with uncertainty | Table + figure with CIs | `results/stage1/error_analysis/{error_distribution.json, figures/error_taxonomy.png}` | complete |
| R4.1 | Stage 1 course-project report (Ph.4) | Full report, traceable | `reports/stage1_report.md` (incl. S1-P2 addendum + merged errors) | complete |
| R5.1 | Benchmark spec + card + guidelines (Ph.5) | Docs complete | `docs/benchmark_spec.md`, `docs/benchmark_card.md` → `docs/dataset_card.md`, `docs/annotation/` (RU+EN) | complete |
| R6.1 | Raw dataset 600+ bilingual questions (Ph.6) | ≥600 pairs post-generation | 624 pairs / 1,248 records; `data/stembench_v1/DATASET_VERSION` (sha256 pinned) | complete |
| R6.2 | Procedural generation, independently verified | Second-code-path verification 100% | 1,516/1,516 verifier records pass; `verification_report.json` | complete |
| R6.3 | Bilingual pairing | pair_id links; identical answers | QC pair-completeness gate; byte-identical rebuild (CI-enforced) | complete |
| R7.1 | 2–3 experts, κ ≥ 0.75 (Ph.7) | Human labels + agreement | `docs/annotation/` package ready | **blocked** (no human experts; release gate; candidate label) |
| R7.2 | HF dataset v1 (Ph.7) | Published or ready bundle | `scripts/publish_hf.py` + card + hash; blocked on HF token | partial (ready, not published) |
| R8.1 | Full-scale run 6–8 models (Ph.8) | 6–8 real model runs | `results/stage2/S2-E1/`: 8 models, 5 complete at n=100 (520 evals, cost 0); gemma/gpt-oss/glm budget-capped at n=1/17/2 (OR 50/day shared; resume command in manifest notes) | partial (5/8 at full n; resume documented) |
| R8.2 | Metrics by subject/language/difficulty | Breakdowns with CIs | `results/stage2/S2-E1/analysis/stage2_analysis.json` (`categories`, `per_model_language`, small cells flagged) | complete |
| R9.1 | Hypothesis testing incl. RU–EN gap (Ph.9) | Paired cluster stats per registry | H4/H5 pair-clustered bootstrap + BH + template-cluster sensitivity: pooled +0.005 [−0.041, +0.050] p=0.82 (null, robust to template clustering) | complete |
| R9.2 | Heatmaps, CI plots, visualizations (Ph.9) | Script-generated figures | `results/stage2/S2-E1/analysis/figures/`: subject/difficulty/answer-type heatmaps, accuracy CI plot, language-gap forest | complete |
| R10.1 | Publish code to GitHub (Ph.10) | Non-destructive push to origin main | pushed `9f2685c..1f59de7` to `origin main` (GitHub `aayushkrm/llm-stembench`); working tree clean | complete |
| R10.2 | Paper draft (Ph.10) | Full EN draft + RU abstract | `paper/paper.md` (all sections incl. §6.4 with S2-E1 results), `paper/abstract_ru.md` (synchronized, null-gap result) | complete (draft; not submitted) |
| E1 | Provider-agnostic adapters (§7.2) | ≥2 real + fake | `providers/` (openrouter, zen, ollama, fake) | complete |
| E2 | Determinism/seeds/stratification recorded | Config+manifests store seeds | manifests; deterministic sampling tested | complete |
| E3 | Prompt templates hashed with runs | hash in every record | `prompt_hash` in all records; manifest stores text+hash | complete |
| E4 | Retries/backoff/budgets/checkpoint-resume | No duplicate calls on resume | `runner.py` (resume tested in `tests/test_runner_offline.py`) | complete |
| E5 | Normalized JSONL records full metadata | Schema-validated | `schemas.ResponseRecord`; all records validate | complete |
| E6 | Extraction: MC/exact/numeric tolerance | Parsers + Cyrillic tests | `parsing.py` 98% covered; bug-fix trail in `rescore_manifest.json` | complete |
| E7 | Dry-run/cost estimator/smoke/fake provider | CLI modes offline | `--dry-run`; budget trackers; fake provider | complete |
| E8 | Config-driven regeneration | One command regenerates | `stembench report`; `docs/reproducibility.md` | complete |
| T1 | Tests throughout (§10) | Suite green; critical coverage | 192 passed, ruff clean; critical modules 76–100% covered (audit.md) | complete |
| T2 | CI workflow | Actions: lint+tests+e2e+determinism+secrets | `.github/workflows/ci.yml`; first GitHub run green (run 32522130061: secret-scan + test 3.10 + test 3.12 incl. lint/coverage/e2e/determinism) | complete |
| Q1 | README/architecture/troubleshooting | Docs complete | `README.md`, `docs/architecture.md`, `docs/reproducibility.md` | complete |
| Q2 | LICENSE/CITATION/changelog/dataset card/checklist | Present, consistent | repo root + `docs/dataset_card.md`, `docs/release_checklist.md` | complete |
| A1 | Zero paid spend | Free endpoints only; cost 0 | allowlists + zero-spend guard (tested); all manifests cost 0 | complete |
| A2 | Honest final audit | audit.md reconciles with matrix | `audit.md` (2026-08-22 final pass: 12 questions answered, D12 found+fixed at audit, sweep clean) | complete |
| A3 | Reproducibility from fresh checkout | Documented commands | `docs/reproducibility.md`; CI fresh-install job | complete |

## Maintenance note
Statuses point at inspectable evidence only. Blocked items name the smallest external
action. R3.2 is partial by design: 71 real errors exist (all annotated); manufacturing
cases to reach 100 would violate the contract's shortfall rule.
