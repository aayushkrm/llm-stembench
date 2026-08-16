# goal.md — Requirement Traceability Matrix

Maps every requirement from `Project.md` (phases 0–10) and `PROJECT_EXECUTION_PROMPT.md`
(§7–§12) to an acceptance check, evidence location, and status.
Statuses: `not started` | `in progress` | `complete` | `partial` | `blocked`.

| ID | Requirement (source) | Acceptance check | Evidence | Status |
|----|----------------------|------------------|----------|--------|
| R0.1 | Literature study: benchmarks, error taxonomies (Project.md Ph.0) | ≥20 verified sources, annotated | `docs/literature/annotated_bibliography.md` (33 verified sources) | complete |
| R0.2 | Pilot dataset selection with criteria (Ph.0) | Selection memo with explicit criteria, recorded before results | `docs/dataset_selection_memo.md` | in progress |
| R0.3 | Research questions + hypothesis registry before full evaluation | Registry with confirmatory/exploratory split | `docs/hypotheses.md` | not started |
| R1.1 | Pipeline: dataset loading, version-pinned (Prompt §7.2) | Loader with pinned revision + checksum; tests pass | `src/stembench/datasets/mmlu_stem.py` | not started |
| R1.2 | Requests to 2–3 models via API adapters (Ph.1) | ≥2 real model runs, raw JSONL + manifests | `results/stage1/`, `results/manifests/` | not started |
| R1.3 | Collect responses and logits/logprobs if available | logprobs stored when provider exposes them; capability detected | run records `confidence_provenance` fields | not started |
| R1.4 | Accuracy, precision, recall metrics (Ph.1) | Metric module + unit tests vs hand fixtures | `src/stembench/metrics/` | not started |
| R2.1 | ECE, MCE, Brier, NLL, reliability diagrams (Ph.2) | Implemented + tested; figures generated | `src/stembench/metrics/calibration.py`, `results/stage1/figures/` | not started |
| R2.2 | Confidence intervals for accuracy (Ph.2) | Wilson + bootstrap CIs, tested | `src/stembench/metrics/intervals.py` | not started |
| R2.3 | χ² test requirement, paired tests where design demands (Ph.2 + Prompt §7.3) | McNemar (paired), Cochran's Q (>2), chi-square implemented | `src/stembench/metrics/significance.py`, tests | not started |
| R2.4 | Cohen's kappa / multi-rater agreement (Ph.2) | Cohen's κ + Fleiss' κ implemented, tested | `src/stembench/metrics/agreement.py` | not started |
| R3.1 | Error taxonomy for exact sciences (Ph.3) | Taxonomy doc with ≥10 categories + adjudication rules | `docs/error_taxonomy.md` | not started |
| R3.2 | Classify 100–200 incorrect responses (Ph.3) | Annotated sample; annotator identity declared; no fake "human" labels | `results/stage1/error_analysis/` | not started |
| R3.3 | Error distribution with uncertainty | Distribution table + figure from artifacts | `results/stage1/error_analysis/` | not started |
| R4.1 | Stage 1 course-project report (Ph.4) | Full report, claims traceable to artifacts | `reports/stage1_report.md` | not started |
| R5.1 | Benchmark structure: subjects, difficulty, protocol (Ph.5) | Spec + benchmark card + annotator guidelines | `docs/benchmark_spec.md`, `docs/benchmark_card.md`, `docs/annotation/` | not started |
| R6.1 | Raw dataset 600+ bilingual questions (Ph.6) | ≥600 RU-EN pairs (≥1,200 language records) post-generation | `data/stembench_v1/` + build report | not started |
| R6.2 | Original/procedural generation, independently verified | Items independently code-verifiable; verification report | `src/stembench/benchmark_gen/`, verification report | not started |
| R6.3 | Bilingual via translation/adaptation | RU/EN semantically paired; `pair_id` links variants | item schema | not started |
| R7.1 | 2–3 experts annotate, κ ≥ 0.75 (Ph.7) | Human expert labels + agreement analysis | `docs/annotation/` package | not started (expected blocked: no human experts available) |
| R7.2 | HuggingFace dataset v1 (Ph.7) | Published or ready-to-publish bundle with card+checksums | `releases/` bundle + upload script | not started |
| R8.1 | Full-scale run 6–8 models (Ph.8) | 6–8 real model runs with logs | `results/stage2/` | not started |
| R8.2 | Metrics by subject, language, difficulty | Category breakdowns with CIs, small cells caveated | `results/stage2/` | not started |
| R9.1 | Hypothesis testing incl. RU-EN gap (Ph.9) | Predefined hypotheses, paired stats on paired items | `docs/hypotheses.md`, `results/stage2/analysis/` | not started |
| R9.2 | Heatmaps, CI plots, visualizations (Ph.9) | Script-generated figures | `results/stage2/figures/` | not started |
| R10.1 | Publish code to GitHub (Ph.10) | Pushed to aayushkrm/llm-stembench | git log / remote | not started |
| R10.2 | Paper draft (Ph.10) | Full EN draft + RU abstract | `paper/paper.md`, `paper/abstract_ru.md` | not started |
| E1 | Provider-agnostic adapter interface (Prompt §7.2) | ≥2 real providers + fake for tests only | `src/stembench/providers/` | not started |
| E2 | Determinism, seeds, stratified sampling recorded | Config + manifest store seeds/strata | run manifests | not started |
| E3 | Prompt templates hashed, versioned with runs | prompt hash in every record | run records | not started |
| E4 | Retries, backoff, rate budgets, checkpoint/resume | Resume makes no duplicate calls; budget tracker | `src/stembench/runner.py` | not started |
| E5 | Normalized JSONL records with full metadata | Schema-validated records | `src/stembench/schemas.py` | not started |
| E6 | Answer extraction: MC, exact, numeric tolerance | Parsers + tests incl. Cyrillic/Unicode | `src/stembench/parsing.py`, tests | not started |
| E7 | Dry-run, cost estimator, smoke subset, fake provider | CLI modes work offline | `stembench run --dry-run` | not started |
| E8 | Config-driven regeneration end-to-end | One command regenerates metrics/tables/figures | `stembench report` | not started |
| T1 | Unit/property/contract/integration/e2e tests | Suite green; critical code covered | `tests/` | not started |
| T2 | CI workflow | Actions: lint + tests + offline e2e | `.github/workflows/ci.yml` | not started |
| Q1 | README quickstart, architecture, troubleshooting | Docs complete | `README.md`, `docs/architecture.md` | not started |
| Q2 | LICENSE, CITATION.cff, changelog, dataset card, checklist | Files present and consistent | repo root, `docs/` | not started |
| A1 | Zero paid spend | All calls to free-tier endpoints; cost 0 in manifests | run manifests | not started |
| A2 | Honest final audit incl. blocked items | audit.md reconciles with this matrix | `audit.md` | not started |
| A3 | Reproducibility from fresh checkout | Documented commands, env-safe | `docs/reproducibility.md` | not started |

## Maintenance note
Update statuses only with inspectable evidence (command output, artifact path, test log).
Never mark `complete` without pointing at evidence. Blocked items must name the smallest
external action needed.
