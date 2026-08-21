# Stage 2 Final Qualifying-Work Report — An Original Bilingual STEM Benchmark and Multi-Model Evaluation

**Project:** LLM-STEMBench · **Run:** S2-E1 · **Dataset:** STEMBench v0.1.0-candidate
(seed 20260817, sha256 `bc74dead…`) · **Date:** 2026-08-21/22
**Regeneration:** `python -m stembench.analysis_stage2 results/stage2/S2-E1` (analysis
JSON + figures); every number below comes from `results/stage2/S2-E1/analysis/`.

## 1. Introduction

Stage 2 delivers the two contributions the course project motivated: (1) **STEMBench**,
an original bilingual Russian–English benchmark of exactly-scored mathematics, physics,
and chemistry problems with procedurally computed and independently verified answers;
and (2) a multi-model evaluation on it with pair-clustered bilingual statistics,
executed entirely on free-tier endpoints under the zero-spend policy. The Stage 1
pilot showed why: on saturated public benchmarks, accessible 2026 models are hard to
separate statistically and a fifth of observed "model errors" were item defects
(`reports/stage1_report.md`).

## 2. The benchmark

Construction, quality gates, licensing, and honest limitations (template dependence,
pending expert validation) are specified in `docs/benchmark_spec.md` and
`docs/dataset_card.md`. Summary: **624 semantic pairs / 1,248 language records**
(chemistry 180, math 244, physics 200; school/university/olympiad tiers; MC 43% /
numeric 44% / exact 13%; correct letters balanced 62/83/58/65 across A–D); answers
computed in code and re-verified by an independent second implementation (1,516/1,516
pass); deterministic byte-identical rebuilds (CI-enforced); CC-BY-4.0 data, MIT code.
Status: **v0.1.0-candidate** — the human expert-validation gate (Fleiss' κ ≥ 0.75
workflow in `docs/annotation/`) has NOT been run; no human κ is reported.

## 3. Evaluation design (preregistered)

- Models (7, free tier, exact IDs in the run manifest): OpenRouter
  `google/gemma-4-31b-it:free`, `openai/gpt-oss-20b:free`, `z-ai/glm-5.2:free`; Zen
  `nemotron-3.5-lightning-free`, `nemotron-3-ultra-free`, `laguna-s-2.1-free`,
  `hy3-free`. Families: Google, OpenAI, Z.ai, NVIDIA (×2), Poolside, Tencent.
- Items: 50 subject-stratified pairs (seed 2026) → both language variants evaluated
  as separate items per model; identical prompts, temperature 0, answer-contract
  template with self-reported confidence.
- Statistics per `docs/hypotheses.md`: H4 per-model EN−RU paired differences with
  pair-clustered bootstrap CIs + BH across models; H5 pooled gap; **template-cluster
  sensitivity** for every inferential claim (decisions.md D10); H6–H8 descriptive
  cells (n<10 flagged); exploratory model×language interaction, per-language
  calibration, failure patterns.
- Budget honesty: OpenRouter free tier shares 50 requests/day across its models —
  per-model n is reported and the resume command extends coverage; Zen has no
  documented account cap (per-model upstream saturation handled by bounded retries).

## 4. Results

*(populated from `results/stage2/S2-E1/analysis/stage2_analysis.json` when the run
completes; per-model n, EN/RU accuracy with Wilson CIs, H4/H5 gaps with pair- and
template-clustered CIs, subject/difficulty/format breakdowns, failure patterns,
per-language calibration.)*

## 5. In-depth analysis

*(difficulty gradients; answer-format effects; calibration by language; empty-response
reliability pattern and its consistency with the Stage 1 error taxonomy; template
sensitivity of all inferential claims.)*

## 6. Limitations

Free-tier budgets cap per-model n (unequal across providers; documented per model);
closed-weight frontier models are out of scope; benchmark items are parameterized
instances of ~60 structural templates (pair-clustered inference carries a
template-cluster sensitivity check; dev split is format-only); difficulty labels and
item validity await human expert review; RU/EN surface-form confounds are bounded by
the paired design but not eliminated.

## 7. Reproducibility

Raw records + manifest committed under `results/stage2/S2-E1/`; deterministic dataset
rebuild (§ `docs/reproducibility.md`); analysis regenerates offline from records;
every figure from `src/stembench/viz/figures.py`.

## 8. Publication status and release gates

Code+data repository pushed non-destructively; dataset bundle release-ready with
hash-pinned card (`scripts/publish_hf.py`; HF upload requires an external token);
paper draft (`paper/`) not submitted anywhere. Remaining gates: human expert
validation (v1.0), HF publication, venue decision.

## 9. Conclusions

*(drawn from §4–5 after the run completes.)*
