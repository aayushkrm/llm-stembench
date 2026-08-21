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

- Models (8, free tier, exact IDs in the run manifest): OpenRouter
  `google/gemma-4-31b-it:free`, `openai/gpt-oss-20b:free`, `z-ai/glm-5.2:free`,
  `stealth/ox-alpha` (decoding `max_tokens: 8192`, `reasoning_effort: max`; verified
  $0 on OpenRouter, sponsor-unlimited — decisions.md D11); Zen
  `nemotron-3.5-lightning-free`, `nemotron-3-ultra-free`, `laguna-s-2.1-free`,
  `hy3-free` (default `max_tokens: 2048`). Families: Google, OpenAI, Z.ai, Stealth,
  NVIDIA (×2), Poolside, Tencent.
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

Run S2-E1 finished 2026-08-22 with **520 evaluated model–item pairs** and overall
status `partial` (an honest field: three OpenRouter models remain below target n after
the shared 50-requests/day free-tier budget was exhausted; the resume command extends
them on a later day). Five models completed all 100 items (both language variants of
all 50 sampled pairs).

> **Scoring correction (2026-08-22, decisions.md D14).** The numeric unit check
> originally extracted units from the *entire* response, so units mentioned in
> reasoning text (e.g. the specific heat `J/(kg·K)` from the question) failed
> numerically correct answers. Error annotation surfaced the pattern ("exact gold
> value, spurious unit tag"); the check is now answer-scoped with unit-equivalence
> normalization (M ≡ mol/L, separator spellings, negative exponents), and all
> records were rescored (`rescore_manifest.json`): **19 False→True flips, 0
> True→False**. Numbers below are post-correction; the pre-correction standings
> under-scored ox-alpha by 11 points and laguna/ultra by 6/2.

**Headline accuracy** (lenient: parse failure counts as incorrect; items = 100;
Wilson 95% CIs; parsed accuracy in parentheses when it differs):

| Model (provider) | n | Accuracy [95% CI] | Parse failures |
|---|---|---|---|
| ox-alpha (OpenRouter) | 100 | **91%** [83.8, 95.2] | 0 |
| nemotron-3.5-lightning (Zen) | 100 | 85% [76.7, 90.7] | 1 |
| hy3 (Zen) | 100 | 82% [73.3, 88.3] (95.3%) | 13 |
| nemotron-3-ultra (Zen) | 100 | 79% [70.0, 85.8] (92.9%) | 14 |
| laguna-s-2.1 (Zen) | 100 | 78% [68.9, 85.0] (94.7%) | 16 |
| gpt-oss-20b (OpenRouter) | 17 | 76.5% — budget-capped | 0 |
| glm-5.2 (OpenRouter) | 2 | 2/2 — budget-capped | 0 |
| gemma-4-31b (OpenRouter) | 1 | 1/1 — budget-capped | 0 |

The five complete models' CIs overlap heavily (78–91%); as in Stage 1, no pairwise
accuracy difference is individually significant at this n.

**Language gaps (H4 per-model, H5 pooled; positive = EN better).** No model shows a
significant EN−RU difference; BH-adjusted p = 1.0 for every complete model.

| Model | n pairs | acc EN | acc RU | diff | pair-clustered 95% CI | p |
|---|---|---|---|---|---|---|
| ox-alpha | 50 | 92.0% | 90.0% | +0.020 | [+0.000, +0.060] | 0.72 |
| hy3 | 43 | 95.3% | 95.3% | 0.000 | [−0.070, +0.070] | 1.00 |
| laguna-s-2.1 | 38 | 94.7% | 94.7% | 0.000 | [−0.079, +0.079] | 1.00 |
| nemotron-3-ultra | 36 | 91.7% | 94.4% | −0.028 | [−0.139, +0.056] | 0.78 |
| nemotron-3.5-lightning | 49 | 85.7% | 85.7% | 0.000 | [−0.082, +0.082] | 1.00 |

**H5 pooled**: diff = **−0.0003**, 95% CI [−0.041, +0.039], p = 0.99 (n = 50 pairs).
Clustering on the 39 distinct templates instead of 50 pairs leaves the conclusion
unchanged: CI [−0.042, +0.039], p = 0.98. One model tilts slightly RU-better, one
slightly EN-better — no systematic direction. **H4/H5 are null: no bilingual
performance gap** on this controlled paired benchmark.

**Breakdowns (accuracy over parsed items, parsed/total shown where they differ;
complete models).** Subject: math is at ceiling for everyone (100%), chemistry
74–88%, physics 84–96%. Difficulty tiers discriminate at the top: olympiad ox-alpha
100% (12/12), hy3 100% — but over only **4 of 12** parsed (8 empty responses; a
small-cell artifact) — laguna 75% (8/12), ultra 64% (11/12), lightning 50% (12/12).
School 89–96%, university 90–95%. Answer format: every MC item a model actually
answered was answered correctly (**100% of parsed MC** for all five; parsed MC
n = 37–44 of 44); numeric 76–97% (the unit-scoring correction raised this band from
69–97%); exact-string 50–92% remains the hardest format and ox-alpha's one weak cell
(50%, 14/14).

**Self-reported calibration.** Mean stated confidence sits at 0.997–1.0 for every
model, so ECE ≈ 1 − accuracy wherever accuracy is below ~0.95: ox-alpha ECE 0.077
(EN) / 0.098 (RU) against accuracy 0.92/0.90; Zen models 0.02–0.10. Self-reports are
compressed at the ceiling and barely discriminate — the same pattern as Stage 1's H2
null, now on an original benchmark.

**Failure patterns.** All 44 parse failures come from four Zen models (hy3 13,
laguna 16, ultra 14, lightning 1; ox-alpha 0). 43 of 44 returned an **empty response
body**: 26 exhausted the token budget (`finish_reason: length`), 4 stopped without
answering, 13 report no finish reason, 1 errored. Failures are language-balanced
(21 EN / 23 RU) and concentrate in free-form formats and the olympiad tier (hy3
parsed only 4 of its 12 olympiad items).

## 5. In-depth analysis

**Language parity is the robust headline.** The paired design removes item difficulty
as a confound, pair-clustered bootstrap accounts for the two variants moving together,
and the template-cluster sensitivity check shows the H5 CI is stable ([−0.041, +0.039]
→ [−0.042, +0.039]) even though the 50 pairs span only 39 templates. With CIs this
tight (±0.04), any true EN−RU gap on this benchmark is at most ~4 percentage points
for these models — far from the gaps reported for earlier model generations on
non-paired translated benchmarks.

**Where the benchmark discriminates.** Not language, not MC: the informative cells are
the olympiad tier (50-point spread among fully-parsed models) and free-form answer
formats. hy3 scores 97% on numeric; ultra leads exact-string answers (92%) where
ox-alpha is weakest (50%). The universal 100%-of-parsed-MC ceiling says the
procedurally generated distractors are too weak for 2026 models — the single clearest
v0.2 design change (harder, error-shaped distractors; decisions.md). hy3's nominal
olympiad 100% is over 4 parsed items and carries no weight; ox-alpha's 100% (12/12)
is the only full-parse olympiad maximum.

**The reasoning model leads — and the reason it was first scored mid-pack is a
measurement lesson.** ox-alpha ran with `reasoning_effort: max` and 8192 output tokens
(accepted by the API; no fallback flagged in any record): after the scoring
correction it ranks first at 91% with zero parse failures, 100% math, 100% olympiad,
and the best self-report calibration of the five (ECE 0.077/0.098). Its one weak cell
is exact-string answers (50%): it returns verbose element names and bold-wrapped
formulas ("**Copper (Cu)**" vs gold "Cu") — an answer-contract mismatch, not a
knowledge failure. Its original mid-pack score was an artifact of whole-response unit
extraction picking units out of its verbose reasoning; the correction moved it +11
points. Meanwhile four Zen models at the default 2048-token budget lost 13–16% of
items to length exhaustion — protocol choices (token budgets, unit-string contracts)
moved measured accuracy by more than model choice at this tier.

**The Stage 1 empty-response cluster is now mechanistically diagnosed.** Stage 1's
largest error class (62% of annotated errors: empty completions from hy3/laguna)
reappears here with full metadata: empty body + `finish_reason: length`. It is a
decoding-budget failure mode, not a comprehension or Cyrillic-parsing failure — it is
language-balanced and fixable by protocol (larger `max_tokens` or separated reasoning
budgets), which changes how such failures should be annotated in future taxonomies
(provider/protocol fault class, not model-knowledge fault class).

**All 89 real Stage 2 errors were annotated (D13; model-annotated, never human).**
On the machine-verified benchmark the taxonomy finds **only two failure modes**:
E10 empty responses 43 (48.3% [38.2, 58.5]) and E6 answer-contract mismatches 46
(51.7% [41.5, 61.8]) — Unicode subscripts ("CH₂" vs "CH2"), element names vs symbols
("Copper (Cu)" vs "Cu"), un-parsed fractions ("2/7" = the gold 0.285714), verbal
Russian multipliers ("в 2 раза"), and truncation-before-answer. Zero knowledge-
category (E1–E5, E9) errors among them. Combined with Stage 1 (71) the annotated
pool is **160 real errors** — within the contract's 100–200 target — with combined
distribution E10 54.4%, E6 33.8%, E8 8.8% (all Stage 1 MMLU item defects), E0 1.9%,
E1 1.2% (`results/error_analysis/`). The annotation loop is also what surfaced the
unit-extraction scoring artifact (§4 note): taxonomy-driven review inspected "errors"
whose responses contained the exact gold value, which is measurement validation
working as designed.

**Template dependence is bounded but real.** 39 templates underlie the 50 sampled
pairs (dataset-wide: 305 unique masked templates for 624 pairs, decisions.md D10).
Every inferential claim above carries its template-clustered variant; none changes
status. Descriptive cells with n < 10 are flagged `small_cell` in the analysis JSON
and excluded from narrative claims.

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

1. **No bilingual gap.** On a paired, procedurally generated Russian–English STEM
   benchmark, five models from four organizations show no EN−RU difference: pooled
   −0.0003 [−0.041, +0.039], every per-model BH-adjusted p = 1.0, robust to
   template-clustered inference. For 2026 free-tier models, Russian-language STEM
   competence is at parity with English; bilingual evaluation effort belongs in
   reliability and coverage, not headline accuracy.
2. **The benchmark's discriminative power is in olympiad-tier and free-form cells,
   not language or multiple choice.** The 100%-of-parsed-MC ceiling across all models
   identifies weak distractors as the top v0.2 priority; numeric and exact-string
   formats and the olympiad tier carry all of the separation.
3. **Protocol artifacts rival model differences.** Two mechanical scoring/protocol
   effects moved measured accuracy more than model choice at this tier: (a) a
   whole-response unit extractor that failed numerically correct verbose answers
   (19 records, up to 11 points per model — caught by the error-annotation loop and
   corrected); (b) a 2048-token decoding budget under which four Zen models lose
   13–16% of items to empty length-exhausted responses (`finish_reason: length`,
   language-balanced). The reasoning model (ox-alpha, 8192 tokens, max effort) leads
   at 91% with zero parse failures and best calibration; its one weakness is
   answer-contract adherence on exact-string items, not knowledge.
4. **Honest boundaries.** Three OpenRouter models remain budget-capped (n = 1–17 of
   100; resume documented), MC distractors need redesign before v1.0, difficulty
   labels and item validity await independent expert review, and the dataset remains
   **v0.1.0-candidate** until that gate (Fleiss' κ ≥ 0.75) is passed.
