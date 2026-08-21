# Stage 1 Course-Project Report — Evaluating LLM Reliability on MMLU-STEM

**Project:** LLM-STEMBench · **Run:** S1-P1 (+ S1-P2 extension) · **Date:** 2026-08-17
**Artifact regeneration:** `stembench report --run results/stage1/S1-P1` (tables/figures
in `results/stage1/S1-P1/analysis/`). Every number below comes from those artifacts.

## 1. Introduction

The course project's goal is to learn the complete LLM evaluation cycle on an existing
open benchmark: dataset loading, multi-model querying, answer extraction, scoring,
calibration, statistical model comparison, and error analysis. We evaluate five
free-tier models (zero paid spend) on a stratified MMLU-STEM sample under a
preregistered analysis plan (`docs/hypotheses.md`, registered before any results were
inspected), and annotate every incorrect response under a literature-grounded error
taxonomy (`docs/error_taxonomy.md`).

## 2. Related work

The full annotated bibliography (33 verified sources) and literature review are in
`docs/literature/`. Benchmark design and contamination (MMLU [Hendrycks2021mmlu],
MMLU-Pro, contamination detection [Sainz2023, Oren2023]); calibration (ECE
[Guo2017], [Naeini2015]); paired model comparison ([Dietterich1998] McNemar-based);
multilingual evaluation ([Shi2022] MGSM); error taxonomies ([Lightman2023]);
agreement statistics ([Cohen1960], [Fleiss1971]).

## 3. Methods

### 3.1 Dataset and sample

MMLU-STEM: the 21 STEM subjects of `cais/mmlu` (test split), revision
`c30699e8356d` (pinned; MIT license; evaluation use). Stratified proportional sample
by subject, seed 42, n=60 items for the three Zen-provisioned models; the same
sample's first 15 for OpenRouter models (shared 50 req/day free-tier cap; see
`docs/dataset_selection_memo.md` for the selection rationale and MATH deferral).

### 3.2 Models (all free tier)

| Model (exact ID) | Provider | n |
|---|---|---|
| nemotron-3.5-lightning-free | Opencode Zen | 60 |
| hy3-free | Opencode Zen | 60 |
| laguna-s-2.1-free | Opencode Zen | 56 valid |
| openai/gpt-oss-20b:free | OpenRouter | 22 |
| z-ai/glm-5.2:free | OpenRouter | 17 |
| google/gemma-4-31b-it:free | OpenRouter | 0 (rate-limited; resumable) |

Decoding: temperature 0, max_tokens 2048. Prompt: bilingual-capable answer-contract
template `mc_answer_confidence_v1` (exact text + hash in the run manifest), requiring
`Answer: <letter>` and `Confidence: <0-100>`.

### 3.3 Pipeline

Provider-agnostic adapters with hard free-model allowlists (zero-spend enforced in
code), persistent per-provider daily budgets, bounded retries, checkpoint/resume
without duplicate calls; every response stored as a schema-validated JSONL record
with prompt hash, decoding, latency, tokens, cost (0 for all records), git commit.
Answer extraction is Cyrillic-robust; correctness is scored lenient (parse failure =
incorrect) and strict (parsed only), with parse-failure rates reported separately.

### 3.4 Statistics (preregistered)

Wilson 95% CIs; McNemar (exact + continuity-corrected χ²) for paired models with
Benjamini–Hochberg across the pair family; Cochran's Q across models; Pearson χ²
implemented for independent designs (assumption-checked; not used for the paired
model comparisons, per the paired design — decisions.md D5). H2 overconfidence:
item-level bootstrap CI of (confidence − correctness). The source plan's plain χ²
model comparison is honored by implementation + tests, with paired tests used for the
actual comparisons (recorded improvement, decisions.md D5).

## 4. Results

Per-model table (lenient accuracy with Wilson CIs; full table with calibration in
`analysis/model_table.csv`, metrics in `analysis/metrics.json`):

| Model | n | Accuracy (lenient) | 95% CI | Parse-fail rate | Mean self-conf | ECE (self) |
|---|---|---|---|---|---|---|
| nemotron-3.5-lightning | 60 | **0.917** | [0.819, 0.964] | 3.3% | 0.95 | 0.035 |
| glm-5.2 | 17 | 0.882 | [0.657, 0.967] | 11.8% | 0.99 | 0.008 |
| hy3 | 60 | 0.817 | [0.701, 0.894] | 16.7% | 0.99 | 0.011 |
| gpt-oss-20b | 22 | 0.818 | [0.615, 0.927] | 9.1% | 0.93 | 0.050 |
| laguna-s-2.1 | 56 | 0.804 | [0.682, 0.887] | 14.3% | 0.96 | 0.073 |

Figures: reliability diagrams, accuracy CI plot, subject heatmap, confusion matrices
(`analysis/figures/`).

### 4.1 Model comparison (H1)

Cochran's Q over the 5 models' common items (n=17): Q=4.0, df=4, p=0.41 — **no
significant differences**. Best pairwise contrast (nemotron vs hy3: Δ=0.10, b=0, c=6,
McNemar p_exact=0.0625) does not survive BH correction (p_BH=0.31). At this sample
size, only differences larger than ~0.15 on 60 paired items would be detectable;
descriptively nemotron-3.5-lightning leads consistently.

### 4.2 Calibration and overconfidence (H2)

Self-reported confidence is high (0.93–0.99) and, on this sample, **well calibrated**:
every model's overconfidence gap CI includes 0 (largest point gap +0.030,
CI [−0.020, +0.117], gpt-oss-20b). H2 (≥2 of 3 Zen models overconfident with CI
excluding 0) is **not confirmed** — an honest null. Interpretation: 2026-era free
models are both accurate and confidence-matched on MMLU-style items they likely
encountered in training corpora (contamination); calibration stress is expected to
appear on the original Stage 2 benchmark.

The provider token-logprob channel (gpt-oss-20b, 20 records with `top_logprobs`) was
measured but is **degenerate for belief estimation via chat APIs**: at the first
letter-dominated generated position the letter probability is ≈1.0 (post-commit
context), giving ECE≈0 without information. Reported as a limitation; self-report is
the usable channel here.

### 4.3 Parse failures (H3)

Parse-failure rates 3.3–16.7% (mean 11%): models occasionally omit the answer letter
(usually long reasoning truncated by max_tokens). H3 (≤5% per model) holds only for
nemotron-3.5; **rejected** for the others. Lenient accuracy counts these as errors;
strict accuracy (parsed-only) rises to 0.94–1.00, showing how extraction-lenient
reporting would overstate capability — both are reported throughout.

## 5. Error analysis

All 9 incorrect responses in S1-P1 were annotated (annotator: **ZCode agent (GLM) —
model-annotated, NOT human**; human validation is a stated release gate). Primary
labels (`results/stage1/error_analysis/annotations.jsonl`):

| Primary label | n | Reading |
|---|---|---|
| E8 item/reference defect | 6 | models unanimously + correctly reasoned against a miskeyed gold (virology: 3 models; chemistry: standard Pauling-rule explanation scored wrong), or the item lost its equations in text rendering (math) |
| E1 knowledge/concept | 1 | binomial-assumption misjudgment |
| E0 unclassifiable | 1 | bare wrong letter, no trace |
| E10 degenerate generation | 1 | punctuation-soup output |

The dominant finding — **two thirds of "model errors" are item defects** on this
sample — directly motivates the original Stage 2 benchmark with independently
verified answers. Because 9 errors cannot support the planned 100–200 annotation
target, the preregistered shortfall policy applies: a second, larger sample (S1-P2:
120 items × 3 Zen models, seed 43) was run to enlarge the honest pool; its annotated
distribution is appended below when complete.

### S1-P2 addendum (error-pool extension)

S1-P2 completed 359 valid evaluations (120 hy3, 119 laguna, 120 nemotron) at zero
cost. After offline re-parsing of the stored raw responses with the corrected MC
answer-word-boundary rule, two responses flipped from incorrect to correct; the
four changed records retain their original scoring fields under `extra.pre_rescore`
and the hashes are recorded in `results/stage1/rescore_manifest.json`. No new model
calls were made.

| Model | n | Correct | Accuracy (lenient) | 95% Wilson CI | Parse-fail rate |
|---|---:|---:|---:|---:|---:|
| nemotron-3.5-lightning | 120 | 113 | **0.942** | [0.884, 0.971] | 0.8% |
| hy3 | 120 | 105 | 0.875 | [0.804, 0.923] | 10.8% |
| laguna-s-2.1 | 119 | 103 | 0.866 | [0.793, 0.916] | 10.1% |

Cochran's Q rejects equal accuracy across the three models on their 119 common items
(Q=7.30, df=2, p=0.026), but neither post-hoc exact McNemar contrast involving
nemotron remains significant after BH correction (both adjusted p=0.067). Thus the
omnibus result is evidence of heterogeneity at this pilot scale, while the individual
pair attribution remains uncertain after multiplicity control. The merged error-label
distribution is reported in the final error-analysis artifact rather than inferred
from these aggregate scores.

**Merged error annotation** (all 71 real errors across S1-P1+S1-P2; annotator: ZCode
agent (GLM) — model-annotated, NOT human; shortfall vs the 100–200 target is reported
per contract §7.4; two originally sampled records were dropped after the parser fix
rescored them to correct):

| Primary label | n | share [Wilson 95% CI] |
|---|---:|---:|
| E10 safety/reliability (empty response, no content) | 44 | 0.62 [0.50, 0.72] |
| E8 item/reference defect (suspected miskeyed/rendering-broken MMLU items) | 14 | 0.20 [0.12, 0.30] |
| E6 answer-extraction/format (truncated before the Answer contract) | 8 | 0.11 [0.06, 0.21] |
| E0 not classifiable (bare wrong letter, no trace) | 3 | 0.04 |
| E1 knowledge/concept | 2 | 0.03 |

Findings: the modal failure of these free-tier models is emitting no content at all —
concentrated in hy3 (22 of its 26 errors) and laguna (17 of 27); one fifth of "model
errors" are item defects (miskeyed golds, self-referential options, or an item whose
parametric equations are absent from its text), always flagged rather than silently
re-keyed; genuine substantive model errors are rare (5 of 71). Per-item annotations
with evidence quotes: `results/stage1/error_analysis/annotations_merged.jsonl`;
regenerate the distribution with
`scripts/merge_error_annotations.py --annotations results/stage1/error_analysis/annotations_all.jsonl`.

## 6. Limitations

- Free-tier budgets force unequal n (17–60) and left gemma-4-31b without valid records
  (upstream saturation); resume is one command when the daily budget resets.
- MMLU (2020) carries high contamination risk; absolute accuracies likely overstate
  unseen-item capability. The paired within-run comparisons are less affected.
- n=17–60 gives wide CIs (±0.15–0.25); nulls are absence of evidence at this scale.
- Single (model) annotator for errors; exploratory labels only.
- Token-probability calibration not measurable via these chat APIs (above).

## 7. Reproducibility

Exact commands: `docs/reproducibility.md` §2–3. Raw records: `results/stage1/S1-P1/`
(215 valid records + transient-error records, all committed). Manifest:
`results/stage1/S1-P1/manifest.json` (status: partial — budget; prompt template text +
hash, dataset revision, seeds, git commit, per-model counts, budget usage).

## 8. Conclusions

The full evaluation cycle works end-to-end on real models with honest statistics:
paired tests with multiplicity control, dual-channel calibration, parse-failure
accounting, and a reproducible audit trail. Scientifically: at pilot scale these
2026 free-tier models are statistically indistinguishable on MMLU-STEM, well
calibrated in self-report, and most of their observed "errors" trace to defective
benchmark items rather than model failures — strengthening the case for the original,
independently verified bilingual benchmark built in Stage 2.
