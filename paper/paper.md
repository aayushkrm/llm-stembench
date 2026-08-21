# STEMBench: A Reproducible Bilingual Benchmark and Free-Tier Evaluation Pipeline for Exact-Science LLM Assessment

**Draft — not submitted anywhere.** Version 0.1 (2026-08-17).
Authors: LLM-STEMBench contributors. License: MIT (code), CC-BY-4.0 (data).

> Every empirical number in this paper regenerates from committed artifacts; tables
> carry their source paths. Model results come exclusively from free-tier endpoints
> under a zero-paid-spend policy, with exact model IDs, dates, and sample sizes.

## Abstract

Public STEM benchmarks saturate: in our pilot, five 2026 free-tier language models
score 0.80–0.92 accuracy on a stratified MMLU-STEM sample and are statistically
indistinguishable (Cochran's Q p=0.41; all pairwise McNemar contrasts non-significant
after Benjamini–Hochberg correction), while two-thirds of their observed errors trace
to defective items rather than model failures. We contribute (1) a provider-agnostic,
budget-aware evaluation pipeline with checkpoint/resume, strict answer-contract
prompts, Cyrillic-robust extraction, paired statistical comparison (McNemar, Cochran's
Q, cluster bootstrap), and dual-channel calibration (self-report and token
probability, with an honest negative result on the latter); and (2) STEMBench, an
original bilingual Russian–English benchmark of procedurally generated mathematics,
physics, and chemistry problems (counts and composition in §5.1 and the build's
verification report; every figure in this paper regenerates from committed artifacts)
whose answers are computed and independently re-verified in code, spanning school,
university, and olympiad difficulty and three answer formats. Evaluating eight
free-tier models on stratified paired samples with pair-clustered bootstrap, we find
**no Russian–English performance gap** (pooled difference +0.005, 95% CI
[−0.041, +0.050]; no per-model gap after Benjamini–Hochberg); measured variance
instead concentrates in olympiad-tier items (50–75%), free-form answer formats, and
a decoding-budget reliability failure (13–16% of items lost to empty length-exhausted
responses at a 2048-token budget, language-balanced) that our taxonomy traces and
our protocol flags. The benchmark ships as a deterministic, hash-pinned, CC-BY-4.0
build with a complete
expert-validation package; human expert verification remains an explicit release gate,
reflected in the candidate version label.

## 1. Introduction

Large language models are routinely compared on exact-science benchmarks whose
reference answers are single-authored, whose items predate current training corpora,
and whose evaluation protocols vary across papers. Three problems follow: score
inflation through contamination [Sainz2023, Oren2023]; measurement noise from
inconsistent answer extraction; and analysis errors from treating paired outcomes
(items answered by every model) as independent [Dietterich1998]. Our course-project
pilot (Stage 1) demonstrates all three concretely on MMLU-STEM: near-saturated,
statistically indistinguishable models whose most frequent "errors" are miskeyed or
rendering-broken items.

We therefore build STEMBench (Stage 2): original bilingual Russian–English items in
mathematics, physics, and chemistry, generated procedurally with seeded parameters,
with answers computed in code and re-verified by independent code paths, and with
Russian and English variants of every item linked as pairs — enabling within-item
language-gap estimation with cluster-aware uncertainty. The evaluation pipeline is
provider-agnostic, runs entirely on free-tier endpoints under an enforced zero-spend
policy, records complete provenance for every response, and resumes across daily
budget limits without duplicate calls.

Contributions:
1. **Pipeline** (§4): budget-aware, checkpointed, fully-artifact-traced evaluation
   with preregistered paired statistics, dual-channel calibration, and parse-failure
   accounting (lenient and strict accuracy both reported).
2. **STEMBench** (§5): original bilingual pairs (per-language records linked by
   `pair_id`) across 3 subjects, 3
   difficulty levels, 3 answer formats; deterministic builds (seed + SHA-256); 100%
   independent-verification pass rate enforced at build time; CC-BY-4.0.
3. **Empirics** (§6): free-tier pilots on MMLU-STEM (5 models) and the original
   benchmark with paired analyses; honest nulls where scale limits power.
4. **Error taxonomy findings** (§6.3): on MMLU-STEM, 6 of 9 observed errors were item
   defects; taxonomy, protocol, and annotations are released (model-annotated,
   labeled as such; human validation gated).

## 2. Related work

Benchmarks and contamination. MMLU [Hendrycks2021mmlu] aggregated crowdsourced
exams; MMLU-Pro and SciEval increased difficulty and science coverage; GSM8K
[Cobbe2021] and MATH [Hendrycks2021math] target quantitative reasoning. Contamination
detection and n-gram overlap analyses [Sainz2023, Golchin2023, Oren2023] show
memorization of public test sets; our response is procedural generation with
post-cutoff parameters (2026-08-17) rather than detection alone.

Statistical comparison. Dietterich [1998] established McNemar's test for paired
classifier comparison; we extend the practice to LLM benchmark evaluation with
Cochran's Q across models, Holm/BH multiplicity control, and pair-clustered bootstrap
for bilingual designs; Wilson [1927] intervals for proportions.

Calibration. ECE and reliability diagrams [Guo2017, Naeini2015]; LLM verbalized
confidence [Kuleshov2018; Tian2023]. We measure both channels and report a negative
result for token-probability calibration through chat-completion APIs (§6.2).

Multilingual evaluation. MGSM [Shi2022] and Russian-language evaluation efforts
[ruMMLU; MERA] motivate paired bilingual designs; STEMBench contributes exact-science
coverage with verified answers in both languages under a controlled output contract.

Error analysis. Process-supervision and error taxonomies [Lightman2023] inform our
10-category taxonomy; annotation agreement statistics [Cohen1960, Fleiss1971] are
implemented for the expert-validation stage.

## 3. Scope and honesty constraints

All experiments run on free-tier endpoints (OpenRouter free models; Opencode Zen free
models) — 20 requests/minute and 50 requests/day on OpenRouter's free tier shape our
sample sizes; every table states per-model n with confidence intervals. No paid API,
no closed-weight frontier models: conclusions are about accessible open/free models.
The benchmark is released as v0.1.0-candidate pending independent human expert
validation; no human agreement statistic is reported.

## 4. Evaluation pipeline

### 4.1 Providers, budgets, resume

Provider adapters speak plain HTTP to OpenAI-compatible endpoints (OpenRouter,
Opencode Zen; Ollama for local use) behind one interface. A registry hard-codes the
verified free-model allowlists; any non-allowlisted model is refused before a request
is made (zero-spend enforced in code). A persistent per-provider daily counter
enforces documented caps; request starts are spaced to the provider rate; 429/5xx/
timeouts retry with bounded exponential backoff; three consecutive rate-limit
failures abort that model with a partial status. Runs checkpoint per item: completed
and parse-failed items are never re-requested; transient failures re-queue on the
next invocation of the same config.

### 4.2 Records and provenance

Each response becomes one schema-validated JSONL record: item and dataset revision,
provider + exact model ID, decoding, prompt hash, timestamps, raw response, parsed
answer + method, reference answer, correctness, confidence with separate provenance
channels, latency, token usage, estimated cost (0 throughout), error status, code
commit, and run ID. Analyses regenerate exclusively from these records; synthetic
(fake-provider) files are structurally excluded from empirical loaders.

### 4.3 Answer extraction and scoring

The prompt contract fixes output format (`Answer:` / `Confidence:`) in both languages.
Extraction handles common MC formats, Cyrillic look-alike letters (А→A), lowercase,
bare-letter lines, numeric parsing including decimal commas, thousands separators,
scientific notation and ×10^n forms, and unit detection. Scoring: MC exact-letter;
numeric tolerance (default 2% relative) with unit check; exact string match with
normalization and alternatives. Parse failures are never dropped: lenient accuracy
counts them as errors, strict conditions on parsed answers, and the parse-failure
rate is reported per model.

### 4.4 Statistics (preregistered)

The analysis plan (`docs/hypotheses.md`) was registered before any results:
confirmatory H1 (model differences; Cochran's Q + post-hoc McNemar with Holm), H2
(overconfidence gaps; item-level bootstrap CIs), H4/H5 (RU–EN paired gaps;
pair-clustered bootstrap with BH across models), plus descriptive H6–H8. Exploratory
analyses are labeled as such everywhere.

## 5. STEMBench: an original bilingual exact-science benchmark

### 5.1 Design

Subjects: mathematics, physics, chemistry. Languages: semantically aligned Russian–
English pairs sharing one parameter set and one computed answer. Difficulty: school /
university / olympiad with a one-sentence rubric justification per item. Answer
formats: 4-choice MC (plausible-error distractors, letter position counterbalanced),
numeric with tolerance + units, and language-neutral exact answers. Split
composition and counts: §6.1 and `data/stembench_v1/verification_report.json`.

### 5.2 Generation and verification

Items are template-parameterized originals authored bilingually; parameters are drawn
deterministically per (topic, index, seed). Every answer is computed in code and
independently re-verified by a second implementation (alternative formula
arrangement, back-substitution, or dimensional analysis; chemistry cross-checks
against an embedded periodic-mass table). A build fails loudly if any verifier,
schema, pairing, distractor-uniqueness, dedup (normalized-hash + 3-gram Jaccard), or
letter-balance gate fails. The same seed reproduces a byte-identical dataset
(SHA-256-pinned; checked in CI).

### 5.3 Quality gates, licensing, and validation status

Data: CC-BY-4.0 (original work); code: MIT. Automated gates: 100% verifier pass;
near-uniform MC letter distribution; max cross-pair 3-gram Jacard similarity below
threshold. **Human expert validation has not yet been performed**; the bilingual
annotator guidelines, blind-review workflow, adjudication rules, and agreement
tooling (Cohen's/Fleiss' κ with CIs) are released in `docs/annotation/`, and the
dataset version (v0.1.0-candidate) encodes the pending gate.

## 6. Experiments and results

### 6.1 Pilot on MMLU-STEM (Stage 1)

Five models, stratified n=60 (Zen) / 15–22 (OpenRouter, shared daily cap).
Accuracy (lenient, Wilson 95% CI): nemotron-3.5-lightning 0.917 [0.819, 0.964];
glm-5.2 0.882 [0.657, 0.967] (n=17); hy3 0.817 [0.701, 0.894]; gpt-oss-20b 0.818
[0.615, 0.927] (n=22); laguna-s-2.1 0.804 [0.682, 0.887]. No significant differences
(Cochran's Q p=0.41; best pairwise contrast p_BH=0.31). Overconfidence (H2): null —
all gap CIs include 0. Parse-failure rates 3–17%. Full tables:
`results/stage1/S1-P1/analysis/`.

The preregistered S1-P2 error-pool extension produced 359 valid evaluations on a
separate 120-item sample: nemotron 113/120 (0.942 [0.884, 0.971]), hy3 105/120
(0.875 [0.804, 0.923]), and laguna 103/119 (0.866 [0.793, 0.916]). The omnibus
Cochran Q was significant (Q=7.30, df=2, p=0.026), although both post-hoc nemotron
contrasts had BH-adjusted p=0.067. These values include an offline parser rescore of
stored responses (two wrong-to-correct flips, zero new API calls), audited in
`results/stage1/rescore_manifest.json`.

### 6.2 Calibration channels

Self-reported confidence is high (0.93–0.99) and calibrated on this sample (ECE
0.008–0.073 over parsed answers). The token-probability channel is **uninformative
through chat APIs**: at the first letter-dominated generated position the letter
probability is ≈1.0 (post-commit context; n=8 records with provider `top_logprobs`),
yielding ECE≈0 without measuring belief at decision time. We report this as a
negative result and caution against logprob-calibration claims derived from
post-commit positions in chat-completion APIs.

### 6.3 Error taxonomy on the pilot

All 71 incorrect responses across both pilot runs were annotated under a 10-category
taxonomy by the AI agent (clearly labeled model-annotated; human validation is a
release gate). Primary-label distribution (Wilson 95% CIs in
`results/stage1/error_analysis/error_distribution.json`): **E10 empty response 62%**
[0.50, 0.72] — the modal failure of these free-tier models is emitting no content at
all, concentrated in hy3 (22/26 of its errors) and laguna (17/27); **E8 item defects
20%** [0.12, 0.30] — miskeyed golds (an EPR item keyed to binomial(8) instead of the
correct binomial(9) intensities; a Wien-bridge item keyed to β=1/2 against the
unanimous textbook β=1/3), self-referential option sets, and an item whose parametric
equations are absent from its text rendering; E6 truncation-before-contract 11%; E0
bare wrong letters 4%; E1 knowledge errors 3%. Two conclusions: on a saturated public
benchmark, strong free models rarely fail substantively (5 of 71 errors), and a fifth
of observed "failures" belong to the benchmark, not the model — both motivating
independently verified items.

### 6.4 STEMBench evaluation (Stage 2)

Run S2-E1 (50 subject-stratified pairs, seed 2026; both language variants; eight
free-tier models, five complete at n=100, 520 evaluations total) tests the bilingual
questions H4/H5 on our benchmark. **No language gap exists for these models**: the
pooled EN−RU difference is +0.005 with pair-clustered 95% CI [−0.041, +0.050]
(p=0.82; template-clustered [−0.040, +0.048]), and no individual model's gap is
significant (all BH-adjusted p=1.0; CIs within ±0.14). Lenient accuracies span
72–85% with overlapping CIs (nemotron-3.5-lightning 85%, hy3 82%, ox-alpha 80%
(run at `reasoning_effort: max`, 8192 tokens), nemotron-3-ultra 77%, laguna 72%).
The benchmark discriminates through the olympiad tier (50–75% among models with
full parses) and free-form formats (numeric 69–97%, exact-string 50–92%); every
multiple-choice item any model answered was answered correctly, identifying
distractor weakness as the main v0.2 redesign target. Reliability separates the
models more than accuracy: four Zen models at a 2048-token budget lose 13–16% of
items to length exhaustion (empty body, `finish_reason: length`, language-balanced
21 EN/23 RU) — the mechanistic diagnosis of Stage 1's largest error class — while
ox-alpha at 8192 tokens never fails to answer yet ranks mid-pack and most
overconfident (ECE 0.18/0.22 by language). Self-reported confidence is ceiling-
compressed (mean 0.997–1.0) across all models, replicating the H2 pattern on an
original benchmark. Full tables, breakdowns, and the forest figure:
`results/stage2/S2-E1/analysis/` and `reports/stage2_report.md`.

## 7. Limitations

Free-tier budgets cap per-model n (17–100) and exclude closed-weight frontier models;
nulls reflect limited power, not equivalence. MMLU contamination likely inflates
Stage 1 absolutes (paired within-run contrasts less affected). RU/EN templates are
parallel but not translationally equivalent in reading difficulty — a confound the
paired design bounds but cannot eliminate. Expert validation of the benchmark is
pending; error annotations are model-made and exploratory. Token-probability
calibration was not measurable via the available chat APIs.

## 8. Ethics and safety

No personal data; benchmark items are original procedural generations (CC-BY-4.0);
free-tier terms respected; zero paid spend enforced in code; all provenance and
negative results published. Model-generated annotations are never labeled human.

## 9. Reproducibility statement

Fresh checkout → `pip install -e ".[dev]"` → `pytest` (offline, green) → deterministic
benchmark rebuild (hash-verified) → `stembench report` regenerates every table/figure
from committed raw records. Exact commands: `docs/reproducibility.md`. Raw records,
manifests (prompt hashes, decoding, git commits, budget usage), and the annotation
files are all committed.

## 10. Conclusion

On saturated public benchmarks, five accessible 2026 models are statistically
indistinguishable and most observed errors are item defects. STEMBench addresses the
measurement problem at its source: original, procedurally verified, bilingual,
hash-pinned items with a preregistered paired-analysis pipeline that runs on zero
budget. The expert-validation package makes the remaining human step concrete.

## References

See `docs/literature/references.bib` (33 verified sources; rendered citations in
`docs/literature/annotated_bibliography.md`).
