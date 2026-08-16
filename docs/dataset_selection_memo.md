# Dataset Selection Memo — Stage 1 Pilot

**Decision recorded 2026-08-17, before inspecting any model results.**

## Candidates considered

| Dataset | Format | License/redistribution | Access | Contamination risk | Multilingual | Logits/confidence | Notes |
|---|---|---|---|---|---|---|---|
| **MMLU (STEM subset)** | 4-way MC | MIT (cais/mmlu); eval use clean | HF hub, cached, pinned revision | high (public since 2020) | EN (+unofficial RU translations exist) | choice-prob via logprobs or self-report | 21 STEM subjects, 3,445 test items |
| MATH | free response (LaTeX) | MIT | HF | high | EN | self-report only | symbolic scoring hard at pilot scale |
| MMLU-Pro | 10-way MC | restrictive-ish (CC-BY-NC-SA subsets) | HF | high | EN | same as MMLU | harder, longer; 10 choices complicate letter-prob calibration |
| GSM8K | free response numeric | MIT | HF | very high | EN | self-report | arithmetic reasoning; RU sibling MGSM exists |
| SciEval | MC + free | CC-BY-NC-SA — evaluation only | HF | medium | EN | mixed | scientific QA; NC license limits redistribution |
| TruthfulQA / HaluEval | MC/free | Apache-2.0 / research | HF | medium | EN | self-report | truthfulness/hallucination focus — out of scope for exact-science pilot, cited in literature review |

## Selection

**Primary pilot: MMLU-STEM** (test split, STEM subject group, revision-pinned).
**Secondary (MATH): deferred** — see below.

## Rationale against the stated criteria

1. **Subject coverage**: 21 STEM subjects spanning math, physics, chemistry, biology,
   CS, EE — matches the project's exact-science scope.
2. **Answer format**: 4-way MC → unambiguous scoring, paired binary outcomes across
   models (enables McNemar/Cochran's Q), and calibration from both self-report and
   choice-letter logprobs where the provider exposes them.
3. **License/availability**: MIT; loaded from the HF hub with a pinned revision
   (c30699e8356d recorded in run manifests); used for evaluation only, not redistributed.
4. **Contamination risk**: high (public since 2020) — acknowledged as a limitation of
   any public-benchmark pilot; this risk is a primary motivation for the original
   Stage 2 benchmark (procedurally generated 2026 parameters).
5. **Multilingual relevance**: EN-only, but the pipeline and prompts are bilingual and
   Stage 1's parsers are Cyrillic-hardened; the bilingual question is answered by design
   in Stage 2 on the original benchmark.
6. **Confidence support**: self-report channel always available; token-logprob channel
   capability-detected per model.
7. **Runtime/comparability**: short items fit free-tier API budgets; MMLU is the most
   widely reported benchmark, so future numbers are comparable to the literature.

## Why MATH is deferred (recorded, not silently dropped)

Free-response symbolic equivalence scoring is implemented (exact + numeric tolerance),
but under the zero-spend free-tier budgets (OpenRouter 50 req/day shared; Zen
intermittent per-model saturation) a MATH sample large enough for the paired analyses
(n ≥ 50 per model) is not achievable inside the pilot window. The free-response path is
exercised for real in Stage 2 (numeric/exact items on the original benchmark). A MATH
mini-run remains a documented optional extension; see goal.md R1.1 note and decisions.md
D1.

## Sampling design

Stratified proportional sample by subject, seed 42, n=60 for Zen-provisioned models,
first 15 of the same sample for OpenRouter models (shared daily cap), same seed across
models so all comparisons are paired on identical items.
