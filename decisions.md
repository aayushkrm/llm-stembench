# decisions.md — decision records

Short architecture/research decision records. Newest last.

## D1 — Pilot dataset: MMLU-STEM primary; MATH secondary deferred
**Decision.** Use MMLU-STEM (the STEM subjects of `cais/mmlu`, HF dataset) as the Stage 1
pilot; MATH secondary stress test deferred.
**Alternatives.** MATH alone; GSM8K; MMLU-Pro.
**Evidence.** MC answer format gives clean scoring, paired statistics across models, and
calibration via choice-letter probabilities/self-report; `cais/mmlu` is broadly used and
redistributable for evaluation; free-tier API budgets (D3) make free-response symbolic
scoring at meaningful N infeasible today.
**Consequences.** Free-response pipeline path is still implemented and unit-tested
(numeric tolerance, exact match); a real MATH mini-run remains a documented optional
extension. Recorded in goal.md as deferred, not silently dropped.

## D2 — Providers: OpenRouter + Opencode Zen (OpenAI-compatible), free models only
**Decision.** One `OpenAICompatProvider` parameterized by base URL; allowlists of free
model IDs per provider; Ollama adapter included for future local use; deterministic
`FakeProvider` for tests only. Persistent per-provider, per-day request counters enforce
the OpenRouter free-tier cap (20 rpm / 50 rpd) without duplicate calls on resume.
**Alternatives.** SDK-specific clients per provider; local-only inference.
**Evidence.** Both providers verified live (progress.md 2026-08-17); keys in gitignored
`.env`; OpenRouter documents free-tier limits; Zen free models intermittently saturated
per-model (429 with retry-later), handled by bounded backoff and item requeueing.
**Consequences.** Zero paid spend is enforced in code (`_assert_free`: refuse calls to
model IDs not on the free allowlist; record `cost` from provider when present).

## D3 — Model lineup under zero-spend
**Decision.** Stage 1 pilot: six real models across both providers — OpenRouter
`google/gemma-4-31b-it:free`, `openai/gpt-oss-20b:free`, `z-ai/glm-5.2:free` (smaller n,
shared 50/day cap) and Zen `nemotron-3.5-lightning-free`, `laguna-s-2.1-free`, `hy3-free`
(larger n if capacity holds). Stage 2: those six plus Zen `nemotron-3-ultra-free`
(7 models, 6 families: Google, OpenAI, Z.ai, NVIDIA, Poolside, Tencent/Hunyuan).
**Alternatives.** Commercial APIs (blocked: no spend authorization); local Ollama
(deferred by user); OpenRouter meta-route `openrouter/free` (rejected: exact model
attribution lost).
**Consequences.** Closed-weight frontier models are out of scope; documented as a
limitation. Sample sizes may differ per model by provider budget; all n reported.

## D4 — Confidence provenance kept separate
**Decision.** Two channels, never merged: (a) provider `top_logprobs` when the provider
exposes them (capability-detected, stored raw); (b) self-reported confidence elicited in
the output contract (`Confidence: 0-100`). Separate ECE per channel; no combined score.
**Evidence.** Contract §7.2; Guo et al. calibration literature (docs/literature).
**Consequences.** Calibration tables state the channel per row.

## D5 — Statistical procedures
**Decision.** Wilson score intervals for single proportions; McNemar test (exact
binomial + continuity-corrected χ²) for two paired models; Cochran's Q (+ post-hoc
McNemar with Holm correction) for >2 paired models; Pearson chi-square implemented and
tested for genuinely independent designs; stratified paired bootstrap (cluster = item for
Stage 1, pair_id for Stage 2 language gaps); Benjamini-Hochberg across the confirmatory
hypothesis family. Effect sizes (accuracy differences, paired Δ with CI) always reported
alongside.
**Alternatives.** Project.md's plain χ² for model comparison.
**Evidence.** Same items answered by every model ⇒ paired binary outcomes ⇒ McNemar/
Cochran per Dietterich (1998); χ² independence assumption violated under pairing.
**Consequences.** The source requirement "χ² test" is honored by implementing and testing
it, with the paired-correct tests used for the actual model comparisons (contract §13
improvement clause).

## D6 — Error annotation is agent-assisted, labeled as such
**Decision.** Stage 1 error taxonomy annotation is performed by the AI agent (ZCode/GLM),
multi-label, with per-item raw rationale preserved; every artifact and report says
"model-annotated", never "manual human annotation". Human expert validation remains an
explicit release gate (blocked item).
**Evidence.** Contract §4/§7.4 (no humans available; no fabricated agreement stats).
**Consequences.** No human kappa is computed anywhere.

## D7 — Stage 2 benchmark: original procedural bilingual generation
**Decision.** All items are original, template-parameterized procedural problems (math,
physics, chemistry) with answers computed and independently re-verified in code
(symbolic/numeric/dimensional checks). Russian and English variants are parallel
templates authored for this project. Data license CC-BY-4.0; code MIT. Dedup by
normalized-text hash + near-duplicate n-gram Jaccard; deterministic builds with seeds,
version IDs and checksums.
**Alternatives.** Copying from textbooks/exams (rejected: copyright + redistribution);
LLM free-generation (rejected as primary: unverifiable; allowed only as clearly labeled
auxiliary with independent verification).
**Consequences.** Fully redistributable; verification report generated per build;
expert human validation still required before v1.0 (pre-release labeling).

## D8 — Publication and release posture
**Decision.** Non-destructive pushes to `aayushkrm/llm-stembench` are authorized and used;
dataset published as a ready-to-upload versioned bundle + script (HuggingFace upload
itself requires an external HF token ⇒ blocked/ready state, honestly labeled). Paper is a
draft; no venue submission. Dataset version `v0.1.0-candidate` until expert validation.

## D9 — Reject the first candidate build and redesign the challenge tier
**Decision.** Do not use the first technically green 624-pair build for Stage 2. Replace
all seven initially labeled olympiad families in place with multi-concept challenge
tasks while preserving the preregistered counts, stable pair order, and answer-type mix.
Require generator-declared concepts and a concrete challenge feature for every such
item, then oversample the tier during genuine expert review.
**Alternatives.** Relabel all 84 items to school/university and leave zero olympiad
coverage; retain the labels and merely caveat them.
**Evidence.** An independent read-only audit found all 84 pairs failed the benchmark's
own difficulty rubric: direct remainder/binomial/quadratic tasks, one-formula circular/
projectile/lens tasks, and standard limiting-reagent calculations.
**Consequences.** Stage 2 is delayed until a rebuilt candidate passes verification and
manual/model-agent review. Declared challenge metadata is an auditable design guard,
not expert validation; `v0.1.0-candidate` and the human release gate remain unchanged.

## D10 — Report procedural-template dependence explicitly
**Decision.** Add stable `template_id` metadata, a number-masked structural duplication
report, dev/test overlap counts, and template-cluster bootstrap sensitivity estimates
alongside the primary pair-clustered bilingual analysis.
**Alternatives.** Treat parameter changes as independent based on digit-preserving
word-3gram Jaccard alone; split the small candidate immediately by whole topic families.
**Evidence.** The rejected build's 624 EN questions collapsed to 305 unique normalized
strings after standalone numbers were masked; 430 pairs belonged to repeated structural
groups and most physics dev templates also appeared in test.
**Consequences.** The original Jaccard gate remains useful for near-verbatim text but is
not described as conceptual independence. Dev is restricted to format validation, and
all inferential claims receive a template-cluster sensitivity check; a future expert-
validated release may adopt a fully template-held-out split.

## D11 — Add stealth/ox-alpha (user-requested) at maximum reasoning
**Decision.** Include OpenRouter `stealth/ox-alpha` as the 8th Stage 2 model with
`reasoning.effort="max"` (user request), max_tokens 8192.
**Evidence.** Live-verified pricing $0/$0 per 1M tokens (free tier, no `:free` suffix —
the allowlist test now encodes "suffix or verified-zero-price"); 1M context; provider
page documents reasoning-model behavior but no effort parameter, so the effort request
is sent per OpenRouter's unified reasoning API and, if the provider rejects it, retried
once at default reasoning with `_reasoning_fallback` stamped in the raw record (never
silent).
**Consequences.** OpenRouter's 50/day cap is shared with the other three OR models;
ox-alpha coverage accrues via the standard resume command as the daily budget resets.
Per-model n reported as always.
**Update (2026-08-22).** The user stated ox-alpha is usable without limit; live checks
agreed (it kept answering with the shared OpenRouter budget exhausted 50/50), so the
provider is constructed with `uncapped_models={"stealth/ox-alpha"}` — it bypasses the
local daily counter while the zero-cost assertion still applies to every response.
`effort: "max"` was accepted (no `_reasoning_fallback` in any of its 100 records; the
API reports `reasoning_tokens: 0`, i.e. it reasons without exposing token accounting).
It completed 100/100 in a single pass (80 correct, 0 parse failures).

## D12 — Manifest "evaluated" must exclude transient-error records
**Decision.** `n_total_evaluated` (and `counts.total_evaluated`) in run manifests
count only records with a real evaluation outcome (`error_status` "" or
`parse_failure`); transient errors (`daily_budget_exceeded`, `rate_limited`,
`timeout`, `provider_error`) are excluded, matching the analysis loaders.
**Evidence.** The S2-E1 manifest reported gemma 5 / gpt-oss 18 / glm 5 "evaluated"
when only 1 / 17 / 2 had responses — the gap was retryable error records (429s) left
in the JSONL after resume. Fixed in `runner.py` and all three existing manifests
(S1-P1, S1-P2, S2-E1) recomputed from raw records; correct/parse-failure counts were
already right and did not change. S1-P1 gemma is now honestly 0 evaluated.
**Consequences.** Coverage reporting can no longer overstate what providers actually
answered; no empirical analysis number changed (loaders always excluded these
records).
