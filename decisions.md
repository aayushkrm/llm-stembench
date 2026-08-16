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
