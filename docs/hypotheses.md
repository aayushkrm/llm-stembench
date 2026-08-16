# Research Questions & Hypothesis Registry

**Registered 2026-08-17, before inspection of any model results** (dry-run artifacts
with the fake provider are synthetic and carry no information about real models).

Model set (free tier, decisions.md D3): OpenRouter `google/gemma-4-31b-it:free`,
`openai/gpt-oss-20b:free`, `z-ai/glm-5.2:free`; Zen `nemotron-3.5-lightning-free`,
`laguna-s-2.1-free`, `hy3-free`, (Stage 2 adds `nemotron-3-ultra-free`).

## Confirmatory hypotheses (Stage 1, MMLU-STEM)

- **H1 (model differences).** Accuracy differs across the three Zen models evaluated on
  the same 60 items. Test: Cochran's Q on the paired correctness matrix; reject H0 at
  α=0.05. Effect sizes: pairwise accuracy differences with Wilson CIs; post-hoc
  pairwise McNemar with Holm correction.
- **H2 (calibration).** Models are overconfident on MMLU-STEM: mean self-reported
  confidence exceeds accuracy (positive confidence–accuracy gap) for at least 2 of 3
  Zen models. Test: paired cluster bootstrap CI of (confidence − correctness) per
  model; confirmatory claim requires CI excluding 0. Report ECE/MCE/Brier per model.
- **H3 (parse failures).** Parse-failure rate ≤ 5% per model under the answer-contract
  prompt; failures counted (not dropped) and lenient vs strict accuracy both reported.
  Descriptive criterion, no significance test.

## Confirmatory hypotheses (Stage 2, original bilingual benchmark)

Primary unit: the question PAIR (ru+en variants of one item share `pair_id`);
cluster-aware statistics resample pairs.

- **H4 (language gap).** For each model, accuracy on EN variants minus accuracy on RU
  variants on paired items. Test: cluster (pair-level) bootstrap 95% CI of the paired
  difference; a gap is claimed only when the CI excludes 0. Multiplicity: BH across
  the 7 model-specific tests within this family.
- **H5 (family-level language gap).** Pooled across models (pairs clustered, model as
  stratification), EN > RU overall. Test: cluster bootstrap CI.
- **H6 (subject effects).** Accuracy differs by subject (math/physics/chemistry).
  Descriptive with CIs per cell; cells with n<10 flagged, not tested.
- **H7 (difficulty gradient).** school > university > olympiad accuracy (one-sided
  Page-like trend assessed descriptively with CIs; Jonckheere–Terpstra if sample sizes
  allow).
- **H8 (answer-format effect).** MC accuracy > numeric/exact accuracy per model
  (descriptive CIs; not inferential — formats differ by construction).

## Exploratory (clearly labeled, no confirmatory claims)

- Interaction model × language on accuracy (stratified tables; mixed-effects logistic
  regression if cell sizes permit, reported as exploratory).
- Calibration differences across languages (ECE_EN vs ECE_RU per model).
- Error-taxonomy distribution differences across subjects and languages on the
  annotated incorrect-response sample (Stage 1) — exploratory due to single-annotator
  (AI-assisted) design.
- Failure/abstention patterns: rate of empty/refusing/overshoot responses by model.

## Multiple-comparison policy

Confirmatory families: {H1 post-hoc pairs → Holm}, {H2 per-model gaps → BH within H2},
{H4 per-model language gaps → BH within H4}. Exploratory analyses are never corrected
into the confirmatory families and are labeled as such in every table.

## Deviations

Any deviation from this registry (sample sizes forced by provider budgets, dropped
models, re-registered hypotheses) is recorded in this file with the reason and date,
never silently.
