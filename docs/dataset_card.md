# STEMBench v0.1.0-candidate — Dataset Card

## Identity

- **Dataset**: STEMBench — bilingual (Russian/English) exact-science benchmark
- **Version**: `v0.1.0-candidate` (see `data/stembench_v1/DATASET_VERSION`)
- **Build**: seed `20260817`; `items.jsonl` SHA-256
  `bc74deada6a70613203dc3c261939a005a726d28064280d823c127c42b358998`; deterministic
  (byte-identical rebuild verified; enforced in CI)
- **Status**: **CANDIDATE / PRE-RELEASE.** Automated verification is complete. No human
  expert has validated the items yet — the expert workflow
  (`docs/annotation/`, gate: Fleiss' κ ≥ 0.75) is a **release blocker for v1.0**. No
  human agreement statistic is reported anywhere.

## Composition (from `verification_report.json`)

- **624 semantic pairs = 1,248 language records** (624 EN + 624 RU; every pair has
  exactly one EN and one RU variant linked by `pair_id` with identical answers).
- Subjects: chemistry 180, math 244, physics 200 pairs.
- Difficulty (rubric per item): school ≈ 41%, university ≈ 45%, olympiad/challenge
  84 pairs ≈ 13% — each challenge item declares ≥2 composed concepts plus a concrete
  challenge feature (audit metadata, not expert validation).
- Answer types: MC 43.0% (4 choices; correct-letter distribution A 62 / B 83 / C 58 /
  D 65), numeric 43.6% (tolerance + units), exact 13.5% (language-neutral canonical).
- Splits: `test` (606 pairs) + `dev` (18 pairs, format demonstration only — dev is
  **not** a held-out conceptual split; see Limitations).

## Creation

Original procedural generation: parameterized bilingual templates authored for this
project (no textbook/exam/dataset copying; no LLM-generated text). Parameters drawn
deterministically per (topic, index, seed); answers **computed in code and
independently re-verified by a second implementation** (alternative formula
arrangement, back-substitution, atom-balance checks, dimensional analysis):
1,516/1,516 verifier records pass. Distractors are plausible-error perturbations
(sign/factor/decimal shifts). Build date 2026-08-21 (parameters post-date current
training cutoffs — reducing, not eliminating, contamination risk).

## Quality gates (build fails if any fails)

Schema validity (pydantic); pair completeness; answer/tolerance/unit consistency
across variants; MC distractor distinctness; near-duplicate word-3-gram Jaccard gate
(max observed 0.78 under the threshold, flagged pair recorded); MC letter balance;
distribution reports; determinism + hash pinning.

## Intended use

- Evaluating LLM accuracy/calibration on school/university/olympiad STEM problems.
- **Paired within-item RU–EN language-gap studies** (cluster on `pair_id`).
- Error-taxonomy research; regression testing across model versions.

## Out-of-scope use

High-stakes decisions about individuals; training-data redistribution beyond the
license; claims of measuring "general intelligence"; cross-template generalization
claims (see limitations).

## Limitations (honest)

- **Template dependence**: items are parameterized instances of ~60 structural
  templates. Masking standalone numbers collapses the 624 EN questions to 305 unique
  strings; most `dev` templates also appear in `test`. Inference should use
  pair-clustered statistics **with the template-cluster sensitivity analysis**
  (`structural_templates` in the verification report); `dev` is for format
  validation only.
- Difficulty labels (especially the challenge tier) are generator-declared audit
  metadata pending expert confirmation.
- Single-parameter-set pairs bound, but do not eliminate, RU/EN surface-form
  confounds (reading difficulty, terminology familiarity).
- 2026-era models' exposure to similar textbook-style formulations cannot be
  excluded.

## Licensing & attribution

- Data: **CC-BY-4.0** (original work; attribution: "LLM-STEMBench contributors,
  STEMBench v0.1.0-candidate, link to repository").
- Code: MIT. Citation metadata in `CITATION.cff`.

## Maintenance

Deterministic rebuilds from seed; breaking changes bump the version and are recorded
in `CHANGELOG.md`; the verification report is regenerated with every build. Expert
validation results (when run) will be published with the annotations under the same
license, and the version promoted to v1.0 only after the κ gate passes.
