# STEMBench Annotation Package

Everything needed to run the expert validation of the benchmark (release gate for
v1.0). **Current status: PRE-RELEASE — no human expert validation has occurred yet.**
Until it has, the dataset carries the label `v0.1.0-candidate` and no human agreement
statistic may be reported anywhere.

## Contents

- `guidelines_en.md` — English annotator guidelines (authoritative)
- `guidelines_ru.md` — Russian annotator guidelines (working translation for experts)
- `annotation_format.md` — machine format for annotations (JSONL + CSV template)
- `workflow.md` — blind-review workflow, conflict resolution, adjudication, quality control

## Release gate (from benchmark spec §10 and goal.md R7.1)

1. 2–3 independent subject-matter experts annotate the sampled validation subset
   (see `workflow.md`) using these guidelines, working blind to each other's labels.
2. Agreement computed with **Fleiss' κ** (3 raters) or pairwise **Cohen's κ**
   (2 raters), implemented and unit-tested in `src/stembench/metrics/agreement.py`.
3. Gate: κ ≥ 0.75 on the correctness decision after adjudication round 1; iterate
   guideline clarifications (documented in `annotation_changelog.md`) until reached,
   while separately tracking correctness against the computed reference answers
   (agreement alone is not success).
4. Individual annotations are preserved (never only consensus labels).
5. Only then: version becomes `v1.0`, dataset card updated, κ reported with CI and
   prevalence caveats.

## Consent and ethics

Experts are volunteers/acknowledged contributors; no personal data beyond an anonymized
rater ID is collected; annotations are released under CC-BY-4.0 with the dataset;
participation may be withdrawn before publication (after publication, withdrawn labels
are removed in the next version).
