# Validation Workflow (facilitator runbook)

## 1. Setup

1. Freeze the dataset build under validation: record its version + SHA-256
   (`data/stembench_v1/DATASET_VERSION`) in the validation session log.
2. Draw the validation sample with the provided script (stratified by subject ×
   difficulty × language, seeded, recorded): `scripts/draw_validation_sample.py`
   — default 120 items (60 pairs × both languages) + the 10-item calibration set.
3. Assign rater IDs R1..R3; experts must not see each other's assignments.
4. Experts read the guidelines (EN authoritative, RU available) and complete the
   calibration set; the facilitator debriefs discrepancies **before** round 1.

## 2. Blind round 1

- Each expert annotates the full sample independently (2–3 h).
- Export JSONL/CSV per rater → `annotations_round1_R*.jsonl`.

## 3. Agreement computation

Run `scripts/aggregate_annotations.py --round 1`. Gate: Fleiss' κ (correct decision)
≥ 0.75. Report pairwise Cohen's κ as supplementary. Record everything, including
agreement on difficulty and language sub-decisions.

## 4. Revision round

Facilitator anonymizes and redistributes only disagreeing items; experts revise or
defend; `annotations_round2_R*.jsonl`. Re-compute agreement. Iterate (max 3 rounds —
if the gate is still missed, record the final κ and the item-level disputes; the
dataset stays pre-release or items are pruned by majority + facilitator adjudication,
logged).

## 5. Correctness vs computed answers

Separately from agreement: compare the majority `correct` decision against the
dataset's computed `canonical_answer`. Items where the majority of experts say
`invalid` while the computation says otherwise are escalated to a full worked-solution
review; either the item is fixed (new version, changelog entry) or removed. The final
report states: n items validated, n changed, n removed, final κ with CI.

## 6. Release

On gate satisfaction: bump to v1.0, regenerate the dataset with
`quality_flags=expert_validated` on validated items, update the dataset card, publish
annotations (CC-BY-4.0) with the dataset, acknowledge experts (or per their consent
choice).

## Honesty rules (non-negotiable)

- No agreement statistic is published from anything other than genuine independent
  human round-1 labels.
- Model-assisted pre-labels (if ever used to speed experts up) must be disclosed in
  the dataset card and excluded from κ computation of the affected fields.
- If fewer than 2 experts complete the round, the dataset remains pre-release.
