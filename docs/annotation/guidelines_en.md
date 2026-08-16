# STEMBench Expert Annotation Guidelines (English)

Version 1.0 — for the validation of STEMBench v0.1.0-candidate.

## Your task

For each assigned item you judge four things, **independently and blind** (do not
discuss items or labels with other annotators until the round is closed):

1. **Correctness of the reference answer** — is `canonical_answer` actually the
   correct answer to the question as posed? (valid / invalid / ambiguous)
2. **Unambiguity** — could a competent solver reasonably defend a different answer?
   (unambiguous / ambiguous / broken)
3. **Difficulty label** — school / university / olympiad, per the rubric below.
4. **Language quality** (for your language of expertise): is the question natural,
   grammatical, and terminologically correct? For RU items also: does it match the EN
   variant's meaning (you may consult the paired item)?

You may use a calculator/computer for arithmetic, but solve problems yourself — the
point is independent expert judgment of the items, not of any model.

## Rubric: difficulty

- **school**: one concept, 1–2 routine steps (typical grades 7–9 work).
- **university**: multi-step application of standard first-year university formulas or
  2–3 concept composition.
- **olympiad**: needs combining ≥2 concepts with a non-obvious step, edge-case care, or
  competition-style insight.

## Rubric: unambiguity codes

| Code | Meaning |
|---|---|
| `unambiguous` | exactly one defensible answer |
| `ambiguous` | two defensible readings leading to different answers (note both) |
| `broken` | missing data, contradictions, or unanswerable as posed |

## Rubric: language codes

| Code | Meaning |
|---|---|
| `ok` | natural, correct terminology |
| `minor` | awkward but unambiguous phrasing |
| `major` | grammatical/terminology error that could mislead a solver |
| `mismatch` | (paired items) RU and EN variants differ in meaning or given data |

## Output format

One row per item (JSONL or CSV — see `annotation_format.md`):

```
item_id | rater_id | correct(valid/invalid) | unambig(unamb/ambig/broken) |
difficulty(school/university/olympiad) | language(ok/minor/major/mismatch) |
comment (free text, cite the exact problematic phrase)
```

Rules:
- Work through items in the given order; do not skip (mark `broken` if unanswerable).
- Comments are mandatory whenever you deviate from the reference or flag ambiguity.
- Do not edit the item text; flag problems instead.
- Time budget: ~1–2 minutes per item; the full validation subset is designed for one
  2–3 hour session per annotator.

## Conflict resolution (after the blind round)

A facilitator (not an annotator) shows disagreeing annotators each other's codes
(anonymized) only for the disagreeing items. Each annotator may revise or defend.
Unresolved items go to adjudication: majority stands; ties are removed from the
accepted set and logged. All original and revised labels are retained in the release.

## Training

Before annotating the real subset, complete the 10-item calibration set
(`data/annotation_calibration/`) whose intended labels are published in
`calibration_answers.md`. Discuss discrepancies with the facilitator first.
