# Annotation Format

Two equivalent formats are accepted. Fields are identical in both.

## JSONL (preferred)

One line per (item, rater, round):

```json
{
  "item_id": "MATH-0042-ru",
  "rater_id": "R2",
  "round": 1,
  "correct": "valid",
  "unambig": "unambiguous",
  "difficulty": "university",
  "language": "ok",
  "comment": "",
  "annotated_at": "2026-09-01T14:20:00+00:00"
}
```

Allowed values:
- `correct`: `valid` | `invalid`
- `unambig`: `unambiguous` | `ambiguous` | `broken`
- `difficulty`: `school` | `university` | `olympiad`
- `language`: `ok` | `minor` | `major` | `mismatch`

`round` ≥ 2 only via facilitator-assisted revision; round-1 rows are never deleted.

## CSV template

`annotation_template.csv` with the same columns; UTF-8, comma-separated, quoted free
text. Export from any spreadsheet tool.

## Aggregation

`scripts/aggregate_annotations.py` (in the release bundle) computes:
- per-item label matrices (kept in the release as `annotations.jsonl`, concatenated);
- Fleiss' κ over `correct` (primary gate), pairwise Cohen's κ as supplementary;
- majority label per item and the accepted/rejected partition for the dataset build;
- a full revision history table (`adjudication_log.csv`).

## Machine-checkable invariants

- every `item_id` exists in the dataset version under validation;
- every rater annotated each assigned item exactly once per round;
- value domains exactly as above (validated by `stembench` schema).
