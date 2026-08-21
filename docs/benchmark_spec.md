# STEMBench v1 — Benchmark Specification

**Status: candidate (v0.1.0-candidate). Human expert validation pending — see
benchmark card.**

## 1. Purpose and intended use

An original bilingual (Russian/English) benchmark of short, exactly-scored problems in
mathematics, physics, and chemistry for evaluating and comparing LLM capability and
calibration on exact-science tasks, including **paired within-item language effects**
(the same semantic item in RU and EN).

**Intended uses**
- evaluating LLM accuracy/calibration on school/university/olympiad STEM problems;
- studying RU–EN performance gaps under controlled content (identical numbers, answers,
  semantics; parallel surface forms);
- regression-testing models across versions; research on error taxonomies.

**Out-of-scope uses**
- high-stakes decisions about individuals (admissions, hiring);
- training-data claims beyond the license (CC-BY-4.0 attribution required);
- general-knowledge or open-ended reasoning evaluation (all items have short verified
  answers by design).

## 2. Composition

- Subjects: mathematics, physics, chemistry.
- Languages: Russian and English, **semantically aligned pairs** (`pair_id` links the
  two variants; identical computed answer, tolerance, units, difficulty).
- Difficulty: school / university / olympiad, each item tagged with a rubric note.
- Answer formats: `mc` (4 choices, 1 correct), `numeric` (canonical decimal + tolerance
  + units), `exact` (language-neutral short canonical string + alternatives).
- Target size: ≥600 raw semantic pairs (≥1,200 language records); ≥500 accepted pairs
  after QC. Achieved counts are published in the build's verification report and
  dataset card — never hardcoded here.
- Splits: `test` (evaluation) and a small `dev` set (few-shot/format demonstration);
  no `train` split (evaluation benchmark).

## 3. Item schema (summary — authoritative version: `src/stembench/schemas.py: BenchmarkItem`)

item_id, pair_id, language(ru|en), subject, topic, stable template_id,
difficulty(+rubric evidence), question, answer_type, choices[label,text] (MC), canonical_answer,
acceptable_alternatives, tolerance{rel,abs}, units, solution, provenance, license,
author, creation_method, translator, verifier[VerifierRecord{method,passed,detail}],
annotation_version, quality_flags, split, contamination_notes.

## 4. Creation method and provenance

All items are **original procedural generations** (D7): parameterized bilingual
templates authored for this project; parameters drawn deterministically per (topic,
index, global seed); answers computed in code and **independently re-verified by a
second code path** (alternative formula arrangement, back-substitution, or dimensional
check). No item is copied from any textbook, exam, or existing dataset; no LLM
generated item text. Parameters and templates date from 2026-08-17 (post-dating most
current training cutoffs, reducing — not eliminating — contamination risk).

## 5. Quality gates (enforced by `stembench.benchmark_gen.build` — build fails otherwise)

1. Schema validity of every record.
2. Pair completeness: every pair_id has exactly one `ru` and one `en` record with
   identical answer content (canonical answer, tolerance, units, type) and matched
   choices.
3. Answer/solution consistency: verifier recompute passes for 100% of items.
4. Numeric answers: parseable, magnitude bounds, tolerance set.
5. MC: distractors distinct from canonical (beyond tolerance for numerics) and
   textually unique; near-uniform correct-letter distribution across A–D.
6. Dedup: unique normalized question hash per language; cross-pair word-3-gram Jaccard
   similarity below threshold within subject+language (max reported). A separate
   number-masked structural report quantifies repeated procedural templates and
   dev/test template overlap; changed numbers are not treated as conceptual independence.
7. Distribution report by subject × difficulty × answer_type × language.
8. Determinism: same seed → identical dataset (byte-identical items.jsonl); version
   file with SHA-256 of items.jsonl.

## 6. Language policy

Parallel templates share parameters and computed answers; Russian uses natural STEM
Russian terminology; names are transliterated pairs (Anna/Анна); units remain SI symbols
in both languages; decimal points in both; the `Answer:`/`Confidence:` output contract
is identical in both languages (labels in English in the RU prompt too, to keep the
extraction contract constant).

## 7. Difficulty rubric

- **school**: single concept, one or two arithmetic steps, standard curriculum
  (grades 7–9).
- **university**: multi-step reasoning or non-trivial formula use; typical first-year
  coursework.
- **olympiad/challenge**: requires combining ≥2 concepts plus a concrete constraint,
  edge condition, or uncommon insight. Every such generator declares its concepts and
  challenge feature, and the build rejects missing declarations. This metadata supports
  auditing but does not establish expert-rated difficulty; that remains a human release
  gate.
Each item's `difficulty_rubric` states the justification in one sentence.

## 8. Scoring

- MC: exact letter match (robust parser handles Cyrillic letters and common formats);
  unparseable → parse failure (counted, excluded from strict accuracy, included in
  lenient accuracy as incorrect).
- Numeric: tolerance-based equality (rel 2% default or abs tolerance), unit checked
  when declared and present in the response.
- Exact: normalized string equality (case/whitespace/punctuation-insensitive) or
  listed alternative.

## 9. Evaluation protocol

Same prompt contract (`mc_answer_confidence_v1` / `free_answer_confidence_v1`),
temperature 0, per-model decoding recorded; identical item set and order for every
model; RU and EN variants delivered as separate items; paired statistics cluster on
`pair_id` (hypotheses H4/H5).

## 10. Versioning and maintenance

Version `v0.1.0-candidate` until human expert validation (release gate). Breaking
changes bump the minor version; parameter/template fixes regenerate deterministically
from seed; changelog records every change; dataset hash published with each release.
