# Error Taxonomy for Exact-Science LLM Responses

Version 1.0 (2026-08-17). Grounded in the literature review (docs/literature): error
categories follow and extend taxonomies from GSM8K/MATH error analyses, process-supervision
work (Lightman et al. 2023), and hallucination benchmarks (TruthfulQA, HaluEval).

## Design principles

- **Multi-label**: one incorrect response may carry several labels (e.g., a unit error
  that also makes the arithmetic wrong). Primary label = earliest failing stage.
- **Applicable to MC and free-response**: extraction failures are errors of the response
  format, not the model's knowledge, and are tracked separately from knowledge errors.
- **Adjudication rule**: label from the response text alone (plus the item and gold
  answer). If evidence is insufficient for a category, use `not_classifiable`.
- **Mutual-exclusivity is at the primary-label level only.**

## Categories

| Code | Category | Definition | Typical evidence |
|---|---|---|---|
| E1 | Knowledge/concept | Misuse or absence of a domain fact, law, or definition | wrong formula named; wrong reagent behavior; misstated law |
| E2 | Reasoning/logic | Correct facts, invalid inference chain | non-sequitur step; circular reasoning; wrong implication direction |
| E3 | Arithmetic/algebra | Domain logic right; a computation/algebra slip | 7×8=54; sign slip when moving terms; wrong root selection |
| E4 | Unit/dimension | Quantity right in wrong units, or unit conversion error | cm vs m; kJ vs J; mol vs mmol; dimension-inconsistent line |
| E5 | Problem-comprehension | Answers a different question than asked | solves for t when asked v; ignores a stated constraint |
| E6 | Answer-extraction/format | Substantive answer plausibly present but not in the required contract (letter/value missing or ambiguous) | no `Answer:` line; two letters; blank answer |
| E7 | Language/translation | Error plausibly induced by language form (RU item or RU output) rather than content | misreads RU phrasing; answers in wrong language with drifted meaning |
| E8 | Ambiguity/reference defect | The item itself admits another defensible answer | flag rather than blame the model; counted separately |
| E9 | Hallucinated assumption | Invents unstated facts to proceed | assumes a value never given; invents a reaction condition |
| E10 | Safety/reliability failure | Refusal, meta-answer, or content-free response | "I cannot answer"; restates the question; empty output |
| E0 | not_classifiable | Incorrect with insufficient trace (e.g., bare wrong letter on MC with no reasoning shown) | none |

## Annotation protocol

1. **Population**: all incorrect responses (correctness = False) from the Stage 1 run
   across models; sample drawn by seeded stratified selection (by model × subject) —
   or the full set if below the 100–200 target.
2. **Unit**: one (model, item) response record.
3. **Labels per record**: `labels: [codes...]` + `primary: code` + `evidence_quote`
   (verbatim span from the response) + `note`.
4. **Annotator**: **AI agent (ZCode/GLM) — model-annotated, NOT a human annotation.**
   Every artifact and report states this. Human expert validation remains a release
   gate (goal.md R7.1, audit.md).
5. **Agreement**: because there is a single (non-human) annotator, no inter-annotator
   agreement statistic is reported for Stage 1 annotations. The multi-rater machinery
   (Cohen's κ, Fleiss' κ) is implemented, unit-tested, and shipped for the human
   validation stage of the benchmark.
6. **Adjudication order** for the primary label:
   E6 (no usable answer) > E10 (refusal/empty) > E8 (item defect, if known) >
   E5 (wrong question) > E1 (fact) > E9 (invented fact) > E2 (logic) > E3 (arithmetic)
   > E4 (units) > E7 (language-induced) > E0.
7. **Storage**: `results/stage1/error_analysis/annotations.jsonl` — one line per
   annotated response with raw response, item, gold, labels, primary, evidence quote,
   annotator identity, and timestamp. Distribution tables/figures regenerate from it.

## Reporting

Distribution of primary labels with Wilson 95% CIs (n = annotated incorrect responses),
plus multi-label incidence. All claims labeled exploratory (single model-annotator).
