# Changelog

All notable changes to LLM-STEMBench. Format follows Keep a Changelog; versioning is
semantic. Dataset releases are versioned separately (see data/stembench_v1/DATASET_VERSION).

## [0.2.0] — 2026-08-22

### Added
- Accepted STEMBench build (624 pairs / 1,248 records; 1,516/1,516 independent answer
  verifications; byte-identical deterministic rebuilds, CI-enforced) with dataset card.
- Run S2-E1: 520 free-tier evaluations, 8 models (5 complete at n=100 incl.
  stealth/ox-alpha at max reasoning, decisions.md D11); pair-clustered bilingual
  analysis (H4/H5) with template-cluster sensitivity, category breakdowns, and
  five figures; `stembench analyze-stage2` CLI.
- Error annotation extended to all 89 real Stage 2 errors (D13); combined annotated
  pool 160 real errors — the 100–200 contract target is met.
- Reports: Stage 2 final report; paper §6.4 + EN/RU abstracts with results.
- Runner records per-record scoring metadata (`extra`), enabling standalone rescoring.

### Fixed
- Manifest `n_total_evaluated` counted retryable error records as evaluated (D12);
  runner fixed, all manifests recomputed from raw records.
- Numeric unit scoring extracted units from the whole response, failing numerically
  correct answers (D14); now answer-scoped with unit-equivalence normalization
  (M ≡ mol/L, separator/exponent forms, RU aliases). S2-E1 rescored: 19 documented
  False→True flips, 0 regressions, per-record provenance preserved.
- Parser regressions (NUM_RE scientific notation, unit-in-word boundaries, bold MC
  letters, stratified-sampling bound); pytest pythonpath for bare-pytest CI runs.

### Changed
- Corrected S2-E1 standings after D14: ox-alpha 91 (leads), lightning 85, hy3 82,
  ultra 79, laguna 78; H4/H5 language-gap nulls unchanged.

## [0.1.0] — 2026-08-17

### Added
- Evaluation pipeline: provider-agnostic adapters (OpenRouter, Opencode Zen, Ollama,
  deterministic fake), budget-aware runner with checkpoint/resume, JSONL records with
  full provenance, prompt templates with per-run hashes.
- Metrics: accuracy/exact-match, macro/micro/weighted P/R/F1, confusion matrices,
  ECE/MCE/Brier/NLL with reliability bins, Wilson intervals, cluster-aware paired
  bootstrap, McNemar (exact + χ²), Cochran's Q, Pearson χ², Holm and BH corrections,
  Cohen's κ and Fleiss' κ.
- Parsing/scoring: Cyrillic-robust MC letter extraction, self-reported confidence,
  exact match with alternatives, numeric tolerance with units.
- STEMBench bilingual benchmark generators (math/physics/chemistry) with independent
  answer verification, QC gates, deterministic versioned builds.
- Stage 1 pilot on MMLU-STEM (six free-tier models) with error taxonomy and
  model-assisted error annotation (labeled as such; human validation is a release gate).
- Stage 2 evaluation and paired bilingual analysis on the original benchmark.
- Documentation: literature review (33 verified sources), benchmark specification and
  card, bilingual annotator guidelines and workflow, dataset selection memo,
  preregistered hypothesis registry, reproducibility guide, paper draft.
- CI: lint, tests, offline e2e, benchmark determinism check, secret scan.

### Known limitations
- All model runs use FREE-TIER endpoints only; sample sizes are budget-constrained
  (per-model n reported everywhere; CIs are wide).
- Benchmark is a v0.1.0-candidate: automated verification complete; independent human
  expert validation pending (release gate).
- MMLU contamination risk acknowledged for Stage 1; Stage 2 benchmark is original and
  procedurally generated (2026-08-17 parameters).
