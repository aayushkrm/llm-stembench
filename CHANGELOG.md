# Changelog

All notable changes to LLM-STEMBench. Format follows Keep a Changelog; versioning is
semantic. Dataset releases are versioned separately (see data/stembench_v1/DATASET_VERSION).

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
