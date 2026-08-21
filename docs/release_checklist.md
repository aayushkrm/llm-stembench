# Release Checklist

## Code + data repository (GitHub: aayushkrm/llm-stembench)

- [x] `pytest tests/ -q` green from clean checkout (205 tests; CI 3.10 + 3.12)
- [x] `ruff check src tests scripts` clean
- [x] CI workflow green (includes offline e2e + benchmark determinism + secret scan;
      latest runs 32525859748 and later)
- [x] `git status` clean; no secrets in tracked files (`.env` untracked; CI secret
      scan passing)
- [x] Raw run records committed (`results/**/!(*fake__*)`), manifests complete
      (S2-E1 carries rescore_manifest.json with per-record provenance)
- [x] `goal.md` and `audit.md` reconciled; every claim in README/reports/paper has an
      artifact path
- [x] Version tagged (annotated tag) matching CHANGELOG (`v0.2.0`)

## Dataset release (HuggingFace Datasets) — READY-RUN-ONCE-PUBLISHED state

Preconditions (current status):
- [x] deterministic build with seed + SHA-256 (`data/stembench_v1/DATASET_VERSION`)
- [x] all QC gates pass; independent answer verification 100%
- [x] dataset card with license (CC-BY-4.0 for data), provenance, limitations
- [ ] **human expert validation (release gate for v1.0)** — PENDING; release stays
      `v0.1.0-candidate` and the card says so explicitly
- [ ] HF token with write scope provided (external action)
- [ ] `scripts/publish_hf.py` run once (creates repo, uploads items + card + version);
      records the published URL + commit SHA in `releases/RELEASE_NOTES.md`

## Paper

- [x] draft complete (abstract … conclusions, references from verified bibliography)
- [x] Russian abstract synchronized
- [ ] venue formatting/submission — NOT done (requires explicit external authorization;
      no submission is claimed anywhere)

## Honesty gates (checked at audit)

- [x] no synthetic/fake records in any empirical table (loaders exclude `fake__*`)
- [x] model-annotated labels never described as human (annotator field verbatim in
      every annotation record)
- [x] pre-release dataset version labeled candidate; no human κ reported
- [x] `audit.md` states final status (PARTIAL) with blockers and the smallest
      external action for each
