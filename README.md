# LLM-STEMBench

Provider-agnostic LLM evaluation pipeline + **STEMBench**, an original bilingual
(Russian–English) mathematics/physics/chemistry benchmark with procedurally generated,
independently verified answers.

> **Status honest-by-design.** All model results in this repository come from
> **free-tier** endpoints under a zero-paid-spend policy; every table states its exact
> model IDs and sample sizes with confidence intervals. The benchmark is
> **v0.1.0-candidate**: automated verification is complete; independent human expert
> validation is a pending release gate (see `docs/annotation/`).

## What's inside

The Git repository is this `llm-stembench/` directory. Its parent `LLM-Bench/`
directory is only a local Codex workspace holding the authoritative project brief and
transfer notes; it is not a second checkout or a second project. Workspace-wide
instructions live in `../AGENTS.md`, while this repository's specific commands and
architecture rules live in `AGENTS.md`.

| Path | Contents |
|---|---|
| `src/stembench/` | The `stembench` Python package (CLI included) |
| `data/stembench_v1/` | Built benchmark (items, verification report, version hash) |
| `results/stage1/`, `results/stage2/` | Raw run records (JSONL), manifests, analyses, figures |
| `docs/` | Literature review (33 verified sources), benchmark spec/card, annotation package, hypothesis registry, reproducibility guide |
| `reports/` | Stage 1 course-project report, Stage 2 final report |
| `paper/` | English paper draft + Russian abstract |
| `tests/` | Unit/property/contract/integration test suite |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # add free-tier API keys (or skip: dry-run needs none)

# offline end-to-end smoke (deterministic fake provider — synthetic records)
stembench run --config configs/stage1_pilot.yaml --dry-run

# real Stage 1 pilot (MMLU-STEM; free-tier models only; budget-aware)
stembench run --config configs/stage1_pilot.yaml

# metrics, tables, figures from a finished run
stembench report --run results/stage1/S1-P1

# rebuild the bilingual benchmark deterministically + QC gates
python -m stembench.benchmark_gen.build --out data/stembench_v1 --seed 20260817

# tests
pytest tests/ -q
```

## Pipeline architecture

```
datasets ──► runner ──► providers ──► parsing/scoring ──► metrics ──► report/viz
(mmlu_stem,   (config,   (openrouter,   (Cyrillic-robust    (accuracy,    (tables,
 stembench)    sample,     zen, ollama,   MC/exact/           calibration,  figures,
               resume,     fake[tests])  numeric)            paired tests)  manifests)
               budgets)
```

- **Providers** implement one interface; only models on hard-coded **free allowlists**
  can ever be called (zero-spend enforced in code, `providers/registry.py`).
- **Runner** records every request/response as a schema-validated JSONL line with full
  provenance (prompt hash, decoding, latency, tokens, cost, git commit, run ID) and
  resumes without duplicate calls (`runner.py`).
- **Statistics** follow the preregistered plan (`docs/hypotheses.md`): Wilson CIs,
  McNemar/Cochran's Q for paired models, cluster (pair-level) bootstrap for RU–EN
  gaps, Holm/BH multiplicity correction, Cohen's/Fleiss' κ for the annotation stage.

## Repository conventions

- Never edit numbers in `reports/` or `paper/` by hand — regenerate them
  (`stembench report`, `stembench analyze-stage2`).
- Synthetic/fake-provider artifacts are named `fake__*` and excluded from empirical
  results by the loaders.
- Secrets live only in `.env` (gitignored); CI scans commits for key-shaped strings.

See `docs/architecture.md` for the module map and `docs/reproducibility.md` for exact
reproduction commands of every published number.
