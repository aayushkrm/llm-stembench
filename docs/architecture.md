# Architecture Overview

## Module map

```
src/stembench/
├── schemas.py            # pydantic contracts: BenchmarkItem, MCItem,
│                         #   FreeResponseItem, RunConfig, ResponseRecord, Manifest
├── prompts.py            # versioned bilingual prompt templates (hashed into runs)
├── parsing.py            # answer extraction: MC letters (incl. Cyrillic), confidence,
│                         #   numeric (incl. RU decimal commas, ×10^n), units, exact
├── scoring.py            # MC / exact / numeric-tolerance correctness decisions
├── runner.py             # config-driven evaluation: stratified sampling, providers,
│                         #   checkpoint/resume, budget stop, manifest writing
├── report.py             # Stage 1 metrics/tables/figures from raw records
├── analysis_stage2.py    # paired bilingual analysis (pair-clustered bootstrap,
│                         #   H4/H5 language gaps, category breakdowns)
├── cli.py                # `stembench` entry point (run / report / models)
├── datasets/
│   ├── mmlu_stem.py      # cais/mmlu STEM subset, revision-pinned, stratified sample
│   └── stembench_ds.py   # built benchmark loader → eval items
├── providers/
│   ├── base.py           # Provider interface, ProviderError, DailyBudgetExceeded
│   ├── openai_compat.py  # OpenRouter/Zen: allowlists, shared budget+rate state,
│   │                     #   bounded retries, backoff
│   ├── ollama.py         # local (future local-model path)
│   ├── fake.py           # deterministic fake — tests only, stamped provider="fake"
│   └── registry.py       # provider construction + verified free-model lists
├── metrics/
│   ├── classification.py # accuracy, confusion, macro/micro/weighted P/R/F1
│   ├── calibration.py    # ECE, MCE, Brier (binary+multiclass), NLL, reliability bins
│   ├── intervals.py      # Wilson, cluster-aware paired bootstrap, difference CIs
│   ├── significance.py   # McNemar (exact+χ²), Cochran Q, χ², Holm, BH
│   └── agreement.py      # Cohen's κ (+SE, CI), Fleiss' κ (human validation stage)
├── benchmark_gen/        # original bilingual benchmark generators + QC + build
└── viz/figures.py        # all publication figures (headless Agg, Okabe-Ito palette)
```

## Data flow and invariants

1. **Config → runner**: a YAML `RunConfig` selects dataset, sample (seeded, stratified),
   models (provider + exact model ID + per-model item cap), decoding.
2. **Records**: every API response becomes exactly one `ResponseRecord` JSONL line —
   the single source of truth for all downstream numbers. Loaders exclude `fake__*`
   files and transient-error records from empirical analysis (error records are still
   reported as coverage/failure statistics).
3. **Resume**: records with `error_status in ("", "parse_failure")` count as done;
   `rate_limited`/`timeout`/`provider_error` are retried on the next run of the same
   config (same output dir), so no duplicate budget spend for completed work.
4. **Budgets**: per-provider daily counters in `data/cache/budget_<provider>.json`
   (date-keyed, thread-safe, shared across model workers). OpenRouter cap 50/day
   (documented free tier); Zen has no documented account cap — upstream per-model
   saturation is handled by bounded retries + per-model abort after 3 consecutive 429s.
5. **Zero spend**: `_assert_free` refuses any model not on the registry allowlist;
   `estimated_cost` from the provider is stored per record (OpenRouter returns cost 0).
6. **Benchmark build**: `benchmark_gen.build` regenerates the dataset from seed,
   re-verifies every answer independently, runs all QC gates, and writes a version
   file with the SHA-256 of `items.jsonl`; any gate failure exits nonzero with no
   partial dataset.

## Extension points

- New provider: subclass `OpenAICompatProvider` (or `Provider`), add to `registry.py`
  with an explicit free-model allowlist if it must remain zero-spend.
- New dataset: return `MCItem`/`FreeResponseItem` lists from a loader, register in
  `runner._load_items_for_run`.
- New metric: pure functions in `metrics/`, wired into `report.py`/`analysis_stage2.py`
  with unit tests against hand-computed fixtures.
