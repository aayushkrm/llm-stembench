# LLM-STEMBench repository guide

## Scope and precedence

This file applies specifically inside the `llm-stembench` Git repository. It extends
the parent workspace instructions in `../AGENTS.md` with live repository details. Both
files apply here; this nearer file takes precedence if a rule ever conflicts. Keep
workspace-wide transfer/contract guidance in the parent file and implementation
commands, architecture, and gotchas here.

## Purpose and durable state

This Python repository implements a provider-agnostic LLM evaluation pipeline, a
procedurally generated bilingual Russian–English STEM benchmark, Stage 1/2 analyses,
and publication artifacts. Keep these files synchronized at real milestones:
`goal.md` (traceability), `progress.md` (commands/results), `decisions.md` (material
design choices), `risks.md`, and `audit.md`.

Major directories:

- `src/stembench/`: package, CLI, schemas, runner, providers, parsing/scoring,
  metrics, reports, visualization, and Stage 2 paired analysis.
- `src/stembench/benchmark_gen/`: deterministic generators, independent verifier,
  QC gates, and atomic dataset build.
- `configs/`: versioned Stage 1/2 run configurations.
- `tests/`: offline unit, property, provider-contract, integration, and report tests.
- `data/stembench_v1/`: generated candidate dataset and verification/hash artifacts.
- `results/`: real raw model records, manifests, regenerated analyses, and figures.
- `docs/`, `reports/`, `paper/`: research design, cards/guidelines, reports, and draft.
- `scripts/`: auditable maintenance, annotation, validation, and release helpers.

## Environment and commands

Use the existing Python 3.12 editable environment; do not create a second environment:

```bash
.venv/bin/python --version
.venv/bin/python -m pytest tests/ -q --cov=stembench --cov-report=term-missing
.venv/bin/ruff check src tests scripts
MPLBACKEND=Agg .venv/bin/stembench report --run results/stage1/S1-P1
.venv/bin/python -m stembench.benchmark_gen.build --out data/stembench_v1 --seed 20260817
```

For isolated budget-aware tests set `STEMBENCH_BUDGET_DIR` to a task-specific `/tmp`
directory. Live runs require the gitignored `.env`; load it without printing values.
Never echo environment variables. The runner and provider allowlists must continue to
refuse paid model IDs.

There is no configured static type checker yet. Do not claim typecheck success unless
one is added and run. CI currently covers Python 3.10/3.12, ruff, pytest+coverage, an
offline fake-provider smoke, deterministic benchmark builds, and a key-pattern scan.

## Architecture boundaries

- `schemas.py` owns validated cross-layer contracts. Change schemas deliberately and
  update loaders, runner, artifacts, and tests together.
- Dataset modules load/convert items; they must not call providers.
- Provider adapters return normalized `Completion` objects. Only the registry enforces
  provider construction and the hard free-model allowlists.
- `runner.py` owns sampling, prompts, checkpoint/resume, and append-only raw records.
  Resume must never duplicate successful or parse-failure calls.
- `parsing.py` extracts auditable values; `scoring.py` applies task semantics. Parser
  changes require regression tests and an offline rescore manifest for existing data.
- Metric functions stay pure and are checked against hand or trusted-library fixtures.
  Use paired tests for shared items and pair-clustered methods for bilingual variants.
- Reports and figures regenerate from raw artifacts. Never silently hand-edit a numeric
  claim without reconciling it to the generated JSON/CSV source.
- Benchmark generators compute candidates; `verify.py` independently recomputes
  answers; `qc.py` owns hard gates; `build.py` may emit data only after all gates pass.
  Never weaken a valid gate merely to make the build green.

## Coding and artifact conventions

- Python style is Ruff (`E,F,I,UP,B`), 100-column target, Python 3.10 compatibility.
- Prefer typed boundaries, deterministic seeds, `pathlib`, UTF-8, and atomic writes for
  replaceable artifacts. Preserve raw provider output unchanged.
- Use `apply_patch` for hand edits. Generated datasets/reports may be rewritten only by
  their declared reproducibility commands.
- Exclude `fake__*` from all empirical loaders and tables. Report parse/failure rates;
  never silently drop invalid outputs.
- Record exact model/provider IDs, sample sizes, costs, hypotheses, null findings,
  parser rescoring, and material generator fixes in the tracking files.
- Inspect a stratified RU/EN dataset sample manually after automated QC. Clearly label
  that review as agent/model review, not expert or human validation.
- Subagents may handle bounded, non-overlapping work. Inspect their diffs and rerun
  their validation locally before relying on them.
- Commit as the configured user only. Never add AI/assistant authorship, co-author
  trailers, sign-offs, acknowledgements, or generated-by metadata. Use a concise
  standard subject with an optional body.

## Known gotchas

- The working tree may contain legitimate in-flight artifacts; preserve them and use
  `git diff`/`git status` before every milestone commit.
- OpenRouter free-tier capacity is account-wide and daily; Zen saturation is transient.
  Resume existing runs rather than creating duplicate experiment IDs.
- Provider errors and parse failures have different retry semantics.
- Stage 2 sampling must preserve complete RU/EN pairs and analyze language gaps at the
  `pair_id` cluster level.
- `data/cache/`, `.env`, coverage/cache files, and synthetic output are not release
  artifacts. Verify ignore rules and run a secret scan before commit/push.
- Human expert validation, Hugging Face upload credentials, and venue submission are
  external gates; code readiness does not clear them.

# Remember to use skills and plugin for your work and tasks whenever any task needed or if using skills or plugin would help with current task like doing it better or improving it or when fixing any problem etc which is available to use in your coding agent (zcode) from -
/Users/akm/.zcode/skills
/Users/akm/.zcode/cli/plugins