# risks.md — risk register

| ID | Risk | Category | Likelihood/Impact | Mitigation |
|----|------|----------|-------------------|------------|
| K1 | Free-tier rate limits (OpenRouter 50/day shared; Zen per-model saturation) starve experiments | Compute | high/high | Budget-aware runner with persistent counters; checkpoint/resume; Zen+OR capacity pooled; honest n reporting |
| K2 | API keys leak into repo/logs/artifacts | Security | low/high | Keys only in gitignored `.env`; providers read env; secret scan in CI; logs redact auth headers |
| K3 | Accidental paid-model call (Zen auto-reload risk) | Cost | low/high | Hard free-model allowlists in provider config; `_assert_free` before every call; cost field asserted 0 when present |
| K4 | Data contamination (MMLU in pretraining corpora) inflates scores | Validity | high/medium | Acknowledge in report; mitigation focus on original Stage 2 benchmark (procedural, post-cutoff parameters); cite contamination literature |
| K5 | Small n under free budgets → wide CIs, weak conclusions | Statistics | high/medium | Report CIs and effect sizes; paired designs for power; label pilot-scale findings; resume command extends n later |
| K6 | Parsing failures biasing accuracy downward | Measurement | medium/medium | Parse-failure rate reported per model; robust parser w/ Cyrillic; failure ≠ correct; sensitivity analysis scoring failures as incorrect |
| K7 | RU-EN template divergence breaking pair equivalence | Benchmark validity | medium/high | Parallel parameterized templates sharing computed answers; equivalence checks; translation review protocol in annotation package |
| K8 | LLM-assisted error annotation biases taxonomy distributions | Annotation | medium/medium | Explicit model-annotated labeling; published per-item rationales; human validation gate |
| K9 | Zen free models change/disappear mid-project | Compute | medium/medium | Record exact model IDs + timestamps in manifests; re-runs pinned by ID; failures recorded, not retried indefinitely |
| K10 | Benchmark generator bugs create wrong reference answers | Data quality | medium/high | Independent verifier recomputes answers via different code path; schema + answer-consistency gates fail the build on any mismatch |
| K11 | Publication overclaim (expert validation, venue, release status) | Integrity | low/high | audit.md reconciles claims; release labeled candidate v0.1.0; paper states draft status |
| K12 | Determinism gaps across providers (temperature support varies) | Reproducibility | medium/low | Decoding settings recorded per record; temperature 0 requested; deviations documented per provider |
| K13 | Procedural variants share templates, reducing effective sample size and leaking structure across dev/test | Statistics/validity | high/medium | Stable template IDs; number-masked duplication report; dev limited to format checks; template-cluster sensitivity CIs; future template-held-out expert release |
| K14 | Generator difficulty labels overstate actual challenge level | Construct validity | medium/high | Replaced rejected challenge families; hard concept/feature metadata gate; stratified agent review; labels remain provisional until genuine subject-expert review |
