"""Evaluation runner: items x models with checkpointing, budgets, retries.

Guarantees:
- resume: (run_id, model, item_id) records already on disk are never re-requested, so a
  resumed run makes no duplicate API calls;
- zero-spend: providers refuse non-free models (see providers.registry);
- atomic-ish writes: records append+flush per item; manifest rewritten at end.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from stembench import prompts
from stembench.parsing import extract_confidence
from stembench.providers.base import Completion, DailyBudgetExceeded, ProviderError
from stembench.providers.registry import build_provider
from stembench.schemas import (
    MCItem,
    ModelSpec,
    ResponseRecord,
    RunConfig,
    Usage,
    utcnow,
)
from stembench.scoring import score_exact, score_mc, score_numeric


def git_state(repo: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True,
                check=True,
            ).stdout.strip()
        )
        return commit, dirty
    except subprocess.CalledProcessError:
        return "", False


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:80]


def _load_items_for_run(config: RunConfig) -> tuple[list[Any], dict[str, Any]]:
    """Build the model-facing item list for this run."""
    if config.dataset == "mmlu_stem":
        from stembench.datasets.mmlu_stem import load_mmlu_stem, stratified_sample

        all_items, snapshot = load_mmlu_stem(split=config.sample.split)
        items = stratified_sample(all_items, config.sample.n, config.sample.seed,
                                  by=config.sample.stratify_by)
        return items, snapshot
    if config.dataset == "stembench_v1":
        from stembench.datasets.stembench_ds import load_stembench, to_eval_items

        bench_items, stats = load_stembench(split=config.sample.split)
        eval_items = to_eval_items(bench_items)
        # sample at the PAIR level so RU/EN variants stay together (paired design)
        import random as _r

        rng = _r.Random(config.sample.seed)
        by_pair: dict[str, list[Any]] = {}
        by_subject: dict[str, list[str]] = {}
        for it in eval_items:
            pid = it.item_id.rsplit("-", 1)[0]
            by_pair.setdefault(pid, []).append(it)
            by_subject.setdefault(str(it.subject), []).append(pid)
        target_pairs = config.sample.n  # n = number of PAIRS for this dataset
        picked_pids: list[str] = []
        pools: dict[str, list[str]] = {}
        for subject, pids in sorted(by_subject.items()):
            pool = sorted(set(pids))
            rng.shuffle(pool)
            pools[subject] = pool
        target_pairs = min(target_pairs, sum(len(pool) for pool in pools.values()))
        while len(picked_pids) < target_pairs:
            progressed = False
            for subject in sorted(pools):
                if pools[subject] and len(picked_pids) < target_pairs:
                    picked_pids.append(pools[subject].pop())
                    progressed = True
            if not progressed:  # defensive: all subject pools exhausted
                break
        eval_items = [it for pid in picked_pids for it in by_pair[pid]]
        eval_items.sort(key=lambda it: it.item_id)
        stats["n_pairs_sampled"] = len(picked_pids)
        return eval_items, stats
    raise ValueError(f"unknown dataset {config.dataset}")


def _build_prompt(item: Any) -> str:
    lang = item.language.value if hasattr(item.language, "value") else str(item.language)
    if isinstance(item, MCItem):
        return prompts.render_mc(item.question, item.choices, language=lang)
    return prompts.render_free(item.question, language=lang)


def _score_item(item: Any, raw_text: str) -> tuple[bool | None, str | None, str]:
    if isinstance(item, MCItem):
        ok, letter, method = score_mc(raw_text, item.gold, n_choices=len(item.choices))
        return ok, letter, method
    if item.answer_type.value == "numeric":
        ok, raw, method = score_numeric(
            raw_text,
            float(item.gold),
            rel_tol=item.tolerance.rel if item.tolerance else None,
            abs_tol=item.tolerance.abs if item.tolerance else None,
            require_unit=item.units,
        )
        return ok, raw, method
    ok, parsed, method = score_exact(raw_text, item.gold, item.alternatives)
    return ok, parsed, method


def _existing_keys(records_path: Path) -> set[str]:
    """Items considered done: successful responses and parse failures (deterministic
    at temperature 0). Transient errors (rate_limited/timeout/provider_error) are
    retried on resume."""
    done: set[str] = set()
    if records_path.exists():
        with open(records_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("error_status", "") in ("", "parse_failure"):
                        done.add(rec["item_id"])
                except json.JSONDecodeError:
                    continue
    return done


def _run_one_model(
    config: RunConfig,
    model_spec: ModelSpec,
    items: list[Any],
    out_dir: Path,
    commit: str,
) -> dict[str, Any]:
    provider = build_provider("fake" if config.dry_run else model_spec.provider)
    records_path = out_dir / f"{_safe_name(model_spec.provider)}__{_safe_name(model_spec.model)}.jsonl"
    done = _existing_keys(records_path)
    todo = [it for it in items if it.item_id not in done][: model_spec.n_items or None]
    counts = {
        "model": model_spec.model,
        "provider": model_spec.provider,
        "n_items_configured": len(items),
        "n_already_done": len(done),
        "n_new_attempted": 0,
        "n_new_completed": 0,
        "n_correct": 0,
        "n_parse_failures": 0,
        "n_errors": 0,
        "status": "complete",
        "error_kinds": {},
    }
    fh = open(records_path, "a", encoding="utf-8")
    consecutive_rate_limited = 0
    try:
        for item in todo:
            counts["n_new_attempted"] += 1
            prompt_text = _build_prompt(item)
            rec = ResponseRecord(
                run_id=config.run_id,
                item_id=item.item_id,
                dataset=config.dataset,
                dataset_revision=config.dataset_revision,
                subject=str(item.subject),
                template_id=item.template_id,
                language=item.language.value if hasattr(item.language, "value") else str(item.language),
                difficulty=str(item.difficulty),
                answer_type="mc" if isinstance(item, MCItem) else item.answer_type.value,
                provider="fake" if config.dry_run else model_spec.provider,
                model=model_spec.model,
                decoding=model_spec.decoding.model_dump(),
                prompt_hash=prompts.template_hash(
                    config.prompt_template if isinstance(item, MCItem)
                    else "free_answer_confidence_v1",
                    item.language.value if hasattr(item.language, "value") else "en",
                ),
                prompt_text=prompt_text,
                started_at=utcnow(),
                code_commit=commit,
            )
            try:
                comp: Completion = provider.complete(
                    model=model_spec.model,
                    messages=[{"role": "user", "content": prompt_text}],
                    max_tokens=model_spec.decoding.max_tokens,
                    temperature=model_spec.decoding.temperature,
                    top_p=model_spec.decoding.top_p,
                    reasoning_effort=model_spec.decoding.reasoning_effort,
                )
            except DailyBudgetExceeded:
                rec.error_status = "daily_budget_exceeded"
                rec.finished_at = utcnow()
                fh.write(rec.model_dump_json() + "\n")
                fh.flush()
                counts["status"] = "partial_daily_budget"
                counts["error_kinds"]["daily_budget_exceeded"] = (
                    counts["error_kinds"].get("daily_budget_exceeded", 0) + 1
                )
                break
            except ProviderError as e:
                rec.error_status = _classify_error(str(e))
                rec.finished_at = utcnow()
                fh.write(rec.model_dump_json() + "\n")
                fh.flush()
                counts["n_errors"] += 1
                kind = rec.error_status
                counts["error_kinds"][kind] = counts["error_kinds"].get(kind, 0) + 1
                if kind == "rate_limited":
                    consecutive_rate_limited += 1
                    if consecutive_rate_limited >= 3:
                        counts["status"] = "partial_rate_limited"
                        break
                    time.sleep(60)
                else:
                    consecutive_rate_limited = 0
                continue
            rec.finished_at = utcnow()
            rec.latency_ms = None
            rec.raw_response = comp.content
            consecutive_rate_limited = 0
            rec.finish_reason = comp.finish_reason
            rec.usage = Usage(**comp.usage) if comp.usage else None
            rec.model_reported = comp.model_reported
            rec.estimated_cost = comp.cost
            if comp.logprobs:
                rec.logprobs_raw = comp.logprobs[:64]
                if "token_logprob" not in rec.confidence_provenance:
                    rec.confidence_provenance.append("token_logprob_available")
            ok, parsed, method = _score_item(item, comp.content)
            rec.parsed_answer = parsed
            rec.parse_method = method
            rec.correctness = ok
            conf = extract_confidence(comp.content)
            if conf is not None:
                rec.self_reported_confidence = conf
                rec.confidence_provenance.append("self_report")
            rec.reference_answer = (
                chr(ord("A") + item.gold) if isinstance(item, MCItem) else item.gold
            )
            if ok is None:
                rec.error_status = "parse_failure"
                counts["n_parse_failures"] += 1
            elif ok:
                counts["n_correct"] += 1
            counts["n_new_completed"] += 1
            fh.write(rec.model_dump_json() + "\n")
            fh.flush()
    finally:
        fh.close()
        if hasattr(provider, "close"):
            provider.close()
    # recompute totals over ALL records for this model (incl. resumed)
    total_done = 0
    total_correct = 0
    total_parse_fail = 0
    with open(records_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("error_status") in (
                "daily_budget_exceeded", "rate_limited", "timeout", "provider_error",
            ):
                continue
            total_done += 1
            if r.get("correctness") is True:
                total_correct += 1
            if r.get("correctness") is None and r.get("error_status") == "parse_failure":
                total_parse_fail += 1
    counts["n_total_evaluated"] = total_done
    counts["n_total_correct"] = total_correct
    counts["n_total_parse_failures"] = total_parse_fail
    if todo and counts["n_new_attempted"] < len(todo):
        counts["status"] = "partial_" + (
            "daily_budget" if counts["status"] != "complete" else "interrupted"
        )
    return counts


def _classify_error(msg: str) -> str:
    m = msg.lower()
    if "429" in m or "rate" in m:
        return "rate_limited"
    if "timeout" in m or "timed out" in m:
        return "timeout"
    return "provider_error"


def run(config: RunConfig, repo_root: Path | None = None) -> dict[str, Any]:
    repo_root = repo_root or Path.cwd()
    out_dir = Path(config.output_dir) / f"stage{config.stage}" / config.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    commit, dirty = git_state(repo_root)

    items, snapshot = _load_items_for_run(config)

    from stembench.schemas import Manifest

    manifest = Manifest(
        run_id=config.run_id,
        stage=config.stage,
        dataset=config.dataset,
        dataset_revision=snapshot.get("revision", ""),
        dataset_snapshot=snapshot,
        sample=config.sample.model_dump(),
        prompt_template=config.prompt_template,
        prompt_template_text=prompts.TEMPLATES[config.prompt_template]["en"],
        prompt_template_hash=prompts.template_hash(config.prompt_template, "en"),
        started_at=utcnow(),
        git_commit=commit,
        code_dirty=dirty,
        notes=config.notes,
    )
    (out_dir / "manifest_start.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )

    # one worker per model: each model is sequential; per-provider pacing and daily
    # budgets are shared+thread-safe, so provider limits hold across model workers
    model_results: list[dict[str, Any]] = [None] * len(config.models)  # type: ignore
    with ThreadPoolExecutor(max_workers=max(1, len(config.models))) as ex:
        futs = {
            ex.submit(_run_one_model, config, ms, items, out_dir, commit): i
            for i, ms in enumerate(config.models)
        }
        for fut, i in futs.items():
            model_results[i] = fut.result()  # noqa: BLE001

    manifest.models = model_results  # type: ignore[assignment]
    manifest.counts = {
        "n_items": len(items),
        "n_models": len(config.models),
        "total_evaluated": sum(m["n_total_evaluated"] for m in model_results),
    }
    statuses = {m["status"] for m in model_results}
    manifest.status = "complete" if statuses == {"complete"} else "partial"
    manifest.finished_at = utcnow()
    manifest.budget_usage = {
        "openrouter_used_today": _budget_used("openrouter"),
        "zen_used_today": _budget_used("zen"),
    }
    (out_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    return manifest.model_dump()


def _budget_used(provider: str) -> int | None:
    try:
        p = build_provider(provider)
        used = p.budget.used_today()
        if hasattr(p, "close"):
            p.close()
        return used
    except Exception:  # noqa: BLE001
        return None
