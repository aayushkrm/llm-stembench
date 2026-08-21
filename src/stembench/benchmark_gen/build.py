"""Deterministic build of the original bilingual STEM benchmark.

Usage:
    PYTHONPATH=src python -m stembench.benchmark_gen.build \
        --out data/stembench_v1 --seed 20260817

Pipeline: generate drafts (seeded, with a same-topic near-duplicate guard) ->
verify every answer through the independent second code path -> run all QC
gates -> write items.jsonl, stats/distribution.csv, verification_report.json
and DATASET_VERSION.  Any verifier or QC failure exits nonzero and leaves the
output directory untouched (build-first, atomic swap at the end).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stembench.schemas import AnswerType, Split

from . import chemistry_gen, math_gen, physics_gen
from ._core import (
    DEFAULT_SEED,
    VERSION,
    GenContext,
    PairBundle,
    PairDraft,
    apply_mc,
    make_items,
)
from .qc import run_qc
from .verify import verify_pair

MODULES = (math_gen, physics_gen, chemistry_gen)
MAX_ATTEMPTS = 60  # resampling budget per item for the near-duplicate guard
DEV_PER_SUBJECT = 6


def _alias(module: Any, key: str) -> str:
    return module.KEY_ALIASES.get(key, key)


def _generate_subject(ctx: GenContext, module: Any) -> list[PairDraft]:
    drafts: list[PairDraft] = []
    subject = module.SUBJECT.value
    idx = 0
    for topic_key, difficulty, count, atype in module.SPEC:
        fn = module.GENERATORS[topic_key]
        for _ in range(count):
            rng = ctx.rng(subject, topic_key, idx)
            draft: PairDraft | None = None
            for _attempt in range(MAX_ATTEMPTS):
                candidate = fn(rng, idx, atype)
                if (
                    not ctx.too_similar(subject, _alias(module, topic_key), "en", candidate.question_en)
                    and not ctx.too_similar(subject, _alias(module, topic_key), "ru", candidate.question_ru)
                ):
                    draft = candidate
                    break
            if draft is None:
                raise RuntimeError(
                    f"{subject}/{topic_key} idx={idx}: could not generate a sufficiently distinct "
                    f"question after {MAX_ATTEMPTS} attempts"
                )
            if draft.difficulty != difficulty:
                raise RuntimeError(
                    f"{subject}/{topic_key} idx={idx}: generator emitted difficulty "
                    f"{draft.difficulty.value}, SPEC requires {difficulty.value}"
                )
            if draft.answer_type != atype:
                raise RuntimeError(
                    f"{subject}/{topic_key} idx={idx}: generator emitted answer type "
                    f"{draft.answer_type.value}, SPEC requires {atype.value}"
                )
            ctx.record(subject, _alias(module, topic_key), "en", draft.question_en)
            ctx.record(subject, _alias(module, topic_key), "ru", draft.question_ru)
            if atype == AnswerType.MC:
                apply_mc(draft, rng)
            base = _alias(module, topic_key)
            try:
                rubric_en, rubric_ru = module.RUBRICS[(base, difficulty)]
            except KeyError as exc:
                raise RuntimeError(f"missing rubric for {base}/{difficulty}") from exc
            draft.rubric_en = rubric_en
            draft.rubric_ru = rubric_ru
            drafts.append(draft)
            idx += 1
    return drafts


def _dev_indices(n: int) -> set[int]:
    """Evenly spaced deterministic dev indices for topic coverage."""
    return {round(k * (n - 1) / (DEV_PER_SUBJECT - 1)) for k in range(DEV_PER_SUBJECT)}


def assemble(seed: int) -> list[PairBundle]:
    """Generate, verify and assemble all bilingual pairs for one seed."""
    ctx = GenContext(seed)
    bundles: list[PairBundle] = []
    failures: list[str] = []
    for module in MODULES:
        drafts = _generate_subject(ctx, module)
        dev_ids = _dev_indices(len(drafts))
        for n, draft in enumerate(drafts, start=1):
            pair_id = f"{module.PREFIX}-{n:04d}"
            split = Split.DEV if (n - 1) in dev_ids else Split.TEST
            base = _alias(module, draft.topic_key)
            mc_texts = list(draft.mc_en) if draft.answer_type == AnswerType.MC else None
            records = verify_pair(
                base,
                dict(draft.params),
                mc_texts,
                candidate_canonical=draft.canonical,
                answer_type=draft.answer_type,
                candidate_numeric_value=draft.numeric_value,
            )
            for rec in records:
                if not rec.passed:
                    failures.append(f"{pair_id} [{base}] {rec.method}: {rec.detail}")
            en, ru = make_items(draft, pair_id, seed, split, records)
            bundles.append(
                PairBundle(
                    pair_id=pair_id,
                    subject=draft.subject,
                    topic=draft.topic,
                    topic_key=base,
                    difficulty=draft.difficulty,
                    answer_type=draft.answer_type,
                    canonical=draft.canonical,
                    split=split,
                    params=dict(draft.params),
                    distractor_tags=draft.distractor_tags,
                    en=en,
                    ru=ru,
                )
            )
    if failures:
        raise RuntimeError("verification failures:\n  " + "\n  ".join(failures))
    return bundles


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_outputs(bundles: list[PairBundle], out_dir: Path, seed: int, metrics: dict[str, Any]) -> None:
    out_dir = out_dir.resolve()
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.tmp-", dir=str(out_dir.parent)))
    try:
        items = sorted(
            (it for b in bundles for it in (b.en, b.ru)),
            key=lambda it: (it.pair_id, it.language.value),
        )
        items_path = tmp / "items.jsonl"
        with open(items_path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(it.model_dump_json() + "\n")
        stats_dir = tmp / "stats"
        stats_dir.mkdir()
        with open(stats_dir / "distribution.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["subject", "difficulty", "answer_type", "language", "n"]
            )
            writer.writeheader()
            for row in metrics["distribution_rows"]:
                writer.writerow(row)
        per_topic: dict[str, dict[str, Any]] = {}
        tag_counts: Counter[str] = Counter()
        for b in bundles:
            slot = per_topic.setdefault(b.topic, {"pairs": 0, "methods": Counter(), "failed": 0})
            slot["pairs"] += 1
            for rec in b.en.verifier:
                slot["methods"][rec.method] += 1
                if not rec.passed:
                    slot["failed"] += 1
            for tag in b.distractor_tags:
                tag_counts[tag] += 1
        report = {
            "version": VERSION,
            "seed": seed,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "counts": {
                "pairs": metrics["n_pairs"],
                "items": metrics["n_items"],
                "pairs_per_subject": metrics["pairs_per_subject"],
                "items_per_language": metrics["items_per_language"],
                "n_dev_pairs": metrics["n_dev_pairs"],
                "n_test_pairs": metrics["n_pairs"] - metrics["n_dev_pairs"],
            },
            "difficulty_per_subject": metrics["difficulty_per_subject"],
            "answer_type_share": metrics["answer_type_share"],
            "mc_letter_distribution": metrics["mc_letter_distribution"],
            "near_duplicates": {
                "metric": "word-3gram Jaccard on normalized question text, per subject+language",
                "max_overall": metrics["max_jaccard_overall"],
                "worst_pair": metrics["max_jaccard_pair"],
                "per_subject_language": metrics["max_jaccard_per_subject_language"],
                "gate": "< 0.80",
            },
            "structural_templates": {
                "metric": "exact normalized question match after masking standalone numeric tokens",
                "purpose": (
                    "descriptive dependence audit; repeated procedural templates are reported, not rejected"
                ),
                "per_subject_language": metrics["structural_templates"],
            },
            "challenge_design_evidence": {
                "caveat": (
                    "generator metadata supports audit but does not replace independent expert difficulty review"
                ),
                "pairs": {
                    bundle.pair_id: {
                        "concepts": bundle.params["challenge_concepts"],
                        "feature": bundle.params["challenge_feature"],
                    }
                    for bundle in bundles
                    if bundle.difficulty.value == "olympiad"
                },
            },
            "verifiers": {
                "methods": metrics["verifier_methods"],
                "passed": metrics["verifier_passed"],
                "failed": metrics["verifier_failed"],
                "pass_rate": metrics["verifier_pass_rate"],
            },
            "per_topic": {
                topic: {
                    "pairs": slot["pairs"],
                    "methods": dict(slot["methods"]),
                    "failed": slot["failed"],
                }
                for topic, slot in sorted(per_topic.items())
            },
            "distractor_perturbation_counts": dict(sorted(tag_counts.items())),
            "checksums": {"items_jsonl_sha256": _sha256_file(items_path)},
        }
        with open(tmp / "verification_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=False)
        version_file = {
            "version": VERSION,
            "seed": seed,
            "generated_at_utc": report["generated_at_utc"],
            "n_pairs": metrics["n_pairs"],
            "n_items": metrics["n_items"],
            "n_items_en": metrics["items_per_language"].get("en", 0),
            "n_items_ru": metrics["items_per_language"].get("ru", 0),
            "n_dev_pairs": metrics["n_dev_pairs"],
            "items_jsonl_sha256": report["checksums"]["items_jsonl_sha256"],
        }
        with open(tmp / "DATASET_VERSION", "w", encoding="utf-8") as f:
            json.dump(version_file, f, ensure_ascii=False, indent=2)
            f.write("\n")
        if out_dir.exists():
            shutil.rmtree(out_dir)
        os.replace(tmp, out_dir)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)


def build_dataset(out_dir: str | Path, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Full deterministic build; returns the QC metrics (raises on any failure)."""
    bundles = assemble(seed)
    result = run_qc(bundles)
    if not result.ok:
        raise RuntimeError("QC gate failures:\n  " + "\n  ".join(result.errors))
    write_outputs(bundles, Path(out_dir), seed, result.metrics)
    return result.metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the original bilingual STEM benchmark")
    parser.add_argument("--out", default="data/stembench_v1", help="output directory")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="global deterministic seed")
    args = parser.parse_args(argv)
    print(f"[build] seed={args.seed} out={args.out}")
    try:
        metrics = build_dataset(args.out, args.seed)
    except RuntimeError as exc:
        print("[build] FAILED", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1
    print(f"[build] pairs={metrics['n_pairs']} items={metrics['n_items']}")
    print(f"[build] pairs per subject: {metrics['pairs_per_subject']}")
    print(f"[build] difficulty per subject: {metrics['difficulty_per_subject']}")
    print(f"[build] answer-type share: {metrics['answer_type_share']}")
    print(f"[build] MC letters: {metrics['mc_letter_distribution']}")
    print(f"[build] max word-3gram Jaccard: {metrics['max_jaccard_overall']} {metrics['max_jaccard_pair']}")
    print(f"[build] verifier pass rate: {metrics['verifier_pass_rate']} ({metrics['verifier_passed']} passed)")
    print(f"[build] dev pairs: {metrics['n_dev_pairs']}")
    print(f"[build] OK -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
