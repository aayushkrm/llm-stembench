#!/usr/bin/env python3
"""Export the stratified sample of incorrect responses for error annotation.

Applies transparent rule-based pre-labels where the label follows mechanically from
the record (parse failure -> E6, empty response -> E10); all other items are left
for (model-assisted, clearly labeled) semantic annotation per docs/error_taxonomy.md.

Usage:
  python scripts/export_error_sample.py --run results/stage1/S1-P1 \
      --target 150 --seed 11 --out results/stage1/error_analysis/error_sample.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--target", type=int, default=150)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run)
    incorrect: list[dict] = []
    n_total = 0
    n_incorrect = 0
    for path in sorted(run_dir.glob("*.jsonl")):
        if path.name.startswith("fake__"):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("error_status") in (
                "daily_budget_exceeded", "rate_limited", "timeout", "provider_error",
            ):
                continue
            n_total += 1
            if rec.get("correctness") is False or rec.get("error_status") == "parse_failure":
                n_incorrect += 1
                incorrect.append(rec)

    rng = random.Random(args.seed)
    # stratified by model x subject, proportional with largest remainder
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for rec in incorrect:
        strata[(rec["model"], rec["subject"])].append(rec)
    target = min(args.target, n_incorrect)
    raw = {k: len(v) / n_incorrect * target for k, v in strata.items()}
    alloc = {k: int(x) for k, x in raw.items()}
    for k in sorted(strata, key=lambda k: raw[k] - alloc[k], reverse=True)[
        : max(target - sum(alloc.values()), 0)
    ]:
        alloc[k] += 1

    sample: list[dict] = []
    for k, recs in sorted(strata.items()):
        sample.extend(rng.sample(recs, min(alloc.get(k, 0), len(recs))))
    sample.sort(key=lambda r: (r["model"], r["item_id"]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for rec in sample:
            pre = None
            if rec.get("error_status") == "parse_failure":
                pre = "E6"
            elif not (rec.get("raw_response") or "").strip():
                pre = "E10"
            slim = {
                "run_id": rec["run_id"],
                "item_id": rec["item_id"],
                "model": rec["model"],
                "provider": rec["provider"],
                "subject": rec["subject"],
                "question": rec["prompt_text"],
                "gold_answer": rec["reference_answer"],
                "response": (rec.get("raw_response") or "")[:4000],
                "parsed_answer": rec.get("parsed_answer"),
                "parse_method": rec.get("parse_method", ""),
                "error_status": rec.get("error_status", ""),
                "rule_prelabel": pre,
                "labels": [],
                "primary": None,
                "evidence_quote": "",
                "note": "",
                "annotator": "rule-prelabel" if pre else "",
            }
            f.write(json.dumps(slim, ensure_ascii=False) + "\n")

    print(f"total evaluated: {n_total}, incorrect: {n_incorrect}, sampled: {len(sample)}")
    prelabels = sum(1 for r in sample if r.get("rule_prelabel"))
    print(f"rule-prelabeled (E6/E10): {prelabels}; semantic annotation needed: "
          f"{len(sample) - prelabels}")


if __name__ == "__main__":
    main()
