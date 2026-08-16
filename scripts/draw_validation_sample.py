#!/usr/bin/env python3
"""Draw the stratified expert-validation sample from a built benchmark.

Usage:
  python scripts/draw_validation_sample.py --data data/stembench_v1/items.jsonl \
      --n-pairs 60 --seed 7 --out data/validation_sample.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/stembench_v1/items.jsonl")
    ap.add_argument("--n-pairs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/validation_sample.jsonl")
    args = ap.parse_args()

    pairs: dict[str, list[dict]] = defaultdict(list)
    for line in Path(args.data).read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            pairs[item["pair_id"]].append(item)

    # stratify by subject x difficulty; both language variants included per pair
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for pid, items in pairs.items():
        meta = items[0]
        strata[(meta["subject"], meta["difficulty"])].append(pid)

    rng = random.Random(args.seed)
    raw_frac = {k: len(v) / len(pairs) for k, v in strata.items()}
    alloc = {k: int(args.n_pairs * f) for k, f in raw_frac.items()}
    remaining = args.n_pairs - sum(alloc.values())
    for k in sorted(strata, key=lambda k: raw_frac[k] - alloc[k], reverse=True)[: max(remaining, 0)]:
        alloc[k] += 1

    chosen: list[str] = []
    for k, pids in sorted(strata.items()):
        take = min(alloc.get(k, 0), len(pids))
        chosen.extend(rng.sample(pids, take))
    chosen.sort()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_items = 0
    with open(out, "w", encoding="utf-8") as f:
        for pid in chosen:
            for item in sorted(pairs[pid], key=lambda i: i["language"]):
                slim = {k: item[k] for k in (
                    "item_id", "pair_id", "language", "subject", "topic", "difficulty",
                    "question", "answer_type", "choices", "canonical_answer",
                    "acceptable_alternatives", "tolerance", "units", "solution",
                )}
                f.write(json.dumps(slim, ensure_ascii=False) + "\n")
                n_items += 1
    print(f"sampled {len(chosen)} pairs ({n_items} language records) -> {out}")
    by_cell = defaultdict(int)
    for pid in chosen:
        m = pairs[pid][0]
        by_cell[(m["subject"], m["difficulty"])] += 1
    for k, v in sorted(by_cell.items()):
        print(f"  {k[0]:<9} {k[1]:<11} {v}")


if __name__ == "__main__":
    main()
