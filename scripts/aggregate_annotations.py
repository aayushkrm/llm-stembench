#!/usr/bin/env python3
"""Aggregate expert annotations: majority labels, Fleiss' kappa (gate), pairwise
Cohen's kappa, adjudication log. Only genuine independent human round-1 labels are
valid input (see docs/annotation/workflow.md honesty rules).

Usage:
  python scripts/aggregate_annotations.py --annotations annotations_round1_R*.jsonl \
      --round 1 --out-dir adjudication
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stembench.metrics.agreement import cohens_kappa, fleiss_kappa  # noqa: E402

CORRECT_CODES = {"valid": 0, "invalid": 1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", nargs="+", required=True)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--gate", type=float, default=0.75)
    ap.add_argument("--out-dir", default="adjudication")
    args = ap.parse_args()

    rows = []
    for pattern in args.annotations:
        for path in sorted(glob.glob(pattern)):
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r.get("round") == args.round:
                        rows.append(r)

    by_item: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_item[r["item_id"]][r["rater_id"]] = r
    raters = sorted({r["rater_id"] for r in rows})
    if len(raters) < 2:
        raise SystemExit("need >= 2 independent raters for agreement analysis")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Fleiss' kappa on the correctness decision -------------------------
    categories = ["valid", "invalid"]
    matrix = []
    kept_items = []
    for item_id, rr in sorted(by_item.items()):
        if len(rr) < 2:
            continue
        counts = [sum(1 for x in rr.values() if x["correct"] == c) for c in categories]
        if sum(counts) >= 2:
            matrix.append(counts)
            kept_items.append(item_id)
    fleiss = fleiss_kappa(matrix) if matrix else None

    # --- pairwise Cohen's kappa --------------------------------------------
    pairwise = {}
    for a, b in itertools.combinations(raters, 2):
        ra, rb = [], []
        for _item_id, rr in sorted(by_item.items()):
            if a in rr and b in rr and rr[a]["correct"] in CORRECT_CODES \
                    and rr[b]["correct"] in CORRECT_CODES:
                ra.append(CORRECT_CODES[rr[a]["correct"]])
                rb.append(CORRECT_CODES[rr[b]["correct"]])
        pairwise[f"{a}-{b}"] = cohens_kappa(ra, rb, 2) if ra else None

    # --- majority labels + disagreements -----------------------------------
    majority = {}
    disagreements = []
    for item_id, rr in sorted(by_item.items()):
        votes = defaultdict(int)
        for x in rr.values():
            votes[x["correct"]] += 1
        top = max(votes.items(), key=lambda kv: kv[1])
        majority[item_id] = {"label": top[0], "votes": dict(votes)}
        if len(votes) > 1 and top[1] <= len(rr) / 2:
            disagreements.append({"item_id": item_id, "votes": dict(votes),
                                  "comments": [x.get("comment", "") for x in rr.values()]})

    report = {
        "round": args.round,
        "n_items": len(by_item),
        "n_raters": len(raters),
        "raters": raters,
        "fleiss_kappa_correct": fleiss,
        "pairwise_cohens_kappa": pairwise,
        "gate": args.gate,
        "gate_passed": bool(fleiss and fleiss["kappa"] >= args.gate),
        "n_disagreements": len(disagreements),
    }
    (out / f"agreement_round{args.round}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / f"majority_round{args.round}.json").write_text(
        json.dumps(majority, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / f"disagreements_round{args.round}.json").write_text(
        json.dumps(disagreements, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["gate_passed"]:
        print(f"\nGATE NOT PASSED (Fleiss kappa {fleiss['kappa']:.3f} < {args.gate}). "
              "See disagreements file; run a revision round.")


if __name__ == "__main__":
    main()
