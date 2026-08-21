#!/usr/bin/env python3
"""Merge error-annotation pools from multiple Stage 1 runs into one annotated set +
distribution table/figure. Usage:

  python scripts/merge_error_annotations.py \
      --annotations results/stage1/error_analysis/annotations.jsonl \
      [--annotations results/stage1/error_analysis/annotations_S1-P2.jsonl] \
      --out-dir results/stage1/error_analysis
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", action="append", required=True)
    ap.add_argument("--out-dir", default="results/stage1/error_analysis")
    args = ap.parse_args()

    rows = []
    seen: set[tuple[str, str]] = set()
    for pattern in args.annotations:
        for path in sorted(Path().glob(pattern)):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                key = (r["model"], r["item_id"], r.get("run_id", ""))
                if key in seen or not r.get("primary"):
                    continue
                seen.add(key)
                rows.append(r)

    out = Path(args.out_dir)
    merged = out / "annotations_merged.jsonl"
    with open(merged, "w", encoding="utf-8") as f:
        for r in sorted(rows, key=lambda r: (r.get("run_id", ""), r["model"], r["item_id"])):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(rows)
    primary = Counter(r["primary"] for r in rows)
    multi = Counter(lab for r in rows for lab in r.get("labels", []))
    from stembench.metrics.intervals import wilson_interval

    dist = {
        "n_annotated": n,
        "annotator": rows[0].get("annotator", "") if rows else "",
        "primary_distribution": {},
        "multilabel_incidence": {k: v / n for k, v in sorted(multi.items())},
        "source_files": args.annotations,
    }
    for code, k in sorted(primary.items()):
        lo, hi = wilson_interval(k, n)
        dist["primary_distribution"][code] = {
            "n": k, "share": k / n, "ci_lo": lo, "ci_hi": hi,
        }
    (out / "error_distribution.json").write_text(
        json.dumps(dist, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    try:
        from stembench.viz.figures import error_taxonomy_bar

        error_taxonomy_bar(
            {k: v["share"] for k, v in dist["primary_distribution"].items()},
            out / "figures", n,
        )
    except Exception as e:  # noqa: BLE001
        dist["figure_error"] = str(e)
        (out / "error_distribution.json").write_text(
            json.dumps(dist, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(json.dumps(dist["primary_distribution"], indent=2))
    print(f"merged {n} annotated errors -> {merged}")


if __name__ == "__main__":
    main()
