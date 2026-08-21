"""Stage 2 analysis: paired bilingual (RU/EN) evaluation on the original benchmark.

All language-gap statistics cluster on `pair_id` (the two variants of one semantic item
move together under resampling) per hypotheses.md H4/H5. BH correction is applied
within the confirmatory H4 family (per-model gaps). Everything regenerates from raw
run records.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from stembench.metrics.calibration import (
    brier_score,
    expected_calibration_error,
)
from stembench.metrics.intervals import (
    bootstrap_difference_ci,
    wilson_interval,
)
from stembench.metrics.significance import benjamini_hochberg


def load_stage2(run_dir: Path) -> dict[str, list[dict]]:
    """-> {model_key: [records]} excluding transient-error and synthetic records."""
    runs: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(run_dir.glob("*.jsonl")):
        if path.name.startswith("fake__"):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("error_status") in (
                    "daily_budget_exceeded", "rate_limited", "timeout", "provider_error",
                ):
                    continue
                runs[f"{rec['provider']}::{rec['model']}"].append(rec)
    return dict(runs)


def _pair_id(item_id: str) -> str:
    return item_id.rsplit("-", 1)[0]


def per_model_language_accuracy(runs: dict[str, list[dict]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, recs in sorted(runs.items()):
        entry: dict[str, Any] = {"model": key, "n": len(recs)}
        for lang in ("en", "ru"):
            rs = [r for r in recs if r.get("language") == lang]
            correct = sum(1 for r in rs if r["correctness"] is True)
            parsed = [r for r in rs if r["correctness"] is not None]
            lo, hi = wilson_interval(correct, len(parsed) or 1)
            entry[lang] = {
                "n": len(rs),
                "n_parsed": len(parsed),
                "correct": correct,
                "acc": correct / len(parsed) if parsed else float("nan"),
                "ci_lo": lo,
                "ci_hi": hi,
                "parse_failures": len(rs) - len(parsed),
            }
        # self-report calibration per language
        for lang in ("en", "ru"):
            parsed = [
                r for r in recs
                if r.get("language") == lang and r["correctness"] is not None
                and r.get("self_reported_confidence") is not None
            ]
            if parsed:
                conf = np.array([r["self_reported_confidence"] for r in parsed])
                cor = np.array([float(r["correctness"]) for r in parsed])
                entry[f"{lang}_calibration"] = {
                    "n": len(parsed),
                    "ECE": expected_calibration_error(conf, cor),
                    "brier": brier_score(conf, cor),
                    "mean_conf": float(conf.mean()),
                    "acc": float(cor.mean()),
                }
        out[key] = entry
    return out


def language_gaps(runs: dict[str, list[dict]], n_boot: int = 10000,
                  seed: int = 20260817) -> dict[str, Any]:
    """H4 per-model + H5 pooled EN-RU paired differences, cluster bootstrap on pair_id."""
    per_model: dict[str, dict[str, Any]] = {}
    pooled_pairs: dict[str, list[float]] = defaultdict(list)
    pooled_templates: dict[str, str] = {}
    for key, recs in sorted(runs.items()):
        by_pair: dict[str, dict[str, float | None]] = defaultdict(dict)
        pair_templates: dict[str, str] = {}
        for r in recs:
            if r["correctness"] is None:
                continue
            pair_id = _pair_id(r["item_id"])
            by_pair[pair_id][r.get("language", "en")] = float(
                r["correctness"]
            )
            pair_templates[pair_id] = r.get("template_id") or f"unknown:{pair_id}"
        pairs = {p: v for p, v in by_pair.items() if "en" in v and "ru" in v}
        if len(pairs) < 10:
            per_model[key] = {"n_pairs": len(pairs), "note": "too few complete pairs"}
            continue
        ids = sorted(pairs)
        en = np.array([pairs[p]["en"] for p in ids])
        ru = np.array([pairs[p]["ru"] for p in ids])
        clusters = np.arange(len(ids))
        res = bootstrap_difference_ci(en, ru, clusters=clusters, n_boot=n_boot, seed=seed)
        res["acc_en"] = float(en.mean())
        res["acc_ru"] = float(ru.mean())
        res["n_pairs"] = len(ids)
        template_clusters = np.array([pair_templates[pair_id] for pair_id in ids])
        res["template_cluster_sensitivity"] = {
            **bootstrap_difference_ci(
                en,
                ru,
                clusters=template_clusters,
                n_boot=n_boot,
                seed=seed,
            ),
            "n_templates": int(len(np.unique(template_clusters))),
        }
        per_model[key] = res
        for p in ids:
            pooled_pairs[p].append(pairs[p]["en"] - pairs[p]["ru"])
            pooled_templates[p] = pair_templates[p]
    # H5 pooled across models, clustered on pair
    h5: dict[str, Any] = {"n_pairs": len(pooled_pairs)}
    if pooled_pairs:
        ids = sorted(pooled_pairs)
        vals = np.array([np.mean(pooled_pairs[p]) for p in ids])
        clusters = np.arange(len(ids))
        rng_pct = np.random.default_rng(seed)
        boots = np.empty(2000)
        n_c = len(ids)
        for b in range(2000):
            pick = rng_pct.integers(0, n_c, size=n_c)
            boots[b] = vals[pick].mean()
        lo, hi = np.percentile(boots, [2.5, 97.5])
        p_boot = min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))
        h5.update(
            diff=float(vals.mean()), ci_lo=float(lo), ci_hi=float(hi),
            p_bootstrap=float(p_boot),
        )
        template_clusters = np.array([pooled_templates[pair_id] for pair_id in ids])
        h5["template_cluster_sensitivity"] = {
            **bootstrap_difference_ci(
                vals,
                np.zeros_like(vals),
                clusters=template_clusters,
                n_boot=n_boot,
                seed=seed,
            ),
            "n_templates": int(len(np.unique(template_clusters))),
        }
    # BH within the H4 family
    testable = {k: v for k, v in per_model.items() if "p_bootstrap" in v}
    if testable:
        keys = sorted(testable)
        adj, rej = benjamini_hochberg([testable[k]["p_bootstrap"] for k in keys])
        for k, a, r in zip(keys, adj, rej, strict=True):
            testable[k]["p_bh"] = a
            testable[k]["significant_0.05"] = r
    return {"H4_per_model": per_model, "H5_pooled": h5}


def category_breakdowns(runs: dict[str, list[dict]]) -> dict[str, Any]:
    """H6 subject cells, H7 difficulty, H8 answer type — descriptive with CIs."""
    def cells(recs: list[dict], field: str) -> dict[str, dict[str, float]]:
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in recs:
            groups[str(r.get(field, "?"))].append(r)
        out = {}
        for g, rs in sorted(groups.items()):
            parsed = [r for r in rs if r["correctness"] is not None]
            correct = sum(1 for r in parsed if r["correctness"])
            lo, hi = wilson_interval(correct, len(parsed) or 1)
            out[g] = {
                "n": len(rs), "n_parsed": len(parsed), "acc": correct / len(parsed)
                if parsed else float("nan"),
                "ci_lo": lo, "ci_hi": hi, "small_cell": len(parsed) < 10,
            }
        return out

    out: dict[str, Any] = {}
    for key, recs in sorted(runs.items()):
        out[key] = {
            "by_subject": cells(recs, "subject"),
            "by_difficulty": cells(recs, "difficulty"),
            "by_answer_type": cells(recs, "answer_type"),
            "by_language": cells(recs, "language"),
        }
    # difficulty trend per model (school >= university >= olympiad descriptive)
    trends = {}
    for key, _recs in sorted(runs.items()):
        d = out[key]["by_difficulty"]
        seq = [d.get(level, {}).get("acc", float("nan")) for level in
               ("school", "university", "olympiad")]
        if not any(np.isnan(x) for x in seq):
            trends[key] = {"school": seq[0], "university": seq[1], "olympiad": seq[2],
                           "monotone_decreasing": bool(seq[0] > seq[1] > seq[2])}
    out["difficulty_trend"] = trends
    return out


def interaction_table(runs: dict[str, list[dict]]) -> dict[str, dict[str, dict[str, float]]]:
    """Exploratory model × language accuracy table (label exploratory downstream)."""
    table: dict[str, dict[str, dict[str, float]]] = {}
    for key, recs in sorted(runs.items()):
        entry: dict[str, dict[str, float]] = {}
        for lang in ("en", "ru"):
            parsed = [
                r for r in recs
                if r.get("language") == lang and r["correctness"] is not None
            ]
            entry[lang] = {
                "acc": (sum(1 for r in parsed if r["correctness"]) / len(parsed))
                if parsed else float("nan"),
                "n": len(parsed),
            }
        table[key] = entry
    return table


def failure_patterns(runs: dict[str, list[dict]]) -> dict[str, dict[str, float]]:
    out = {}
    for key, recs in sorted(runs.items()):
        n = len(recs)
        parse_fail = sum(1 for r in recs if r["correctness"] is None)
        empty = sum(1 for r in recs if not (r.get("raw_response") or "").strip())
        out[key] = {
            "n": n,
            "parse_failure_rate": parse_fail / n if n else 0.0,
            "empty_response_rate": empty / n if n else 0.0,
        }
    return out


def generate_stage2_report(run_dir: Path, out_dir: Path | None = None) -> dict[str, Any]:
    out_dir = out_dir or run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = load_stage2(run_dir)
    report = {
        "run_dir": str(run_dir),
        "per_model_language": per_model_language_accuracy(runs),
        "language_gaps": language_gaps(runs),
        "categories": category_breakdowns(runs),
        "interaction_model_x_language_exploratory": interaction_table(runs),
        "failure_patterns": failure_patterns(runs),
    }
    (out_dir / "stage2_analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
    )
    try:
        from stembench.viz.figures import (
            accuracy_ci_plot,
            language_gap_forest,
            subject_heatmap,
        )

        fig_dir = out_dir / "figures"
        h4 = report["language_gaps"]["H4_per_model"]
        forest = {
            k: v for k, v in h4.items()
            if "ci_lo" in v and "diff" in v
        }
        if forest:
            language_gap_forest(
                {k.split("::")[-1]: v for k, v in forest.items()},
                fig_dir,
            )
        cats = {m: c for m, c in report["categories"].items()
                if isinstance(c, dict) and "by_subject" in c}
        subject_heatmap({m: c["by_subject"] for m, c in cats.items()}, fig_dir)
        subject_heatmap(
            {m: c["by_difficulty"] for m, c in cats.items()}, fig_dir,
            out_name="difficulty_heatmap.png",
            title="Accuracy by difficulty tier (cells with n<5 blank)",
        )
        subject_heatmap(
            {m: c["by_answer_type"] for m, c in cats.items()}, fig_dir,
            out_name="answer_type_heatmap.png",
            title="Accuracy by answer format (cells with n<5 blank)",
        )
        # lenient accuracy (parse failure = incorrect) with Wilson CIs
        metrics_like: dict[str, dict[str, Any]] = {}
        for key, entry in report["per_model_language"].items():
            n = entry["en"]["n"] + entry["ru"]["n"]
            correct = entry["en"]["correct"] + entry["ru"]["correct"]
            lo, hi = wilson_interval(correct, n or 1)
            metrics_like[key] = {"accuracy_lenient": {
                "acc": correct / n if n else float("nan"),
                "ci_lo": lo, "ci_hi": hi, "n": n,
            }}
        accuracy_ci_plot(metrics_like, fig_dir)
    except Exception as e:  # noqa: BLE001
        report["figure_error"] = str(e)
    return report


if __name__ == "__main__":
    import sys

    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results/stage2/S2-E1")
    rep = generate_stage2_report(run_dir)
    if rep.get("figure_error"):
        print("figure error:", rep["figure_error"], file=sys.stderr)
    print(f"analysis written to {run_dir / 'analysis' / 'stage2_analysis.json'}")
