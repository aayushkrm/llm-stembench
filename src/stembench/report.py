"""Report generation: metrics, tables, figures from a run's JSONL records.

Everything is regenerated from raw records — no hand-edited numbers anywhere.
Statistical choices follow decisions.md D5 (Wilson CIs; McNemar/Cochran for paired
models; BH across the confirmatory family; both strict and lenient accuracy).
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from stembench.metrics.calibration import (
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    negative_log_likelihood,
    reliability_bins,
)
from stembench.metrics.classification import confusion_matrix, prf
from stembench.metrics.intervals import wilson_interval
from stembench.metrics.significance import (
    benjamini_hochberg,
    cochran_q,
    mcnemar_test,
)


def load_run(run_dir: Path) -> dict[str, list[dict]]:
    """-> {model_key: [records]} for records that completed (have raw_response)."""
    runs: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(run_dir.glob("*.jsonl")):
        if path.name.startswith("fake__"):
            continue  # never mix synthetic records into empirical reports
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("error_status") in (
                    "daily_budget_exceeded",
                    "rate_limited",
                    "timeout",
                    "provider_error",
                ):
                    continue
                runs[f"{rec['provider']}::{rec['model']}"].append(rec)
    return dict(runs)


def _acc_ci(correct: int, n: int) -> dict[str, float]:
    lo, hi = wilson_interval(correct, n)
    return {"acc": correct / n if n else float("nan"), "ci_lo": lo, "ci_hi": hi, "n": n}


def model_metrics(records: list[dict]) -> dict[str, Any]:
    n = len(records)
    correct = sum(1 for r in records if r["correctness"] is True)
    parsed = [r for r in records if r["correctness"] is not None]
    strict = _acc_ci(correct, len(parsed)) if parsed else {"acc": float("nan")}
    lenient = _acc_ci(correct, n)
    out: dict[str, Any] = {
        "n": n,
        "n_parsed": len(parsed),
        "parse_failure_rate": (n - len(parsed)) / n if n else 0.0,
        "accuracy_strict": strict,
        "accuracy_lenient": lenient,
    }
    # self-reported confidence calibration (channel: self_report)
    conf = np.array([r["self_reported_confidence"] for r in parsed
                     if r.get("self_reported_confidence") is not None])
    cor = np.array([float(r["correctness"]) for r in parsed
                    if r.get("self_reported_confidence") is not None])
    if len(conf):
        out["calibration_self_report"] = {
            "n": int(len(conf)),
            "population": "parsed answers only (parse failures excluded)",
            "ECE": expected_calibration_error(conf, cor),
            "MCE": maximum_calibration_error(conf, cor),
            "brier": brier_score(conf, cor),
            "mean_confidence": float(conf.mean()),
            "reliability_bins": reliability_bins(conf, cor),
        }
        # H2 (hypotheses.md): overconfidence gap, lenient correctness (parse
        # failures with elicited confidence count as incorrect)
        all_conf = np.array([
            r["self_reported_confidence"] for r in records
            if r.get("self_reported_confidence") is not None
        ])
        all_cor = np.array([
            float(r["correctness"] is True) for r in records
            if r.get("self_reported_confidence") is not None
        ])
        if len(all_conf):
            from stembench.metrics.intervals import bootstrap_difference_ci

            gap = bootstrap_difference_ci(all_conf, all_cor, n_boot=10000, seed=42)
            out["h2_overconfidence_gap_lenient"] = gap
    # letter-probability calibration when provider logprobs exist (channel: token_prob)
    lp_conf, lp_cor = [], []
    for r in parsed:
        p = letter_prob_from_logprobs(r)
        if p is not None and r["correctness"] is not None:
            lp_conf.append(p)
            lp_cor.append(float(r["correctness"]))
    if lp_conf:
        c = np.array(lp_conf)
        y = np.array(lp_cor)
        out["calibration_token_prob"] = {
            "n": int(len(c)),
            "population": "first letter-dominated generated position (post-commit "
            "context: P(letter) is typically ~1.0 and uninformative about belief at "
            "decision time — see report limitations)",
            "degenerate": bool(c.mean() > 0.995 and len(c) < 60),
            "ECE": expected_calibration_error(c, y),
            "MCE": maximum_calibration_error(c, y),
            "brier": brier_score(c, y),
            "NLL": negative_log_likelihood(c),
            "mean_confidence": float(c.mean()),
            "reliability_bins": reliability_bins(c, y),
        }
    # MC confusion over letters (classes = answer letters; caveat: letters are
    # positions, not semantic classes — reported for the classification-metric
    # requirement with an explicit definition)
    mc = [r for r in parsed if r.get("answer_type") == "mc" and r.get("parsed_answer")]
    if mc:
        gold_idx = [ord(r["reference_answer"]) - 65 for r in mc]
        pred_idx = [ord(r["parsed_answer"]) - 65 for r in mc]
        k = 4
        cm = confusion_matrix(gold_idx, pred_idx, k)
        out["mc_confusion"] = cm.tolist()
        out["mc_macro_f1"] = prf(cm, "macro")
    return out


LETTERS = ("A", "B", "C", "D")


def letter_prob_from_logprobs(record: dict) -> float | None:
    """P(chosen letter) at the first generated position whose top tokens look like
    MC letters, normalized over A-D present. Returns None when unusable."""
    lp = record.get("logprobs_raw")
    if not lp:
        return None
    parsed = record.get("parsed_answer")
    if not parsed:
        return None
    for position in lp:
        if not isinstance(position, dict):
            continue
        tops = position.get("top_logprobs") or []
        probs = {}
        for t in tops:
            tok = (t.get("token") or "").strip()
            for L in LETTERS:
                if tok == L or tok.strip("()").strip() == L:
                    probs[L] = probs.get(L, 0.0) + math.exp(t.get("logprob", -100))
        if probs.get(parsed):
            total = sum(probs.values())
            if total > 0.5:  # letters dominate this position
                return min(1.0, probs[parsed] / total)
    return None


def paired_models_analysis(runs: dict[str, list[dict]]) -> dict[str, Any]:
    """McNemar pairwise + Cochran's Q on the intersection of items, with BH adjustment.

    Outcomes are LENIENT (parse failure counts as incorrect): conditioning on parsed
    answers would bias accuracy upward and starve the paired tests.
    """
    by_item: dict[str, dict[str, bool]] = defaultdict(dict)
    for key, recs in runs.items():
        for r in recs:
            by_item[r["item_id"]][key] = r["correctness"] is True
    keys = sorted(runs)
    pairs = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            common = [it for it, m in by_item.items() if a in m and b in m]
            if len(common) < 5:
                continue
            va = np.array([1 if by_item[it][a] else 0 for it in common])
            vb = np.array([1 if by_item[it][b] else 0 for it in common])
            res = mcnemar_test(va, vb)
            pairs.append({
                "model_a": a, "model_b": b, "n_common": len(common),
                "acc_a": float(va.mean()), "acc_b": float(vb.mean()),
                "diff": float(va.mean() - vb.mean()),
                "b": res.b, "c": res.c,
                "p_exact": res.p_exact, "p_chi2": res.p_chi2,
            })
    if pairs:
        adj, rej = benjamini_hochberg([p["p_exact"] for p in pairs])
        for p, a, r in zip(pairs, adj, rej, strict=True):
            p["p_bh"] = a
            p["significant_0.05"] = r
    q = None
    if len(keys) >= 3:
        common_all = [it for it, m in by_item.items() if all(k in m for k in keys)]
        if len(common_all) >= 10:
            M = np.array(
                [[1 if by_item[it][k] else 0 for k in keys] for it in common_all]
            )
            q = cochran_q(M)
            q["n_items"] = len(common_all)
            q["models"] = keys
    return {"pairwise_mcnemar": pairs, "cochran_q": q}


def breakdown(records: list[dict], field: str, min_cell: int = 5) -> dict[str, Any]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[str(r.get(field, "?"))].append(r)
    out = {}
    for g, rs in sorted(groups.items()):
        correct = sum(1 for r in rs if r["correctness"] is True)
        n = len(rs)
        entry = _acc_ci(correct, n)
        entry["small_cell"] = n < min_cell
        out[g] = entry
    return out


def generate_report(run_dir: Path, out_dir: Path | None = None) -> dict[str, Any]:
    out_dir = out_dir or run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = load_run(run_dir)
    metrics = {key: model_metrics(recs) for key, recs in runs.items()}
    stats = paired_models_analysis(runs)
    by_subject = {
        key: breakdown(recs, "subject") for key, recs in runs.items()
    }
    report = {
        "run_dir": str(run_dir),
        "models": metrics,
        "comparisons": stats,
        "by_subject": by_subject,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    _write_model_table(metrics, out_dir / "model_table.csv")
    _write_pair_table(stats, out_dir / "pairwise_table.csv")
    try:
        from stembench.viz import figures as F

        F.reliability_grid(metrics, out_dir / "figures")
        F.accuracy_ci_plot(metrics, out_dir / "figures")
        F.subject_heatmap(by_subject, out_dir / "figures")
        F.confusion_grid(metrics, out_dir / "figures")
    except Exception as e:  # noqa: BLE001 — figures are best-effort but logged
        report["figure_error"] = str(e)
    return report


def _write_model_table(metrics: dict, path: Path) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "model", "n", "n_parsed", "acc_lenient", "ci_lo", "ci_hi",
            "acc_strict", "parse_fail_rate", "mean_self_conf",
            "ECE_self", "MCE_self", "brier_self",
            "ECE_tokenprob", "n_tokenprob",
        ])
        for key, m in sorted(metrics.items()):
            s = m.get("accuracy_strict", {})
            lent = m["accuracy_lenient"]
            cal = m.get("calibration_self_report", {})
            tcal = m.get("calibration_token_prob", {})
            w.writerow([
                key, m["n"], m["n_parsed"],
                f"{lent['acc']:.4f}", f"{lent['ci_lo']:.4f}", f"{lent['ci_hi']:.4f}",
                f"{s.get('acc', float('nan')):.4f}",
                f"{m['parse_failure_rate']:.4f}",
                f"{cal.get('mean_confidence', float('nan')):.4f}",
                f"{cal.get('ECE', float('nan')):.4f}",
                f"{cal.get('MCE', float('nan')):.4f}",
                f"{cal.get('brier', float('nan')):.4f}",
                f"{tcal.get('ECE', float('nan')):.4f}",
                tcal.get("n", 0),
            ])


def _write_pair_table(stats: dict, path: Path) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model_a", "model_b", "n_common", "acc_a", "acc_b", "diff",
                    "b", "c", "p_exact", "p_bh", "significant_0.05"])
        for p in stats["pairwise_mcnemar"]:
            w.writerow([
                p["model_a"], p["model_b"], p["n_common"],
                f"{p['acc_a']:.4f}", f"{p['acc_b']:.4f}", f"{p['diff']:.4f}",
                p["b"], p["c"], f"{p['p_exact']:.4g}", f"{p['p_bh']:.4g}",
                p["significant_0.05"],
            ])
