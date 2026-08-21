"""Stage 2 paired bilingual analysis on synthetic records (offline, deterministic)."""

from __future__ import annotations

import json

import numpy as np

from stembench.analysis_stage2 import (
    category_breakdowns,
    failure_patterns,
    generate_stage2_report,
    language_gaps,
    load_stage2,
    per_model_language_accuracy,
)


def _mk(model: str, pair: str, lang: str, correct, template: str) -> dict:
    return {
        "run_id": "T2", "item_id": f"{pair}-{lang}", "dataset": "stembench_v1",
        "subject": "math", "language": lang, "difficulty": "school",
        "answer_type": "mc", "provider": "zen", "model": model,
        "raw_response": "Answer: A", "parsed_answer": "A", "reference_answer": "A",
        "correctness": correct, "template_id": template,
        "self_reported_confidence": 0.9 if correct else 0.7,
        "error_status": "" if correct is not None else "parse_failure",
    }


def _records() -> list[dict]:
    """40 pairs x 2 langs; EN correct on 30, RU correct on 20 (gap = +0.25 for EN);
    templates: 8 templates x 5 pairs each (test sensitivity clustering)."""
    rng = np.random.default_rng(7)
    recs = []
    for i in range(40):
        pair = f"MATH-{i:04d}"
        tmpl = f"tmpl-{i % 8}"
        en_ok = bool(rng.random() < 0.75)
        ru_ok = bool(rng.random() < 0.5)
        recs.append(_mk("m1", pair, "en", en_ok, tmpl))
        recs.append(_mk("m1", pair, "ru", ru_ok, tmpl))
    return recs


def test_language_gaps_pair_clustered_and_template_sensitivity():
    runs = {"zen::m1": _records()}
    out = language_gaps(runs, n_boot=2000, seed=1)
    h4 = out["H4_per_model"]["zen::m1"]
    assert h4["n_pairs"] == 40
    assert h4["template_cluster_sensitivity"]["n_templates"] == 8
    # diff == mean(en) - mean(ru) exactly
    assert h4["diff"] == h4["acc_en"] - h4["acc_ru"]
    # CIs contain the point estimate
    assert h4["ci_lo"] <= h4["diff"] <= h4["ci_hi"]
    # sensitivity CI is wider or equal when clustering on templates (fewer units)
    sens = h4["template_cluster_sensitivity"]
    assert sens["n_templates"] == 8
    assert sens["ci_lo"] <= sens["diff"] <= sens["ci_hi"]
    # template sensitivity is computed on fewer resampling units; its CI need not be
    # wider in every sample, but the point estimate must match the pair-clustered one
    assert sens["diff"] == h4["diff"]


def test_h5_pooled_and_too_few_pairs_note():
    runs = {"zen::m1": _records()[:16]}  # only 8 complete pairs < 10
    out = language_gaps(runs, n_boot=500, seed=1)
    assert out["H4_per_model"]["zen::m1"]["note"] == "too few complete pairs"
    runs2 = {"zen::m1": _records()}
    h5 = language_gaps(runs2, n_boot=500, seed=1)["H5_pooled"]
    assert h5["n_pairs"] == 40
    assert h5["ci_lo"] <= h5["diff"] <= h5["ci_hi"]
    assert "template_cluster_sensitivity" in h5


def test_per_model_language_accuracy_hand_case():
    recs = [
        _mk("m1", "P-01", "en", True, "t"),
        _mk("m1", "P-01", "ru", False, "t"),
        _mk("m1", "P-02", "en", True, "t"),
        _mk("m1", "P-02", "ru", None, "t"),  # parse failure: excluded from strict acc
    ]
    out = per_model_language_accuracy({"zen::m1": recs})
    e = out["zen::m1"]
    # en: 2/2 = 1.0 ; ru strict: 0/1 = 0.0 with 1 parse failure recorded
    assert e["en"]["acc"] == 1.0 and e["en"]["n"] == 2
    assert e["ru"]["n_parsed"] == 1 and e["ru"]["acc"] == 0.0
    assert e["ru"]["parse_failures"] == 1
    # calibration sub-dicts computed on parsed answers only
    assert e["en_calibration"]["n"] == 2 and e["ru_calibration"]["n"] == 1


def test_category_breakdowns_and_failures():
    runs = {"zen::m1": _records()[:10] + [_mk("m1", "MATH-9999", "en", None, "t9")]}
    cats = category_breakdowns(runs)
    m = cats["zen::m1"]
    assert m["by_language"]["en"]["n"] == 6 and m["by_language"]["ru"]["n"] == 5
    assert m["by_answer_type"]["mc"]["n"] == 11
    fp = failure_patterns(runs)
    assert fp["zen::m1"]["parse_failure_rate"] == pytest_approx(1 / 11)


def pytest_approx(x):
    import pytest

    return pytest.approx(x)


def test_generate_stage2_report_end_to_end(tmp_path):
    run_dir = tmp_path / "S2X"
    run_dir.mkdir()
    with open(run_dir / "zen__m1.jsonl", "w") as f:
        for r in _records():
            f.write(json.dumps(r) + "\n")
    # fake-provider file must be excluded
    with open(run_dir / "fake__synthetic.jsonl", "w") as f:
        f.write(json.dumps(_mk("fm", "X-1", "en", True, "t")) + "\n")
    # transient-error record must be excluded
    bad = _mk("m1", "MATH-9999", "en", None, "t9")
    bad["error_status"] = "rate_limited"
    with open(run_dir / "zen__m1.jsonl", "a") as f:
        f.write(json.dumps(bad) + "\n")

    rep = generate_stage2_report(run_dir)
    loaded = load_stage2(run_dir)
    assert set(loaded) == {"zen::m1"}  # fake excluded
    assert all(r["item_id"] != "MATH-9999-en" for r in loaded["zen::m1"])
    out = run_dir / "analysis"
    assert (out / "stage2_analysis.json").exists()
    data = json.loads((out / "stage2_analysis.json").read_text())
    assert "language_gaps" in data and "H4_per_model" in data["language_gaps"]
    assert rep["failure_patterns"]["zen::m1"]["n"] == 80
