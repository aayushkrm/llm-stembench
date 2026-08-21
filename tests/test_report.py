"""Tests for stembench.report: synthetic-run metrics, tables, figures, fake-record
skipping, and letter_prob_from_logprobs. All numbers hand-computed (see comments).

Fixture design (40 records + 5 skipped fake records):
- items item_00..item_19; MC letters: gold = "ABCD"[i % 4].
- zen__m.jsonl (provider "zen", model "m"), 20 records:
    * correct on items 00-11 (12), self-confidence 0.9
    * incorrect on items 12-15 (4), self-confidence 0.6; item_12 carries token
      logprobs whose letter probabilities are A=0.3, B=0.6, C=0.05, D=0.05 with
      parsed answer B -> letter prob 0.6
    * parse-failure (correctness None, error_status parse_failure) on 16-19 (4)
- openrouter__o.jsonl (provider "openrouter", model "o"), 20 records:
    * correct on 00-09 (10) with confidence 0.7, incorrect on 10-19 (10) with 0.5
- fake__x.jsonl: 5 correct fake records that load_run must skip entirely.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from stembench.report import (
    generate_report,
    letter_prob_from_logprobs,
    load_run,
    model_metrics,
)
from stembench.schemas import ResponseRecord


def _record(i, provider, model, correct, subject, conf, parse_fail=False, logprobs=None):
    gold = "ABCD"[i % 4]
    parsed = None if parse_fail else (gold if correct else "ABCD"[(i % 4 + 1) % 4])
    return ResponseRecord(
        run_id="REPORT-TEST",
        item_id=f"item_{i:02d}",
        dataset="synth",
        subject=subject,
        answer_type="mc",
        provider=provider,
        model=model,
        raw_response="Answer: ...",
        parsed_answer=parsed,
        correctness=None if parse_fail else correct,
        self_reported_confidence=None if parse_fail else conf,
        reference_answer=gold,
        error_status="parse_failure" if parse_fail else "",
        logprobs_raw=logprobs,
    ).model_dump()


@pytest.fixture(scope="session")
def run_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("report_run") / "run"
    d.mkdir()
    zen, orr = [], []
    for i in range(20):
        lp = None
        if i == 12:
            # hand-computed letter probabilities: exp(log p) sums to
            # 0.3 + 0.6 + 0.05 + 0.05 = 1.0; parsed answer B -> 0.6 / 1.0 = 0.6
            lp = [{"top_logprobs": [
                {"token": "A", "logprob": math.log(0.3)},
                {"token": "B", "logprob": math.log(0.6)},
                {"token": "C", "logprob": math.log(0.05)},
                {"token": "D", "logprob": math.log(0.05)},
            ]}]
        zen.append(_record(i, "zen", "m", correct=i < 12,
                           subject=["physics", "chemistry"][i % 2],
                           conf=0.9 if i < 12 else 0.6,
                           parse_fail=i >= 16, logprobs=lp))
        orr.append(_record(i, "openrouter", "o", correct=i < 10,
                           subject=["physics", "chemistry"][i % 2],
                           conf=0.7 if i < 10 else 0.5))
    (d / "zen__m.jsonl").write_text(
        "\n".join(json.dumps(r) for r in zen) + "\n", encoding="utf-8")
    (d / "openrouter__o.jsonl").write_text(
        "\n".join(json.dumps(r) for r in orr) + "\n", encoding="utf-8")
    (d / "fake__x.jsonl").write_text(
        "\n".join(json.dumps(_record(i, "fake", "x", correct=True,
                                     subject="physics", conf=0.9))
                  for i in range(5)) + "\n", encoding="utf-8")
    return d


@pytest.fixture(scope="session")
def generated(run_dir) -> dict:
    """Run generate_report once for the whole module (figures + bootstraps are slow)."""
    return generate_report(run_dir)


# --------------------------------------------------------------------------
# load_run
# --------------------------------------------------------------------------
def test_load_run_skips_fake_files(run_dir):
    runs = load_run(run_dir)
    # only the two empirical model files; the fake__ file is skipped entirely
    assert set(runs) == {"openrouter::o", "zen::m"}
    assert len(runs["zen::m"]) == 20
    assert len(runs["openrouter::o"]) == 20


# --------------------------------------------------------------------------
# letter_prob_from_logprobs
# --------------------------------------------------------------------------
def test_letter_prob_from_logprobs_hand_case():
    rec = {
        "parsed_answer": "B",
        "logprobs_raw": [{"top_logprobs": [
            {"token": "A", "logprob": math.log(0.3)},
            {"token": "B", "logprob": math.log(0.6)},
            {"token": "C", "logprob": math.log(0.05)},
            {"token": "D", "logprob": math.log(0.05)},
        ]}],
    }
    # probabilities: A 0.3, B 0.6, C 0.05, D 0.05; letters sum to 1.0 > 0.5
    # -> P(B) = 0.6 / 1.0 = 0.6
    assert letter_prob_from_logprobs(rec) == pytest.approx(0.6)


def test_letter_prob_from_logprobs_parenthesized_tokens():
    rec = {
        "parsed_answer": "B",
        "logprobs_raw": [{"top_logprobs": [
            {"token": "A", "logprob": math.log(0.4)},
            {"token": "(B)", "logprob": math.log(0.3)},
            {"token": "C", "logprob": math.log(0.1)},
            {"token": "D", "logprob": math.log(0.2)},
        ]}],
    }
    # "(B)".strip("()") == "B" counts as the letter; total = 1.0 -> 0.3 / 1.0 = 0.3
    assert letter_prob_from_logprobs(rec) == pytest.approx(0.3)


def test_letter_prob_from_logprobs_none_cases():
    # no logprobs / no parsed answer -> None
    assert letter_prob_from_logprobs({"parsed_answer": "B"}) is None
    assert letter_prob_from_logprobs({"parsed_answer": "B", "logprobs_raw": None}) is None
    assert letter_prob_from_logprobs({"logprobs_raw": [{"top_logprobs": [
        {"token": "A", "logprob": 0.0}]}]}) is None
    # letters do not dominate the position (mass 0.4 + 0.05 = 0.45 <= 0.5) -> None
    rec = {"parsed_answer": "A", "logprobs_raw": [{"top_logprobs": [
        {"token": "A", "logprob": math.log(0.4)},
        {"token": "B", "logprob": math.log(0.05)},
    ]}]}
    assert letter_prob_from_logprobs(rec) is None


def test_letter_prob_from_logprobs_clamped_to_one():
    # duplicate B entries accumulate to 1.4 of a 1.4 total -> min(1.0, 1.0) = 1.0
    rec = {"parsed_answer": "B", "logprobs_raw": [{"top_logprobs": [
        {"token": "B", "logprob": math.log(0.7)},
        {"token": "B", "logprob": math.log(0.7)},
    ]}]}
    assert letter_prob_from_logprobs(rec) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# model metrics (hand-computed)
# --------------------------------------------------------------------------
def test_model_metrics_accuracy_hand_computed(run_dir):
    runs = load_run(run_dir)
    zen = model_metrics(runs["zen::m"])
    # 12 correct of 20 records -> lenient 12/20 = 0.6
    assert zen["accuracy_lenient"]["acc"] == pytest.approx(0.6)
    assert zen["accuracy_lenient"]["n"] == 20
    # 12 correct of 16 parsed -> strict 12/16 = 0.75
    assert zen["accuracy_strict"]["acc"] == pytest.approx(0.75)
    assert zen["n_parsed"] == 16
    # 4 parse failures / 20 = 0.2
    assert zen["parse_failure_rate"] == pytest.approx(0.2)

    orr = model_metrics(runs["openrouter::o"])
    assert orr["accuracy_lenient"]["acc"] == pytest.approx(10 / 20)
    assert orr["accuracy_strict"]["acc"] == pytest.approx(10 / 20)
    assert orr["parse_failure_rate"] == 0.0


def test_model_metrics_calibration_self_report_hand_computed(run_dir):
    runs = load_run(run_dir)
    zen = model_metrics(runs["zen::m"])["calibration_self_report"]
    # 12 parsed records at conf 0.9 (all correct) + 4 at conf 0.6 (all wrong):
    #   mean_conf = (12*0.9 + 4*0.6)/16 = 13.2/16 = 0.825
    #   ECE = (12/16)*|0.9-1| + (4/16)*|0.6-0| = 0.075 + 0.15 = 0.225
    #   MCE = max(0.1, 0.6) = 0.6
    #   brier = (12*0.1^2 + 4*0.6^2)/16 = (0.12 + 1.44)/16 = 1.56/16 = 0.0975
    assert zen["n"] == 16
    assert zen["mean_confidence"] == pytest.approx(0.825)
    assert zen["ECE"] == pytest.approx(0.225)
    assert zen["MCE"] == pytest.approx(0.6)
    assert zen["brier"] == pytest.approx(0.0975)

    orr = model_metrics(runs["openrouter::o"])["calibration_self_report"]
    # 10 at 0.7 (correct) + 10 at 0.5 (wrong):
    #   mean = 0.6 ; ECE = 0.5*0.3 + 0.5*0.5 = 0.4 ; MCE = 0.5
    #   brier = (10*0.09 + 10*0.25)/20 = 3.4/20 = 0.17
    assert orr["mean_confidence"] == pytest.approx(0.6)
    assert orr["ECE"] == pytest.approx(0.4)
    assert orr["MCE"] == pytest.approx(0.5)
    assert orr["brier"] == pytest.approx(0.17)


def test_model_metrics_token_prob_channel_hand_computed(run_dir):
    runs = load_run(run_dir)
    zen = model_metrics(runs["zen::m"])["calibration_token_prob"]
    # a single usable logprobs record: letter prob 0.6, outcome wrong (y = 0)
    #   ECE = |0.6 - 0| = 0.6 ; brier = 0.6^2 = 0.36 ; NLL = -ln(0.6) = 0.510826
    assert zen["n"] == 1
    assert zen["ECE"] == pytest.approx(0.6)
    assert zen["brier"] == pytest.approx(0.36)
    assert zen["NLL"] == pytest.approx(-math.log(0.6))
    # openrouter records carry no logprobs -> channel absent
    assert "calibration_token_prob" not in model_metrics(runs["openrouter::o"])


def test_model_metrics_mc_confusion_trace(run_dir):
    runs = load_run(run_dir)
    zen = model_metrics(runs["zen::m"])
    cm = np.array(zen["mc_confusion"])
    # parsed==gold exactly on the 12 correct records -> diagonal = 12
    assert cm.sum() == 16  # 16 parsed MC records
    assert np.trace(cm) == 12


# --------------------------------------------------------------------------
# generate_report: files, tables, figures
# --------------------------------------------------------------------------
def test_generate_report_writes_metrics_json(run_dir, generated):
    report = generated
    out = run_dir / "analysis"
    assert (out / "metrics.json").exists()
    on_disk = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert set(on_disk["models"]) == {"openrouter::o", "zen::m"}
    assert on_disk["models"]["zen::m"]["accuracy_lenient"]["acc"] == pytest.approx(0.6)
    assert on_disk["models"]["zen::m"]["calibration_self_report"]["brier"] == pytest.approx(
        0.0975
    )
    assert on_disk["comparisons"] == report["comparisons"]
    # (a full report == on_disk equality is impossible: empty reliability bins hold
    # NaN, and NaN != NaN after the JSON roundtrip)
    # figures must render headless (Agg): no figure_error recorded
    assert "figure_error" not in report
    figures = sorted(p.name for p in (out / "figures").iterdir())
    for expected in ("reliability.png", "accuracy_ci.png",
                     "subject_heatmap.png", "confusion.png"):
        assert expected in figures


def test_generate_report_model_table_csv(run_dir, generated):
    with open(run_dir / "analysis" / "model_table.csv", newline="",
              encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == [
        "model", "n", "n_parsed", "acc_lenient", "ci_lo", "ci_hi",
        "acc_strict", "parse_fail_rate", "mean_self_conf",
        "ECE_self", "MCE_self", "brier_self", "ECE_tokenprob", "n_tokenprob",
    ]
    # rows sorted by model key: openrouter::o first, then zen::m
    assert [r[0] for r in rows[1:]] == ["openrouter::o", "zen::m"]
    orow, zrow = rows[1], rows[2]
    # openrouter: n=20, parsed=20, lenient 0.5, strict 0.5, pfr 0.0,
    # mean conf 0.6, ECE 0.4, MCE 0.5, brier 0.17, no token channel (n=0)
    assert orow[1:3] == ["20", "20"]
    assert orow[3] == "0.5000"
    assert orow[6] == "0.5000"
    assert orow[7] == "0.0000"
    assert orow[8:12] == ["0.6000", "0.4000", "0.5000", "0.1700"]
    assert orow[12] == "nan" and orow[13] == "0"
    # Wilson CI bounds are valid proportions
    assert 0.0 <= float(orow[4]) <= float(orow[5]) <= 1.0
    # zen: n=20, parsed=16, lenient 0.6, strict 0.75, pfr 0.2,
    # mean conf 0.825, ECE 0.225, MCE 0.6, brier 0.0975, tokenprob n=1
    assert zrow[1:3] == ["20", "16"]
    assert zrow[3] == "0.6000"
    assert zrow[6] == "0.7500"
    assert zrow[7] == "0.2000"
    assert zrow[8:12] == ["0.8250", "0.2250", "0.6000", "0.0975"]
    assert zrow[12] == "0.6000" and zrow[13] == "1"


def test_generate_report_pairwise_table_csv(run_dir, generated):
    with open(run_dir / "analysis" / "pairwise_table.csv", newline="",
              encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["model_a", "model_b", "n_common", "acc_a", "acc_b", "diff",
                       "b", "c", "p_exact", "p_bh", "significant_0.05"]
    assert len(rows) == 2  # header + the single pair
    row = rows[1]
    # Hand computation under the source's LENIENT pairing rule (report.py
    # paired_models_analysis: correctness is True -> 1, anything else incl. parse
    # failure -> 0), with a = openrouter::o, b = zen::m (keys sorted alphabetically):
    #   common items = 20 (every item has a record for both models)
    #   a correct on 00-09 -> acc_a = 10/20 = 0.5
    #   b correct on 00-11 -> acc_b = 12/20 = 0.6 ; diff = 0.5 - 0.6 = -0.1
    #   b01 = #(a=1, b=0) = 0  (a's corrects 00-09 are all also b-correct)
    #   b10 = #(a=0, b=1) = items 10, 11 = 2
    #   p_exact per the implemented formula = min(1, 2 * binomtest(0, 2, 0.5).pvalue)
    #     P(X=0 | Binom(2, 1/2)) = 1/4 -> scipy two-sided p = 1/2 -> doubled = 1.0
    assert row[0] == "openrouter::o" and row[1] == "zen::m"
    assert row[2] == "20"
    assert row[3] == "0.5000"
    assert row[4] == "0.6000"
    assert row[5] == "-0.1000"
    assert row[6:8] == ["0", "2"]
    assert row[8] == "1" and row[9] == "1" and row[10] == "False"


def test_generate_report_by_subject_hand_computed(run_dir, generated):
    report = generated
    # subjects alternate by item index; zen correct on even i in 0..11 and odd i
    # in 0..11 -> physics records = even i (10 items, 6 correct) -> acc 0.6
    zen_phys = report["by_subject"]["zen::m"]["physics"]
    assert zen_phys["n"] == 10
    assert zen_phys["acc"] == pytest.approx(0.6)
    orr_phys = report["by_subject"]["openrouter::o"]["physics"]
    assert orr_phys["n"] == 10
    assert orr_phys["acc"] == pytest.approx(0.5)
