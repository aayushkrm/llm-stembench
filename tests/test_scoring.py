"""Tests for stembench.scoring: MC, exact, numeric (with unit policy), parse-failure rate.

All expected values hand-derived in comments.

KNOWN GENUINE SRC BUG documented by an intentionally failing assertion:
- BUG-S1 (consequence of BUG-P3 in stembench/parsing.py): scoring.py:63-71 documents
  "empty parsed unit -> warn flag, not failure", but extract_unit("Answer: 5.2")
  returns "N" (the "n" of the word "Answer" matches the single-letter unit "N"), so
  score_numeric marks a value-correct, unit-omitting answer WRONG whenever the
  reference declares any unit.
"""

from __future__ import annotations

import pytest

from stembench.scoring import parse_failure_rate, score_exact, score_mc, score_numeric


# --------------------------------------------------------------------------
# Multiple choice
# --------------------------------------------------------------------------
def test_score_mc_correct():
    # gold_index=1 -> gold letter "B"; parsed "B" -> correct
    ok, letter, method = score_mc("Answer: B", gold_index=1, n_choices=4)
    assert ok is True
    assert letter == "B"
    assert method == "pattern1"


def test_score_mc_incorrect():
    # gold letter "B" vs parsed "C" -> incorrect
    ok, letter, _ = score_mc("The correct answer is: c", gold_index=1, n_choices=4)
    assert ok is False
    assert letter == "C"


def test_score_mc_parse_failure():
    # unparseable output -> (None, "", "none"): counted as parse failure, not wrong
    assert score_mc("I cannot decide.", gold_index=2, n_choices=4) == (None, "", "none")
    assert score_mc("", gold_index=2, n_choices=4) == (None, "", "none")


def test_score_mc_respects_n_choices():
    # letter D is beyond n_choices=2 -> unparseable, never silently scored
    ok, letter, method = score_mc("Answer: D", gold_index=1, n_choices=2)
    assert ok is None
    assert letter == ""
    assert method == "none"


# --------------------------------------------------------------------------
# Exact match
# --------------------------------------------------------------------------
def test_score_exact_case_insensitive():
    # normalize_exact lowercases: "Photosynthesis" == "PHOTOSYNTHESIS"
    ok, parsed, _ = score_exact("Answer: PHOTOSYNTHESIS", "Photosynthesis")
    assert ok is True
    assert parsed == "PHOTOSYNTHESIS"


def test_score_exact_punctuation_insensitive():
    # normalize_exact strips quotes « » " ' ` * _ # $ and trailing . ; !
    # '"Photosynthesis."' -> photosynthesis == photosynthesis
    ok, _, _ = score_exact('Answer: "Photosynthesis."\nConfidence: 80', "photosynthesis")
    assert ok is True
    # leading/trailing whitespace is normalized too
    assert score_exact("Answer:   42 kg  ", "42 kg")[0] is True


def test_score_exact_alternatives():
    # primary gold "H2O" does not match, but the acceptable alternative "water" does
    ok, parsed, _ = score_exact("Answer: Water", "H2O", alternatives=["water"])
    assert ok is True
    assert parsed == "Water"
    # no alternatives and mismatch -> incorrect
    assert score_exact("Answer: Water", "H2O")[0] is False


def test_score_exact_parse_failure():
    assert score_exact("", "H2O") == (None, "", "none")


# --------------------------------------------------------------------------
# Numeric
# --------------------------------------------------------------------------
def test_score_numeric_rel_tolerance_accept():
    # gold 9.81, rel_tol 0.02: |10 - 9.81| = 0.19 <= 0.02 * 9.81 = 0.1962 -> accept
    ok, raw, method = score_numeric("Answer: 10", gold=9.81, rel_tol=0.02)
    assert ok is True
    assert raw == "10"
    assert method.startswith("numeric(")


def test_score_numeric_rel_tolerance_reject():
    # |12 - 9.81| = 2.19 > 0.1962 -> reject
    ok, _, _ = score_numeric("Answer: 12", gold=9.81, rel_tol=0.02)
    assert ok is False


def test_score_numeric_abs_tolerance():
    # abs_tol 0.5: |9.5 - 9.81| = 0.31 <= 0.5 -> accept; |9.0 - 9.81| = 0.81 > 0.5 -> reject
    assert score_numeric("Answer: 9.5", gold=9.81, abs_tol=0.5)[0] is True
    assert score_numeric("Answer: 9.0", gold=9.81, abs_tol=0.5)[0] is False


def test_score_numeric_default_tolerance():
    # no tolerances given -> math.isclose default rel_tol=1e-6:
    # "9.810001" vs 9.81 differs by 1e-6 <= 1e-6*9.81 -> accept
    assert score_numeric("Answer: 9.810001", gold=9.81)[0] is True
    # "9.82" differs by 0.01 -> reject
    assert score_numeric("Answer: 9.82", gold=9.81)[0] is False


def test_score_numeric_wrong_unit_enforced():
    # value 5 is right but the stated unit ("m") mismatches the required "s" -> wrong.
    # (Outcome False holds today; note the extracted unit string is actually "N" due
    # to BUG-P3 in parsing.extract_unit, but the mismatch verdict is the same.)
    ok, _, _ = score_numeric("Answer: 5 m", gold=5.0, abs_tol=1e-9, require_unit="s")
    assert ok is False


def test_score_numeric_matching_unit_accepted():
    # unit "s" stated and required -> correct. (The response is a bare value+unit so
    # that extract_unit finds "s"; the word "Answer" itself would trigger BUG-P3.)
    ok, raw, method = score_numeric("5 s", gold=5.0, abs_tol=1e-9, require_unit="s")
    assert ok is True
    assert raw == "5"
    assert method == "numeric(unit=s)"


def test_score_numeric_empty_unit_still_scored():
    # A response that is a bare number (no prose letters at all) yields an empty
    # parsed unit: per the documented policy the item is still scored and the
    # method notes unit=none.
    ok, raw, method = score_numeric("5.2", gold=5.2, abs_tol=1e-9, require_unit="s")
    assert ok is True
    assert raw == "5.2"
    assert method == "numeric(unit=none)"


def test_score_numeric_empty_unit_policy_with_answer_prefix_bug():
    # BUG-S1 (kept failing): same policy for a realistic response "Answer: 5.2" --
    # the model states no unit, so per scoring.py's documented rule (empty parsed
    # unit -> warn, not failure) this must be scored correct with unit=none.
    # Actual: (False, "5.2", "numeric(unit=N)") because extract_unit finds the
    # letter "N" inside the word "Answer" (BUG-P3) and the mismatch check fires.
    ok, raw, method = score_numeric("Answer: 5.2", gold=5.2, abs_tol=1e-9, require_unit="s")
    assert ok is True
    assert raw == "5.2"
    assert method == "numeric(unit=none)"


def test_score_numeric_parse_failure():
    assert score_numeric("no numbers here", gold=1.0, abs_tol=1e-9) == (None, "", "none")


# --------------------------------------------------------------------------
# Parse-failure rate
# --------------------------------------------------------------------------
def test_parse_failure_rate():
    # 1 of 3 records has correctness None -> 1/3
    records = [{"correctness": True}, {"correctness": False}, {"correctness": None}]
    assert parse_failure_rate(records) == pytest.approx(1 / 3)


def test_parse_failure_rate_zero_and_empty():
    assert parse_failure_rate([{"correctness": True}, {"correctness": False}]) == 0.0
    assert parse_failure_rate([]) == 0.0
