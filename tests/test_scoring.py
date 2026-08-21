"""Tests for stembench.scoring: MC, exact, numeric (with unit policy), parse-failure rate.

All expected values hand-derived in comments.

History: BUG-P3/BUG-S1 (unit letters matched inside words, e.g. "N" in "Answer")
were fixed with word-boundary extraction; the unit check is now answer-scoped
(``Answer:`` segment or final line) with membership matching, so units stated only
in reasoning text can no longer fail a numerically correct answer.
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
    # Realistic response "Answer: 5.2": no unit stated in the answer region, so per
    # the documented policy (no unit stated -> warn, not failure) it scores correct
    # with unit=none.
    ok, raw, method = score_numeric("Answer: 5.2", gold=5.2, abs_tol=1e-9, require_unit="s")
    assert ok is True
    assert raw == "5.2"
    assert method == "numeric(unit=none)"


# --------------------------------------------------------------------------
# Answer-scoped unit policy (S2-E1 regression: units in reasoning text must not
# fail numerically correct answers)
# --------------------------------------------------------------------------
def test_score_numeric_unit_in_reasoning_ignored():
    # The question/reasoning states c = 4200 J/(kg·K); the answer region is a bare
    # number. The old whole-response extraction captured "J/(kg·K)" and failed this.
    resp = (
        "Q = mcΔT where c = 4200 J/(kg·K), m = 1.9 kg, ΔT = 50 K\n"
        "Q = 1.9 × 4200 × 50 = 399000\n"
        "Answer: 399000"
    )
    ok, raw, method = score_numeric(resp, gold=399000.0, rel_tol=0.02, require_unit="J")
    assert ok is True
    assert raw == "399000"
    assert method == "numeric(unit=none)"


def test_score_numeric_required_unit_among_several_tokens():
    # Model states the value in J first and a kJ equivalent in parens: the required
    # unit J is among the answer-region tokens -> correct.
    ok, _, method = score_numeric(
        "Answer: 399,000 J (399 kJ)", gold=399000.0, rel_tol=0.02, require_unit="J",
    )
    assert ok is True
    assert "J" in method and "kJ" in method


def test_score_numeric_ru_unit_alias():
    # Russian unit spelling in the answer region maps to the canonical symbol.
    ok, _, method = score_numeric(
        "Answer: 3.58 моль", gold=3.57973, rel_tol=0.02, require_unit="mol",
    )
    assert ok is True
    assert "mol" in method


def test_score_numeric_unit_separator_normalization():
    # "J/(kg·K)" in the response vs required "J/(kg*K)" -> same unit.
    ok, _, _ = score_numeric(
        "Answer: 4200 J/(kg·K)", gold=4200.0, rel_tol=0.01, require_unit="J/(kg*K)",
    )
    assert ok is True


def test_score_numeric_wrong_unit_in_answer_region_still_fails():
    # Unit genuinely stated in the answer region but wrong -> failure stands.
    ok, _, method = score_numeric(
        "Given p = 376.3 kPa, V = 26.1 L, T = 330 K\nn = 3.58\nAnswer: 3.58 kPa",
        gold=3.58, rel_tol=0.02, require_unit="mol",
    )
    assert ok is False
    assert "kPa" in method


def test_score_numeric_molar_M_equals_mol_per_L():
    # S2-E1 regression: models answer "0.8 M" (molar) for items whose reference
    # unit is mol/L — the same concentration unit.
    ok, _, _ = score_numeric("Answer: 0.8 M", gold=0.8, abs_tol=1e-9, require_unit="mol/L")
    assert ok is True


def test_score_numeric_momentum_compound_unit_separators():
    # S2-E1 regression: "252 kg·m/s" vs required "kg*m/s" — same momentum unit.
    ok, _, _ = score_numeric("Answer: 252 kg·m/s", gold=252.0, abs_tol=1e-9, require_unit="kg*m/s")
    assert ok is True


def test_score_numeric_spaced_compound_unit():
    # Spaces inside a compound unit ("mol / L") must not split it.
    ok, _, _ = score_numeric("Answer: 2 mol / L", gold=2.0, abs_tol=1e-9, require_unit="mol/L")
    assert ok is True


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
