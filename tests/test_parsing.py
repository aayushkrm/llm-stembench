"""Unit tests for stembench.parsing: MC letter extraction, confidence, numeric, units,
exact answers, Cyrillic normalization.

Expected values are hand-derived in comments next to each assertion.

KNOWN GENUINE SRC BUGS documented by intentionally failing assertions here:
- BUG-P2: scientific notation "3.2×10^4" / "6.022*10^23" yields only the mantissa.
  Cause: NUM_RE (parsing.py:34-41) lists the plain-number alternative BEFORE the
  x10^n alternative; the plain alternative already matches the mantissa alone (its
  exponent part is optional), so the x10^n branch (parsing.py:38, ~104-107) is dead
  code.
- BUG-P3: extract_unit("Answer: 5") returns "N". Cause: parsing.py UNIT_WORDS
  contains single-letter units ("N", "J", "W", ...) matched as substrings of
  ordinary prose; the "n" in the word "Answer" matches "N" (checked first among
  len-1 units).
- BUG-P4: a bare bolded final letter "**D**" is not extracted: pattern3
  (parsing.py ANSWER_PATTERNS) requires a trailing "." or ")" and the last-line
  fallback LABEL_RE cannot start with "*".

NOTE: the capitalized-Cyrillic "Ответ: А" case was broken at the time these tests
were written (homoglyph normalization mangled the keyword before pattern2 ran) and
is fixed in the current working tree (patterns now also run on the raw text); the
tests below keep covering it.
"""

from __future__ import annotations

import pytest

from stembench.parsing import (
    extract_confidence,
    extract_exact_answer,
    extract_mc_answer,
    extract_numeric,
    extract_unit,
    normalize_cyrillic_letters,
)


# --------------------------------------------------------------------------
# MC letter extraction
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,n,expected_letter,expected_method",
    [
        # "Answer: B" -> pattern1 matches "answer" keyword + letter B
        ("Answer: B", 4, "B", "pattern1"),
        # "answer is (C)": pattern1 allows "is", optional parens around the letter
        ("answer is (C)", 4, "C", "pattern1"),
        # "A) foo": pattern3 matches a line starting with the label followed by ")"
        ("A) foo", 4, "A", "pattern3"),
        # patterns 1-3 fail; last non-empty line "B" is a bare letter -> last_line
        ("Reasoning...\nB", 4, "B", "last_line"),
        # lowercase letter, "correct ... answer is:" prefix; IGNORECASE pattern1
        ("The correct answer is: c", 4, "C", "pattern1"),
        # documented pattern3 form "**B)**"
        ("**B)**", 4, "B", "pattern3"),
        # bolded letter after the keyword: pattern1 allows ** around the letter
        ("Answer: **D**", 4, "D", "pattern1"),
        ("Final answer: A", 4, "A", "pattern1"),
        ("ANSWER: B", 4, "B", "pattern1"),
        ("Answer:B", 4, "B", "pattern1"),
        ("My answer is B\nConfidence: 90", 4, "B", "pattern1"),
        ("The answer is **C**", 4, "C", "pattern1"),
        # lowercase Cyrillic keyword survives normalization (lowercase "о" is not in
        # the homoglyph map), so pattern2 fires on "ответ"
        ("ответ: B", 4, "B", "pattern2"),
    ],
)
def test_extract_mc_answer_positive(text, n, expected_letter, expected_method):
    got = extract_mc_answer(text, n_choices=n)
    assert got == (expected_letter, expected_method)


def test_extract_mc_answer_cyrillic_capitalized():
    # "Ответ: А" with Cyrillic О (U+041E) and А (U+0410): pattern2 matches the raw
    # Cyrillic keyword and the Cyrillic letter normalizes to "A".
    # (This was BUG-P1 -- normalization mangled "Ответ" to "Oтвет" before the
    # keyword pattern could fire -- fixed in the current working tree.)
    assert extract_mc_answer("Ответ: А", n_choices=4) == ("A", "pattern2")


def test_extract_mc_answer_bold_letter_alone():
    # BUG-P4 (kept failing): a whole final line "**D**" is a very common model output.
    # Expected ("D", ...) -- either pattern3 or the last-line fallback should get it.
    # Actual: None (pattern3 needs a trailing "." or ")" and LABEL_RE cannot start
    # with "*").
    got = extract_mc_answer("Some reasoning.\n**D**", n_choices=4)
    assert got is not None and got[0] == "D"


@pytest.mark.parametrize(
    "text,n",
    [
        ("", 4),                # empty
        ("   \n  \n", 4),       # whitespace only
        ("no letter here", 4),  # no answer keyword, no bare letter
        ("The answer is maybe", 4),  # word after "answer is" is not a choice letter
        ("Answer: E", 4),       # E is beyond n_choices=4 (valid: A-D)
        ("Answer: C", 2),       # only A-B valid when n_choices=2
    ],
)
def test_extract_mc_answer_none(text, n):
    assert extract_mc_answer(text, n_choices=n) is None


def test_extract_mc_answer_n_choices_edge():
    # with n_choices=2 the exclusive upper bound is 'C', so B is still valid
    assert extract_mc_answer("Answer: B", n_choices=2) == ("B", "pattern1")


# 20 response formats x 4 letters (A-D): every format must yield the letter.
FORMATS = [
    "Answer: {L}",
    "answer is ({L})",
    "The correct answer is: {l}",
    "Answer: **{L}**",
    "Final answer: {L}",
    "ответ: {L}",                              # lowercase Cyrillic keyword (works)
    "Ответ: {L}",                              # capitalized Cyrillic keyword (BUG-P1)
    "{L}) foo",
    "The answer is {L}.",
    "**{L})**",
    "answer: {l}",
    "My answer is {L}\nConfidence: 90",
    "I think the answer is **{L}**",
    "ANSWER: {L}",
    "Answer:{L}",
    "Ответ: {L}.",
    "so the answer is ({l})",
    "Answer: ({L})",
    "Reasoning text.\nAnswer: {L}\nConfidence: 85",
    "Sure.\n{L}.",
]


@pytest.mark.parametrize("letter", ["A", "B", "C", "D"])
def test_extract_mc_answer_format_sweep(letter):
    failures = []
    for fmt in FORMATS:
        text = fmt.format(L=letter, l=letter.lower())
        got = extract_mc_answer(text, n_choices=4)
        if got is None or got[0] != letter:
            failures.append((text, got))
    assert failures == []


def test_extract_mc_answer_format_sweep_cyrillic_capitalized():
    # capitalized Cyrillic keyword, with and without a trailing period (regression
    # guard for the former BUG-P1)
    for letter in "ABCD":
        assert extract_mc_answer(f"Ответ: {letter}", n_choices=4) is not None
        assert extract_mc_answer(f"Ответ: {letter}.", n_choices=4) is not None


# --------------------------------------------------------------------------
# Confidence extraction
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        # 87 / 100 -> 0.87
        ("Confidence: 87", 0.87),
        # "100%" -> 100/100 = 1.0
        ("Confidence: 100%", 1.0),
        # Russian keyword, 50 -> 0.5
        ("уверенность: 50", 0.5),
        ("Уверенность: 50", 0.5),
        # trailing % with a space is still accepted: 100 -> 1.0
        ("Confidence: 100 %", 1.0),
        # confidence inside a longer response
        ("Answer: B\nConfidence: 75", 0.75),
    ],
)
def test_extract_confidence_positive(text, expected):
    assert extract_confidence(text) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text",
    [
        "Confidence: 150",     # 150 > 100 is malformed per the 0-100 contract -> None
        "Confidence: 1500",    # malformed
        "Answer: B",           # no confidence line
        "I am fairly sure.",   # no confidence keyword
        "",                    # empty
        None,                  # None input guarded
    ],
)
def test_extract_confidence_none(text):
    assert extract_confidence(text) is None


# --------------------------------------------------------------------------
# Numeric extraction
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("Answer: 3.14", 3.14),
        # RU decimal comma: "3,14" -> 3.14
        ("Answer: 3,14", 3.14),
        # space-separated thousands: 1 234 567 -> 1234567
        ("Answer: 1 234 567", 1234567.0),
        # comma thousands with decimal point: 1,234,567.8 -> 1234567.8
        ("Answer: 1,234,567.8", 1234567.8),
        # scientific notation: -5.2e-3 = -0.0052
        ("Answer: -5.2e-3", -0.0052),
    ],
)
def test_extract_numeric_positive(text, expected):
    got = extract_numeric(text)
    assert got is not None
    assert got[0] == pytest.approx(expected)


def test_extract_numeric_scientific_times_ten_bug():
    # BUG-P2 (kept failing): "3.2×10^4" must give 3.2 * 10^4 = 32000.
    # Actual: (3.2, "3.2") -- the plain-number alternative in NUM_RE shadows the
    # x10^n alternative (see module docstring).
    got = extract_numeric("Answer: 3.2×10^4")
    assert got is not None and got[0] == pytest.approx(32000.0)


def test_extract_numeric_scientific_star_ten_bug():
    # BUG-P2 (kept failing): "6.022*10^23" must give 6.022e23.
    got = extract_numeric("Answer: 6.022*10^23")
    assert got is not None and got[0] == pytest.approx(6.022e23)


def test_extract_numeric_no_number():
    assert extract_numeric("just text, no numbers here") is None
    assert extract_numeric("") is None


def test_extract_numeric_prefers_answer_segment_over_confidence():
    # The Answer segment ("25") must win; the "Confidence: 90" number and the
    # "0-100" contract text after it must not be picked up as the answer.
    got = extract_numeric("Answer: 25\nConfidence: 90")
    assert got == (25.0, "25")
    got2 = extract_numeric("The velocity is 12 m/s.\nAnswer: 12\nConfidence: 95")
    assert got2 == (12.0, "12")


def test_extract_numeric_returns_raw_match():
    # the second tuple element is the matched raw string (auditable parse)
    assert extract_numeric("Answer: 3,14")[1] == "3,14"
    assert extract_numeric("Answer: -5.2e-3")[1] == "-5.2e-3"


# --------------------------------------------------------------------------
# Unit extraction
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("The speed is 5 m/s", "m/s"),      # "m/s" is a listed unit
        ("M = 44 g/mol", "g/mol"),         # longest-match: g/mol beats single g
        ("T = 25 °C", "°C"),
        ("3.14159", ""),                    # no letters at all -> no unit
        ("1234.5", ""),
    ],
)
def test_extract_unit_positive(text, expected):
    assert extract_unit(text) == expected


def test_extract_unit_no_false_positive_in_answer_word():
    # BUG-P3 (kept failing): "Answer: 5" states no unit, so extract_unit must be "".
    # Actual: "N" -- the single-letter unit "N" substring-matches the "n" in "Answer".
    assert extract_unit("Answer: 5") == ""


def test_extract_unit_finds_stated_unit_despite_word_letters():
    # BUG-P3 (kept failing): the stated unit here is "m"; it must be found.
    # Actual: "N" (the "n" of "Answer" matches the len-1 unit "N" first).
    assert extract_unit("Answer: 5 m") == "m"


# --------------------------------------------------------------------------
# Exact-answer extraction
# --------------------------------------------------------------------------
def test_extract_exact_answer_prefers_answer_segment():
    # content after "Answer:" up to end of line, with a trailing Confidence part stripped
    assert extract_exact_answer("blah blah\nAnswer: 42 kg\nConfidence: 80") == "42 kg"
    assert extract_exact_answer("The answer is Mars.\nConfidence: 90") == "Mars."


def test_extract_exact_answer_falls_back_to_last_line():
    # no "Answer:" marker -> last non-empty line
    assert extract_exact_answer("first line\nlast non-empty line") == "last non-empty line"
    assert extract_exact_answer("only line") == "only line"


def test_extract_exact_answer_empty():
    assert extract_exact_answer("") is None
    assert extract_exact_answer("   \n\n") is None


# --------------------------------------------------------------------------
# Cyrillic homoglyph normalization
# --------------------------------------------------------------------------
def test_normalize_cyrillic_letters_mapping():
    # The four MC-relevant homoglyphs map to Latin look-alikes:
    # Cyrillic А (U+0410) -> Latin A, В (U+0412) -> B, С (U+0421) -> C, Е (U+0415) -> E
    assert normalize_cyrillic_letters("А") == "A"
    assert normalize_cyrillic_letters("В") == "B"
    assert normalize_cyrillic_letters("С") == "C"
    assert normalize_cyrillic_letters("Е") == "E"
    # unaffected: Latin letters and Cyrillic letters with no homoglyph (Д, Г)
    assert normalize_cyrillic_letters("ABCD") == "ABCD"
    assert normalize_cyrillic_letters("ДГ") == "ДГ"
