"""Tests for stembench.prompts: rendering and template hashing."""

from __future__ import annotations

import pytest

from stembench.prompts import (
    render_free,
    render_mc,
    template_hash,
    template_text,
)

CHOICES = ["one", "two", "three", "four"]
QUESTION = "What is the derivative of x^2?"


def test_render_mc_english():
    p = render_mc(QUESTION, CHOICES, language="en")
    assert QUESTION in p
    # every choice label with its text, on separate lines
    assert "A. one" in p
    assert "B. two" in p
    assert "C. three" in p
    assert "D. four" in p
    # the required answer contract
    assert "Answer:" in p
    assert "Confidence:" in p
    # English template wording
    assert "Solve the following multiple-choice problem" in p
    assert "Problem:" in p


def test_render_mc_russian():
    p = render_mc("Чему равно 2+2?", CHOICES, language="ru")
    assert "Чему равно 2+2?" in p
    for label, text in zip("ABCD", CHOICES, strict=True):
        assert f"{label}. {text}" in p
    # the contract keywords stay in English so parsers work for both languages
    assert "Answer:" in p
    assert "Confidence:" in p
    # Russian template wording
    assert "Решите следующую задачу" in p
    assert "Задача:" in p


def test_render_mc_custom_labels():
    p = render_mc("q?", ["x", "y"], language="en", labels=["1", "2"])
    assert "1. x" in p and "2. y" in p


def test_render_mc_too_many_choices_raises():
    # labels default to A-F; 7 choices exceed the label pool -> assertion error
    with pytest.raises(AssertionError):
        render_mc("q?", [str(i) for i in range(7)], language="en")


def test_render_free_both_languages():
    en = render_free(QUESTION, language="en")
    assert QUESTION in en
    assert "Answer:" in en and "Confidence:" in en
    assert "Solve the following problem" in en
    assert "Problem:" in en

    ru = render_free("Чему равна скорость света?", language="ru")
    assert "Чему равна скорость света?" in ru
    assert "Answer:" in ru and "Confidence:" in ru
    assert "Задача:" in ru
    # free-response templates do not contain choice labels
    assert "A." not in ru


def test_template_hash_deterministic_and_distinct():
    en_mc = template_hash("mc_answer_confidence_v1", "en")
    # deterministic: same (name, language) -> same digest
    assert template_hash("mc_answer_confidence_v1", "en") == en_mc
    assert len(en_mc) == 64  # sha256 hex digest
    # different languages hash differently (exact text differs)
    assert template_hash("mc_answer_confidence_v1", "ru") != en_mc
    # different templates hash differently
    assert template_hash("free_answer_confidence_v1", "en") != en_mc
    assert template_hash("free_answer_confidence_v1", "en") != template_hash(
        "free_answer_confidence_v1", "ru"
    )


def test_template_text_matches_rendered_template():
    # template_text exposes the exact versioned template text for manifests
    en = template_text("mc_answer_confidence_v1", "en")
    assert "{question}" in en and "{choices}" in en
    ru = template_text("mc_answer_confidence_v1", "ru")
    assert "Задача" in ru
    with pytest.raises(KeyError):
        template_text("no_such_template", "en")
