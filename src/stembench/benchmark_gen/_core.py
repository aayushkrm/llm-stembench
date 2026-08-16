"""Shared infrastructure for the deterministic bilingual STEM benchmark generators.

Conventions (binding for the whole package)
-------------------------------------------
Determinism
    Every random draw comes from :func:`derive_rng`: a numpy PCG64 generator
    seeded with the first 8 bytes of sha256("seed|part1|part2|...") modulo 2**32.
    The builtin ``hash()`` is deliberately NOT used (it is process-salted for
    ``str``).  Given one global seed, every parameter value, distractor pick and
    MC shuffle is reproducible bit-for-bit.

Decimal separator
    "." is used in BOTH languages (prompts, options, answers, solutions).
    Russian typography normally uses "," but the dataset pins "." everywhere so
    that canonical answers and choice texts are directly comparable across the
    two language variants.

Units
    SI symbols are never translated: "m/s", "J", "mol/L", "g/mol", "RUB" appear
    verbatim in Russian and English texts alike.

Bilingual pairing
    One parameter set per pair; the Russian and English texts are parallel
    hand-authored templates.  People are transliterated (Анна/Anna) so the
    semantics stay aligned.
"""

from __future__ import annotations

import hashlib
import math
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from stembench.schemas import (
    AnswerType,
    BenchmarkItem,
    Choice,
    Difficulty,
    Split,
    Subject,
    Tolerance,
    VerifierRecord,
)

DEFAULT_SEED = 20260817
VERSION = "0.1.0-candidate"

# Generation-time anti-duplicate guard: a new question must stay below this
# character-3gram Jaccard similarity against every previously emitted question
# of the SAME topic key and subject (in either language).
GEN_SIMILARITY_CAP = 0.72
# Hard QC gate for near-duplicates across the whole subject+language group.
QC_SIMILARITY_CAP = 0.80


# --------------------------------------------------------------------------- #
# Deterministic randomness
# --------------------------------------------------------------------------- #
def derive_seed(seed: int, *parts: object) -> int:
    """Stable 32-bit seed: sha256 of pipe-joined parts, first 8 bytes."""
    payload = "|".join([str(seed)] + [str(p) for p in parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def derive_rng(seed: int, *parts: object) -> np.random.Generator:
    """numpy PCG64 generator deterministically derived from (seed, parts)."""
    return np.random.default_rng(derive_seed(seed, *parts))


# --------------------------------------------------------------------------- #
# Numbers and formatting
# --------------------------------------------------------------------------- #
def fmt(x: float | int) -> str:
    """Canonical decimal string: '.' separator, at most 6 significant digits.

    Integers render without a decimal point; trailing zeros are stripped.
    """
    xf = float(x)
    if not math.isfinite(xf):
        raise ValueError(f"non-finite number: {x}")
    if xf == int(xf) and abs(xf) < 1e15:
        return str(int(xf))
    exponent = math.floor(math.log10(abs(xf)))
    decimals = max(0, min(6, 5 - exponent))
    s = f"{round(xf, decimals):.{decimals}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s in ("", "-0"):
        s = "0"
    return s


def sig_digits(s: str) -> int:
    """Count significant decimal digits of a plain number string."""
    body = s.strip().lstrip("+-")
    if "e" in body.lower():
        body = body.lower().split("e")[0]
    body = body.replace(".", "")
    stripped = body.lstrip("0")
    return len(stripped) if stripped else 1


def frac_str(n: int, d: int) -> str:
    """Reduced fraction 'n/d' with 'd == 1' collapsing to a plain integer."""
    if d == 0:
        raise ValueError("zero denominator")
    g = math.gcd(abs(n), abs(d))
    n2, d2 = n // g, d // g
    if d2 < 0:
        n2, d2 = -n2, -d2
    return str(n2) if d2 == 1 else f"{n2}/{d2}"


def poly_str(terms: list[tuple[int, str]]) -> str:
    """Render polynomial terms like [(1,'x^2'), (-5,'x'), (6,'')] -> 'x^2 - 5x + 6'."""
    out: list[str] = []
    for coef, sym in terms:
        if coef == 0:
            continue
        mag = abs(coef)
        body = sym if (mag == 1 and sym) else f"{mag}{sym}"
        if not out:
            out.append(("-" if coef < 0 else "") + body)
        else:
            out.append(("- " if coef < 0 else "+ ") + body)
    return " ".join(out) if out else "0"


def num_tolerance(value: float) -> Tolerance:
    """rel = 2%% for rounded results; abs for (near-)zero values."""
    if abs(value) < 1e-9:
        return Tolerance(abs=0.01)
    return Tolerance(rel=0.02)


# --------------------------------------------------------------------------- #
# People and Russian morphology
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Person:
    en: str
    ru_nom: str  # nominative: Анна
    ru_gen: str  # genitive: Анны
    gender: str  # "m" | "f"


PEOPLE: tuple[Person, ...] = (
    Person("Anna", "Анна", "Анны", "f"),
    Person("Sergey", "Сергей", "Сергея", "m"),
    Person("Maria", "Мария", "Марии", "f"),
    Person("Dmitry", "Дмитрий", "Дмитрия", "m"),
    Person("Elena", "Елена", "Елены", "f"),
    Person("Ivan", "Иван", "Ивана", "m"),
    Person("Olga", "Ольга", "Ольги", "f"),
    Person("Pavel", "Павел", "Павла", "m"),
    Person("Natalia", "Наталья", "Натальи", "f"),
    Person("Boris", "Борис", "Бориса", "m"),
    Person("Tatiana", "Татьяна", "Татьяны", "f"),
    Person("Kirill", "Кирилл", "Кирилла", "m"),
)


def ru_past(gender: str, m_form: str, f_form: str) -> str:
    return m_form if gender == "m" else f_form


def ru_plural(n: int, one: str, few: str, many: str) -> str:
    """Russian plural agreement: 1 яблоко / 2 яблока / 5 яблок."""
    if n % 100 in (11, 12, 13, 14):
        return many
    if n % 10 == 1:
        return one
    if n % 10 in (2, 3, 4):
        return few
    return many


def ru_was(gender: str) -> str:
    return ru_past(gender, "был", "была")


# --------------------------------------------------------------------------- #
# Text normalization / near-duplicate detection
# --------------------------------------------------------------------------- #
def normalize_text(s: str) -> str:
    """NFKC + lowercase + whitespace collapse (digits and punctuation kept)."""
    return " ".join(unicodedata.normalize("NFKC", s).lower().split())


def char_trigrams(s: str) -> set[str]:
    n = normalize_text(s)
    return {n[i : i + 3] for i in range(len(n) - 2)} if len(n) >= 3 else {n}


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


# --------------------------------------------------------------------------- #
# Pair drafts and MC machinery
# --------------------------------------------------------------------------- #
@dataclass
class Row:
    """One generation task: `count` pairs of (topic_key, difficulty, answer_type)."""

    topic_key: str
    difficulty: Difficulty
    count: int
    answer_type: AnswerType


@dataclass
class PairDraft:
    """Language-independent core of one bilingual pair (pre item construction)."""

    subject: Subject
    topic: str  # display topic
    topic_key: str  # verifier registry key
    difficulty: Difficulty
    answer_type: AnswerType
    canonical: str  # value string for numeric/exact; becomes the letter for MC
    numeric_value: Optional[float] = None
    units: str = ""
    question_en: str = ""
    question_ru: str = ""
    solution_en: str = ""
    solution_ru: str = ""
    mc_en: tuple[str, ...] = ()  # exactly 4 for MC, CORRECT ANSWER FIRST
    mc_ru: tuple[str, ...] = ()
    distractor_tags: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)
    rubric_en: str = ""
    rubric_ru: str = ""
    acceptable_alternatives: tuple[str, ...] = ()


def _rel_diff(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / denom


def pick_distractors(
    rng: np.random.Generator,
    value: float,
    candidates: list[tuple[float, str]],
    n: int = 3,
) -> list[tuple[float, str]]:
    """Choose `n` distinct plausible-error distractors for a numeric MC item.

    ``candidates`` holds (wrong value, perturbation tag) pairs produced by the
    topic generator (sign error, factor-of-2, unit slip, off-by-one, ...).
    """
    value_f = float(value)
    usable: list[tuple[float, str]] = []
    for cand, tag in candidates:
        cf = float(cand)
        if not math.isfinite(cf):
            continue
        if _rel_diff(cf, value_f) < 0.05 and abs(cf - value_f) < 0.5:
            continue  # too close to the canonical answer
        if any(_rel_diff(cf, prev) < 0.05 and abs(cf - prev) < 0.5 for prev, _ in usable):
            continue  # duplicate of an already picked distractor
        usable.append((cf, tag))
    if len(usable) < n:
        raise ValueError(f"not enough distinct distractors for value={value}: {usable}")
    idx = rng.choice(len(usable), size=n, replace=False)
    return [usable[int(i)] for i in idx]


def apply_mc(draft: PairDraft, rng: np.random.Generator) -> None:
    """Shuffle MC options in place: uniform correct letter, deterministic order.

    The correct option is placed at a uniformly drawn position in A..D; the
    three distractors fill the remaining slots in a drawn permutation.
    """
    if draft.answer_type != AnswerType.MC:
        raise ValueError("apply_mc on non-MC draft")
    if len(draft.mc_en) != 4 or len(draft.mc_ru) != 4:
        raise ValueError("MC draft needs exactly 4 options")
    pos = int(rng.integers(0, 4))
    perm = [int(p) for p in rng.permutation([1, 2, 3])]
    order = perm[:pos] + [0] + perm[pos:]
    draft.mc_en = tuple(draft.mc_en[i] for i in order)
    draft.mc_ru = tuple(draft.mc_ru[i] for i in order)
    draft.distractor_tags = tuple(draft.distractor_tags[i - 1] for i in order if i != 0)
    letter = "ABCD"[order.index(0)]
    draft.canonical = letter
    draft.params["correct_letter"] = letter
    draft.params["mc_values"] = [
        (draft.params["choice_values"][i] if isinstance(draft.params.get("choice_values"), list) else None)
        for i in order
    ]


def final_en(answer: str, units: str) -> str:
    tail = f" {units}" if units else ""
    return f"Answer: {answer}{tail}."


def final_ru(answer: str, units: str) -> str:
    tail = f" {units}" if units else ""
    return f"Ответ: {answer}{tail}."


def sol_en(steps: list[str], answer: str, units: str) -> str:
    body = "\n".join(f"{i}) {s}" for i, s in enumerate(steps, 1))
    return f"{body}\n{final_en(answer, units)}"


def sol_ru(steps: list[str], answer: str, units: str) -> str:
    body = "\n".join(f"{i}) {s}" for i, s in enumerate(steps, 1))
    return f"{body}\n{final_ru(answer, units)}"


def default_alternatives(answer_type: AnswerType, canonical: str) -> tuple[str, ...]:
    """Language-neutral extras: comma-decimal variant for RU habits, RU yes/no."""
    if answer_type == AnswerType.NUMERIC and "." in canonical:
        return (canonical.replace(".", ","),)
    if answer_type == AnswerType.EXACT:
        table = {"yes": ("да",), "no": ("нет",), "real": ("действительное",), "virtual": ("мнимое",)}
        return table.get(canonical, ())
    return ()


def make_items(
    draft: PairDraft,
    pair_id: str,
    seed: int,
    split: Split,
    verifiers: list[VerifierRecord],
) -> tuple[BenchmarkItem, BenchmarkItem]:
    """Assemble the (en, ru) BenchmarkItem pair from a finalized draft."""
    common = dict(
        pair_id=pair_id,
        subject=draft.subject,
        topic=draft.topic,
        difficulty=draft.difficulty,
        answer_type=draft.answer_type,
        canonical_answer=draft.canonical,
        acceptable_alternatives=list(draft.acceptable_alternatives) or list(
            default_alternatives(draft.answer_type, draft.canonical)
        ),
        tolerance=num_tolerance(draft.numeric_value) if draft.answer_type == AnswerType.NUMERIC else None,
        units=draft.units,
        provenance="original_procedural",
        license="CC-BY-4.0",
        author="LLM-STEMBench procedural generator",
        creation_method=f"template+parameters, seeded (seed={seed})",
        translator="parallel templates authored bilingually",
        annotation_version="0",
        split=split,
        contamination_notes="generated 2026-08-17; parameters post-cutoff",
        verifier=verifiers,
    )
    choices_en = (
        [Choice(label="ABCD"[i], text=t) for i, t in enumerate(draft.mc_en)]
        if draft.answer_type == AnswerType.MC
        else []
    )
    choices_ru = (
        [Choice(label="ABCD"[i], text=t) for i, t in enumerate(draft.mc_ru)]
        if draft.answer_type == AnswerType.MC
        else []
    )
    en = BenchmarkItem(
        item_id=f"{pair_id}-en",
        language="en",
        question=draft.question_en,
        choices=choices_en,
        solution=draft.solution_en,
        difficulty_rubric=draft.rubric_en,
        **common,
    )
    ru = BenchmarkItem(
        item_id=f"{pair_id}-ru",
        language="ru",
        question=draft.question_ru,
        choices=choices_ru,
        solution=draft.solution_ru,
        difficulty_rubric=draft.rubric_ru,
        **common,
    )
    return en, ru


@dataclass
class PairBundle:
    """A finalized bilingual pair plus build metadata."""

    pair_id: str
    subject: Subject
    topic: str
    topic_key: str
    difficulty: Difficulty
    answer_type: AnswerType
    canonical: str
    split: Split
    params: dict[str, Any]
    distractor_tags: tuple[str, ...]
    en: BenchmarkItem
    ru: BenchmarkItem


class GenContext:
    """Carries the global seed and the anti-duplicate registry for one build."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._seen: dict[tuple[str, str, str], list[set[str]]] = {}

    def rng(self, subject: str, topic_key: str, idx: int, salt: str = "") -> np.random.Generator:
        return derive_rng(self.seed, subject, topic_key, idx, salt)

    def too_similar(self, subject: str, topic_key: str, lang: str, question: str) -> bool:
        """True if the question is too close to a same-topic predecessor."""
        bucket = self._seen.get((subject, topic_key, lang), ())
        tri = char_trigrams(question)
        return any(jaccard(tri, prev) >= GEN_SIMILARITY_CAP for prev in bucket)

    def record(self, subject: str, topic_key: str, lang: str, question: str) -> None:
        self._seen.setdefault((subject, topic_key, lang), []).append(char_trigrams(question))


GeneratorFn = Callable[[np.random.Generator, int, AnswerType], PairDraft]
