"""Quality-control gates for the generated bilingual benchmark.

The build runs every gate before any file is written; a single error makes the
build exit nonzero ("no partial dataset").  Near-duplicate detection uses
character 3-gram Jaccard similarity of the normalized question text within
each (subject, language) group; the generation-time guard keeps same-topic
questions further apart (0.72), the hard QC gate is 0.80.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from stembench.schemas import AnswerType, BenchmarkItem, Language, Split

from ._core import QC_SIMILARITY_CAP, PairBundle, jaccard, normalize_text, sig_digits, word_trigrams

CYRILLIC = re.compile(r"[а-яА-ЯёЁ]")

# Bounds for the distribution gates (fractions of pairs/items).
MIN_PAIRS_TOTAL = 620
MIN_ITEMS_TOTAL = 1240
SUBJECT_MIN_PAIRS = {"math": 230, "physics": 190, "chemistry": 170}
DIFFICULTY_BOUNDS = {"school": (0.28, 0.58), "university": (0.28, 0.62), "olympiad": (0.05, 0.25)}
ANSWER_TYPE_BOUNDS = {"mc": (0.36, 0.52), "numeric": (0.33, 0.48), "exact": (0.08, 0.22)}
LETTER_SHARE_BOUNDS = (0.17, 0.33)


@dataclass
class QCResult:
    errors: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def _parse_number(text: str) -> float | None:
    t = text.strip().split()[0] if text.strip() else ""
    try:
        return float(t)
    except ValueError:
        try:
            num, den = t.split("/")
            return float(num) / float(den)
        except Exception:
            return None


def _sig_ok(canonical: str) -> bool:
    return sig_digits(canonical) <= 6


def qc_schema(items: list[BenchmarkItem], errors: list[str]) -> None:
    for it in items:
        try:
            BenchmarkItem.model_validate(it.model_dump())
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"schema round-trip failed for {it.item_id}: {exc}")


def qc_pairing(bundles: list[PairBundle], errors: list[str]) -> None:
    by_pair: dict[str, list[BenchmarkItem]] = defaultdict(list)
    for b in bundles:
        by_pair[b.en.pair_id].append(b.en)
        by_pair[b.ru.pair_id].append(b.ru)
    for pid, group in by_pair.items():
        if len(group) != 2:
            errors.append(f"{pid}: expected 2 items, found {len(group)}")
            continue
        langs = {g.language for g in group}
        if langs != {Language.EN, Language.RU}:
            errors.append(f"{pid}: language set is {langs}")
            continue
        en = next(g for g in group if g.language == Language.EN)
        ru = next(g for g in group if g.language == Language.RU)
        for attr in (
            "canonical_answer", "answer_type", "subject", "topic", "difficulty",
            "units", "split", "acceptable_alternatives",
        ):
            if getattr(en, attr) != getattr(ru, attr):
                errors.append(f"{pid}: {attr} differs between en/ru")
        if (en.tolerance.model_dump() if en.tolerance else None) != (
            ru.tolerance.model_dump() if ru.tolerance else None
        ):
            errors.append(f"{pid}: tolerance differs between en/ru")
        if [v.method for v in en.verifier] != [v.method for v in ru.verifier]:
            errors.append(f"{pid}: verifier methods differ between en/ru")
        if en.answer_type == AnswerType.MC:
            if [c.label for c in en.choices] != [c.label for c in ru.choices] or len(en.choices) != 4:
                errors.append(f"{pid}: MC labels misaligned")
                continue
            for ce, cr in zip(en.choices, ru.choices):
                if ce.text == cr.text:
                    continue  # language-invariant numeric options are fine
                ve, vr = _parse_number(ce.text), _parse_number(cr.text)
                if ve is None or vr is None or abs(ve - vr) > 1e-9 * max(1.0, abs(ve)):
                    errors.append(f"{pid}: choice {ce.label} differs across languages in a non-numeric way")


def qc_mc(bundles: list[PairBundle], errors: list[str], metrics: dict[str, Any]) -> None:
    letters: Counter[str] = Counter()
    mc_pairs = [b for b in bundles if b.answer_type == AnswerType.MC]
    for b in mc_pairs:
        pid = b.pair_id
        en = b.en
        if en.canonical_answer not in ("A", "B", "C", "D"):
            errors.append(f"{pid}: MC canonical answer {en.canonical_answer!r} is not a letter")
            continue
        letters[en.canonical_answer] += 1
        texts = [normalize_text(c.text) for c in en.choices]
        if len(set(texts)) != 4:
            errors.append(f"{pid}: duplicate MC option texts after normalization")
        vals = [_parse_number(c.text) for c in en.choices]
        canon_i = "ABCD".index(en.canonical_answer)
        canon_v = vals[canon_i]
        if canon_v is not None:
            for i, v in enumerate(vals):
                if v is None or i == canon_i:
                    continue
                rel = abs(v - canon_v) / max(abs(v), abs(canon_v), 1e-9)
                if rel <= 0.02:
                    errors.append(
                        f"{pid}: MC distractor {en.choices[i].label}={v} too close to canonical {canon_v}"
                    )
    n_mc = len(mc_pairs)
    metrics["mc_letter_distribution"] = dict(sorted(letters.items()))
    metrics["n_mc_pairs"] = n_mc
    if n_mc:
        for letter in "ABCD":
            share = letters.get(letter, 0) / n_mc
            if not (LETTER_SHARE_BOUNDS[0] <= share <= LETTER_SHARE_BOUNDS[1]):
                errors.append(
                    f"MC letter {letter} share {share:.3f} outside {LETTER_SHARE_BOUNDS}"
                )


def qc_dedup(bundles: list[PairBundle], errors: list[str], metrics: dict[str, Any]) -> None:
    groups: dict[tuple[str, str], list[tuple[str, set[str], str]]] = defaultdict(list)
    hashes: dict[tuple[str, str], set[str]] = defaultdict(set)
    for b in bundles:
        for item in (b.en, b.ru):
            key = (b.subject.value, item.language.value)
            norm = normalize_text(item.question)
            if norm in hashes[key]:
                errors.append(f"{item.item_id}: exact duplicate normalized question text")
            hashes[key].add(norm)
            groups[key].append((item.item_id, word_trigrams(item.question), norm))
    worst = ("", "", 0.0)
    per_group: dict[str, float] = {}
    for key, entries in groups.items():
        max_sim = 0.0
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                sim = jaccard(entries[i][1], entries[j][1])
                if sim > max_sim:
                    max_sim = sim
                    if sim > worst[2]:
                        worst = (entries[i][0], entries[j][0], sim)
        per_group["/".join(key)] = round(max_sim, 4)
        if max_sim >= QC_SIMILARITY_CAP:
            errors.append(
                f"near-duplicate in {key}: max word-3gram Jaccard {max_sim:.3f} >= {QC_SIMILARITY_CAP}"
            )
    metrics["max_jaccard_per_subject_language"] = per_group
    metrics["max_jaccard_overall"] = round(worst[2], 4)
    metrics["max_jaccard_pair"] = [worst[0], worst[1]]


def qc_distribution(bundles: list[PairBundle], metrics: dict[str, Any], errors: list[str]) -> None:
    items = [it for b in bundles for it in (b.en, b.ru)]
    n_pairs = len(bundles)
    metrics["n_pairs"] = n_pairs
    metrics["n_items"] = len(items)
    if n_pairs < MIN_PAIRS_TOTAL:
        errors.append(f"total pairs {n_pairs} < {MIN_PAIRS_TOTAL}")
    if len(items) < MIN_ITEMS_TOTAL:
        errors.append(f"total items {len(items)} < {MIN_ITEMS_TOTAL}")
    per_subject: Counter[str] = Counter(b.subject.value for b in bundles)
    metrics["pairs_per_subject"] = dict(sorted(per_subject.items()))
    for subj, minimum in SUBJECT_MIN_PAIRS.items():
        if per_subject.get(subj, 0) < minimum:
            errors.append(f"{subj}: {per_subject.get(subj, 0)} pairs < {minimum}")
    diff_per_subject: dict[str, Counter[str]] = {}
    for subj in per_subject:
        diff_per_subject[subj] = Counter(b.difficulty.value for b in bundles if b.subject.value == subj)
        n_subj = per_subject[subj]
        for diff, (lo, hi) in DIFFICULTY_BOUNDS.items():
            share = diff_per_subject[subj].get(diff, 0) / n_subj
            if not (lo <= share <= hi):
                errors.append(f"{subj}/{diff}: share {share:.3f} outside ({lo}, {hi})")
    metrics["difficulty_per_subject"] = {s: dict(sorted(c.items())) for s, c in diff_per_subject.items()}
    at: Counter[str] = Counter(b.answer_type.value for b in bundles)
    metrics["answer_type_share"] = {k: round(at.get(k, 0) / n_pairs, 4) for k in ("mc", "numeric", "exact")}
    for a, (lo, hi) in ANSWER_TYPE_BOUNDS.items():
        share = at.get(a, 0) / n_pairs
        if not (lo <= share <= hi):
            errors.append(f"answer type {a}: share {share:.3f} outside ({lo}, {hi})")
    lang_counts: Counter[str] = Counter(it.language.value for it in items)
    metrics["items_per_language"] = dict(sorted(lang_counts.items()))
    if len(set(lang_counts.values())) != 1:
        errors.append(f"language counts unbalanced: {dict(lang_counts)}")
    rows: Counter[tuple[str, str, str, str]] = Counter()
    for it in items:
        rows[(it.subject.value, it.difficulty.value, it.answer_type.value, it.language.value)] += 1
    metrics["distribution_rows"] = [
        {"subject": s, "difficulty": d, "answer_type": a, "language": lang, "n": n}
        for (s, d, a, lang), n in sorted(rows.items())
    ]


def qc_answers(bundles: list[PairBundle], errors: list[str]) -> None:
    for b in bundles:
        for it in (b.en, b.ru):
            pid_lang = it.item_id
            q = it.question.strip()
            if not q.endswith("?"):
                errors.append(f"{pid_lang}: question does not end with '?'")
            if "#" in q:
                errors.append(f"{pid_lang}: markdown header inside question")
            if not (20 <= len(q) <= 800):
                errors.append(f"{pid_lang}: question length {len(q)} out of range")
            steps = [ln for ln in it.solution.splitlines() if re.match(r"^\d+\)", ln)]
            if not (2 <= len(steps) <= 5):
                errors.append(f"{pid_lang}: solution has {len(steps)} numbered steps (need 2-5)")
            if it.canonical_answer not in it.solution:
                errors.append(f"{pid_lang}: canonical answer missing from solution")
            if not it.difficulty_rubric:
                errors.append(f"{pid_lang}: empty difficulty_rubric")
            if it.answer_type == AnswerType.NUMERIC:
                try:
                    v = float(it.canonical_answer)
                except ValueError:
                    errors.append(f"{pid_lang}: numeric canonical {it.canonical_answer!r} not a float")
                    continue
                if abs(v) >= 1e9:
                    errors.append(f"{pid_lang}: |answer| {v} >= 1e9")
                if not _sig_ok(it.canonical_answer):
                    errors.append(f"{pid_lang}: canonical {it.canonical_answer!r} exceeds 6 significant digits")
                if it.tolerance is None or (it.tolerance.rel is None and it.tolerance.abs is None):
                    errors.append(f"{pid_lang}: numeric item without tolerance")
            elif it.answer_type == AnswerType.EXACT:
                if not it.canonical_answer or len(it.canonical_answer) > 40:
                    errors.append(f"{pid_lang}: exact canonical malformed: {it.canonical_answer!r}")
                if CYRILLIC.search(it.canonical_answer):
                    errors.append(f"{pid_lang}: Cyrillic in canonical answer {it.canonical_answer!r}")


def qc_splits(bundles: list[PairBundle], metrics: dict[str, Any], errors: list[str]) -> None:
    per_subject: dict[str, Counter[str]] = defaultdict(Counter)
    for b in bundles:
        per_subject[b.subject.value][b.split.value] += 1
    total_dev = sum(c.get(Split.DEV.value, 0) for c in per_subject.values())
    metrics["splits_per_subject"] = {s: dict(sorted(c.items())) for s, c in per_subject.items()}
    metrics["n_dev_pairs"] = total_dev
    if not (12 <= total_dev <= 21):
        errors.append(f"total dev pairs {total_dev} outside 12..21")
    for s, c in per_subject.items():
        dev = c.get(Split.DEV.value, 0)
        if not (5 <= dev <= 7):
            errors.append(f"{s}: {dev} dev pairs outside 5..7")
        train = c.get(Split.TRAIN.value, 0)
        if train:
            errors.append(f"{s}: {train} train pairs (must be 0)")


def qc_verifiers(bundles: list[PairBundle], metrics: dict[str, Any], errors: list[str]) -> None:
    methods: Counter[str] = Counter()
    passed = 0
    failed = 0
    for b in bundles:
        if not b.en.verifier:
            errors.append(f"{b.pair_id}: no verifier records")
            continue
        for rec in b.en.verifier:
            methods[rec.method] += 1
            if rec.passed:
                passed += 1
            else:
                failed += 1
                errors.append(f"{b.pair_id}: verifier {rec.method} failed: {rec.detail}")
        if [r.passed for r in b.en.verifier] != [r.passed for r in b.ru.verifier]:
            errors.append(f"{b.pair_id}: verifier outcomes differ between languages")
    metrics["verifier_methods"] = dict(sorted(methods.items()))
    metrics["verifier_passed"] = passed
    metrics["verifier_failed"] = failed
    metrics["verifier_pass_rate"] = round(passed / (passed + failed), 6) if (passed + failed) else 0.0


def run_qc(bundles: list[PairBundle]) -> QCResult:
    """Run all gates; any error in the result must abort the build."""
    errors: list[str] = []
    metrics: dict[str, Any] = {}
    items = [it for b in bundles for it in (b.en, b.ru)]
    qc_schema(items, errors)
    qc_pairing(bundles, errors)
    qc_mc(bundles, errors, metrics)
    qc_dedup(bundles, errors, metrics)
    qc_distribution(bundles, errors)
    qc_answers(bundles, errors)
    qc_splits(bundles, errors)
    qc_verifiers(bundles, errors, metrics)
    return QCResult(errors=errors, metrics=metrics)
