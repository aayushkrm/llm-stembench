"""Scoring: compare parsed answers to references.

Correctness conventions (documented in reports):
- MC: parsed letter == gold letter.
- EXACT: normalized string equality (case/punct/whitespace-insensitive), or membership
  in acceptable alternatives.
- NUMERIC: |parsed - gold| <= abs_tol OR relative tolerance; unit string equality is
  recorded but only enforced when the reference declares units AND the parsed unit is
  non-empty and mismatched (empty parsed unit -> warn flag, not failure).
- Unparseable output: correctness=None (parse failure tracked separately, never silently
  dropped; both strict and lenient accuracies reported downstream).
"""

from __future__ import annotations

import math

from stembench.parsing import (
    extract_exact_answer,
    extract_mc_answer,
    extract_numeric,
    extract_unit,
    normalize_exact,
)


def score_mc(raw_text: str, gold_index: int, n_choices: int = 4) -> tuple[bool | None, str, str]:
    """-> (correct, parsed_letter, method)"""
    got = extract_mc_answer(raw_text, n_choices=n_choices)
    if got is None:
        return None, "", "none"
    letter, method = got
    gold = chr(ord("A") + gold_index)
    return letter == gold, letter, method


def score_exact(
    raw_text: str, gold: str, alternatives: list[str] | None = None
) -> tuple[bool | None, str, str]:
    parsed = extract_exact_answer(raw_text)
    if parsed is None:
        return None, "", "none"
    ok = normalize_exact(parsed) == normalize_exact(gold)
    if not ok and alternatives:
        ok = any(normalize_exact(parsed) == normalize_exact(a) for a in alternatives)
    return ok, parsed, "exact"


def score_numeric(
    raw_text: str,
    gold: float,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
    require_unit: str = "",
) -> tuple[bool | None, str, str]:
    got = extract_numeric(raw_text)
    if got is None:
        return None, "", "none"
    val, raw = got
    unit = extract_unit(raw_text)
    ok = False
    if rel_tol is not None and gold != 0:
        ok = ok or math.isclose(val, gold, rel_tol=rel_tol, abs_tol=0.0)
    if abs_tol is not None:
        ok = ok or math.isclose(val, gold, rel_tol=0.0, abs_tol=abs_tol)
    if rel_tol is None and abs_tol is None:
        ok = math.isclose(val, gold, rel_tol=1e-6)
    if ok and require_unit and unit and unit.lower() not in require_unit.lower():
        ok = False  # explicit wrong unit
    return ok, raw, f"numeric(unit={unit or 'none'})"


def parse_failure_rate(records: list[dict]) -> float:
    if not records:
        return 0.0
    failed = sum(1 for r in records if r.get("correctness") is None)
    return failed / len(records)
