"""Auditable answer extraction: multiple choice, exact match, numeric with units.

Every parse returns (value, method) so downstream tables can audit how each answer was
extracted. Cyrillic look-alike letters (А/В/Е/С/Н/К/М/Т…) are normalized because Russian
models frequently answer with Cyrillic letters.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

# Cyrillic letters that look like Latin MC labels -> Latin
CYRILLIC_TO_LATIN = {
    "А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H", "К": "K",
    "М": "M", "О": "O", "Р": "P", "Т": "T", "Х": "X", "У": "Y",
}

LABEL_RE = re.compile(r"^\(?([A-Fa-f])\)?[\.\)\:]?", re.UNICODE)

ANSWER_PATTERNS = [
    # "Answer: B" / "Answer: (B)" / "Ответ: B" / "answer is C"
    re.compile(r"(?:final\s+)?answer\s*(?:is)?\s*[:＝:]?\s*\(?\**\s*([A-Fa-fА-Яа-я])\s*\**\)?", re.IGNORECASE),
    re.compile(r"(?:ответ|ответик)\s*(?:is|=|:)?\s*\(?\**\s*([A-Fa-fА-Яа-я])\s*\**\)?", re.IGNORECASE),
    re.compile(r"^\**\(?([A-F])\)?\**\s*[\.\)]", re.MULTILINE),  # line starting with "B." / "**B)**"
]

CONFIDENCE_RE = re.compile(
    r"(?:confidence|уверенность|уверенність)\s*[:＝:=]?\s*\**\s*(\d{1,3})\s*(?:%|percent|процент[ао]?в?)?\s*\**",
    re.IGNORECASE,
)

NUM_RE = re.compile(
    r"""[-−+]?\s?\d{1,3}(?:[ \u00A0]\d{3})+(?:[.,]\d+)?   # 1 234 567.8 (space thousands)
      |[-−+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?                 # 1,234,567.8
      |[-−+]?\d+(?:[.,]\d+)?(?:\s?[eE]\s?[-+]?\d+)?       # plain / 1e-3
      |[-−+]?\d+(?:[.,]\d+)?\s?[×x*]\s?10\^?[\{\(]?([-+]?\d+)[\}\)]?  # 3.2×10^4
    """,
    re.VERBOSE,
)

UNIT_WORDS = [
    "m/s", "km/h", "m/s^2", "m/s²", "N", "J", "kJ", "W", "kW", "Pa", "kPa", "atm",
    "V", "A", "Ω", "ohm", "C", "F", "mol", "mol/L", "M", "g", "kg", "mg", "m", "km",
    "cm", "mm", "s", "ms", "min", "h", "L", "mL", "°C", "°F", "K", "Hz", "kJ/mol",
    "g/mol", "J/(kg·K)", "m^2", "m^3", "cm^3", "dm^3", "%", "units",
]


def normalize_cyrillic_letters(text: str) -> str:
    out = []
    for ch in text:
        out.append(CYRILLIC_TO_LATIN.get(ch, ch))
    return "".join(out)


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_punct(text: str) -> str:
    return re.sub(r"[«»\"'`*_#$]+", "", text)


def extract_mc_answer(text: str, n_choices: int = 4) -> Optional[tuple[str, str]]:
    """Extract a choice letter. Returns (letter, method) or None."""
    if not text or not text.strip():
        return None
    limit = chr(ord("A") + n_choices)  # exclusive upper bound
    norm = normalize_cyrillic_letters(text)

    for i, pat in enumerate(ANSWER_PATTERNS):
        for m in pat.finditer(norm):
            letter = m.group(1).upper()
            if "A" <= letter < limit:
                return letter, f"pattern{i + 1}"

    # Bare letter as the first non-whitespace character of a short final line
    lines = [ln.strip() for ln in norm.strip().splitlines() if ln.strip()]
    if lines:
        m = LABEL_RE.match(lines[-1])
        if m and len(lines[-1]) <= 4:
            letter = m.group(1).upper()
            if "A" <= letter < limit:
                return letter, "last_line"
    return None


def extract_confidence(text: str) -> Optional[float]:
    """Self-reported confidence in [0,1], from 'Confidence: 87' style contract."""
    m = CONFIDENCE_RE.search(text or "")
    if not m:
        return None
    val = float(m.group(1))
    if val > 100:  # malformed
        return None
    return val / 100.0


def _number_from_match(m: re.Match) -> float | None:
    s = m.group(0).replace("\u00A0", " ").replace("−", "-").replace(" ", "")
    # 3.2×10^4 style
    x10 = re.search(r"([-\+]?\d+(?:[.,]\d+)?)\s?[×x*]\s?10\^?[\{\(]?([-\+]?\d+)", s)
    if x10:
        mant = float(x10.group(1).replace(",", "."))
        return mant * 10 ** int(x10.group(2))
    # 1 234 567 style (space thousand groups)
    if re.fullmatch(r"[-\+]?\d{1,3}( ?\d{3})+(?:[.,]\d+)?", s):
        s = s.replace(" ", "").replace(",", ".")
        return float(s)
    # 1,234,567.8 (comma thousands with decimal point)
    if re.fullmatch(r"[-\+]?\d{1,3}(,\d{3})+(\.\d+)?", s):
        return float(s.replace(",", ""))
    # Decimal comma -> point (RU convention), e.g. "3,14"
    if re.fullmatch(r"[-\+]?\d+,\d+", s):
        return float(s.replace(",", "."))
    try:
        return float(s)
    except ValueError:
        return None


def extract_numeric(text: str) -> Optional[tuple[float, str]]:
    """Extract the first plausible numeric value from text after 'Answer:' if present.

    Returns (value, raw_match). Looks preferentially inside an Answer: segment.
    """
    if not text:
        return None
    norm = normalize_cyrillic_letters(text)
    ans_part = norm
    m = re.search(r"answer\s*(?:is)?\s*[:＝:=]?\s*(.+?)(?:\n|$)", norm, re.IGNORECASE)
    if m:
        ans_part = m.group(1)
    # avoid matching the "0-100" of the confidence contract
    ans_part = re.sub(r"confidence\s*[:＝:=]?\s*\d{1,3}", "", ans_part, flags=re.IGNORECASE)
    for cand in (ans_part, norm):
        for m in NUM_RE.finditer(cand):
            val = _number_from_match(m)
            if val is not None:
                return val, m.group(0)
    return None


def extract_unit(text: str) -> str:
    norm = normalize_ws(text)
    for u in sorted(UNIT_WORDS, key=len, reverse=True):
        if u.lower() in norm.lower():
            return u
    return ""


def normalize_exact(text: str) -> str:
    t = strip_punct(normalize_ws(normalize_cyrillic_letters(text or "")))
    t = t.rstrip(".;!")
    return t.lower()


def extract_exact_answer(text: str) -> Optional[str]:
    """Prefer the content after 'Answer:'; else the last non-empty line."""
    if not text or not text.strip():
        return None
    norm = text.strip()
    m = re.search(r"answer\s*(?:is)?\s*[:＝:=]?\s*(.+?)(?:\n|$)", norm, re.IGNORECASE)
    seg = m.group(1).strip() if m else ""
    seg = re.sub(r"confidence\s*[:＝:=]?.*$", "", seg, flags=re.IGNORECASE).strip()
    if seg:
        return seg
    lines = [ln.strip() for ln in norm.splitlines() if ln.strip()]
    return lines[-1] if lines else None
