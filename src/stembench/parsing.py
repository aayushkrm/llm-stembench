"""Auditable answer extraction: multiple choice, exact match, numeric with units.

Every parse returns (value, method) so downstream tables can audit how each answer was
extracted. Cyrillic look-alike letters (А/В/Е/С/Н/К/М/Т…) are normalized because Russian
models frequently answer with Cyrillic letters.
"""

from __future__ import annotations

import re

# Cyrillic letters that look like Latin MC labels -> Latin
CYRILLIC_TO_LATIN = {
    "А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H", "К": "K",
    "М": "M", "О": "O", "Р": "P", "Т": "T", "Х": "X", "У": "Y",
}

LABEL_RE = re.compile(r"^\(?([A-Fa-f])\)?[\.\)\:]?", re.UNICODE)

# A captured letter must not be part of a longer word (e.g. the 'A' of 'Answer'):
_NOT_WORD = r"(?![A-Za-zА-Яа-яЁё])"

ANSWER_PATTERNS = [
    # "Answer: B" / "Answer: (B)" / "ответ: B" / "answer is C"
    re.compile(
        r"(?:final\s+)?answer\s*(?:is)?\s*[:＝:]?\s*\(?\**\s*([A-Fa-fА-Яа-я])"
        + _NOT_WORD + r"\s*\**\)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:ответ|ответик)\s*(?:is|=|:)?\s*\(?\**\s*([A-Fa-fА-Яа-я])"
        + _NOT_WORD + r"\s*\**\)?",
        re.IGNORECASE,
    ),
    re.compile(r"^\**\(?([A-F])\)?\**\s*[\.\)]", re.MULTILINE),  # line starting "B." / "**B)**"
]

CONFIDENCE_RE = re.compile(
    r"(?:confidence|уверенность|уверенність)\s*[:＝:=]?\s*\**\s*(\d{1,3})\s*(?:%|percent|процент[ао]?в?)?\s*\**",
    re.IGNORECASE,
)

NUM_RE = re.compile(
    r"""[-−+]?\d+(?:[.,]\d+)?\s?[×x*]\s?10\^?[\{\(]?([-+]?\d+)[\}\)]?  # 3.2×10^4 form first
      |[-−+]?\s?\d{1,3}(?:[ \u00A0]\d{3})+(?:[.,]\d+)?  # 1 234 567.8 space-thousands
      |[-−+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?  # 1,234,567.8 comma-thousands
      |[-−+]?\d+(?:[.,]\d+)?(?:\s?[eE]\s?[-+]?\d+)?  # plain or 1e-3 form
    """,
    re.VERBOSE,
)

UNIT_WORDS = [
    "m/s", "km/h", "m/s^2", "m/s²", "N", "J", "kJ", "W", "kW", "Pa", "kPa", "atm",
    "V", "A", "Ω", "ohm", "C", "F", "mol", "mol/L", "M", "g", "kg", "mg", "m", "km",
    "cm", "mm", "s", "ms", "min", "h", "L", "mL", "°C", "°F", "K", "Hz", "kJ/mol",
    "g/mol", "J/(kg·K)", "kg*m/s", "m^2", "m^3", "cm^3", "dm^3", "%", "units",
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


def extract_mc_answer(text: str, n_choices: int = 4) -> tuple[str, str] | None:
    """Extract a choice letter. Returns (letter, method) or None."""
    if not text or not text.strip():
        return None
    limit = chr(ord("A") + n_choices)  # exclusive upper bound
    norm = normalize_cyrillic_letters(text)

    # match against both the raw text (RU keywords intact) and the normalized text
    for source in (text, norm):
        for i, pat in enumerate(ANSWER_PATTERNS):
            for m in pat.finditer(source):
                letter = normalize_cyrillic_letters(m.group(1)).upper()
                if "A" <= letter < limit:
                    return letter, f"pattern{i + 1}"

    # Bare letter as a short final line, possibly bold/underscore-wrapped: "**D**",
    # "__B__", "(C)", "A."
    lines = [ln.strip() for ln in norm.strip().splitlines() if ln.strip()]
    if lines:
        stripped = re.sub(r"^[*_~>\s]+|[*_~>\s]+$", "", lines[-1])
        m = LABEL_RE.match(stripped)
        if m and len(stripped) <= 4:
            letter = m.group(1).upper()
            if "A" <= letter < limit:
                return letter, "last_line"
    return None


def extract_confidence(text: str) -> float | None:
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


def extract_numeric(text: str) -> tuple[float, str] | None:
    """Extract the first plausible numeric value from text after 'Answer:' if present.

    Returns (value, raw_match). Looks preferentially inside an Answer: segment.
    """
    if not text:
        return None
    norm = normalize_cyrillic_letters(text)
    ans_part = norm
    m = re.search(
        r"(?:answer|ответ)\s*(?:is|=|:)?\s*[:＝:=]?\s*(.+?)(?:\n|$)", norm,
        re.IGNORECASE,
    )
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
    """Return the first unit symbol that appears as a standalone token.

    Word-boundary aware: the 'N' in 'Answer' or the 'm' in 'team' must not match.
    Units are case-meaningful (M = molar, m = metre): an exact-case pass runs first,
    a case-insensitive pass second.
    """
    norm = normalize_ws(text)
    for flags in (0, re.IGNORECASE):
        for u in sorted(UNIT_WORDS, key=len, reverse=True):
            pattern = rf"(?<![\w.]){re.escape(u)}(?![\w.])"
            if re.search(pattern, norm, flags):
                return u
    return ""


# Russian unit spellings -> canonical UNIT_WORDS entries (lowercase keys).
RU_UNIT_ALIASES = {
    "моль/л": "mol/L", "г/моль": "g/mol", "м/с²": "m/s²", "м/с": "m/s",
    "км/ч": "km/h", "кдж": "kJ", "дж": "J", "моль": "mol", "кг": "kg",
    "мг": "mg", "км": "km", "см": "cm", "мм": "mm", "мл": "mL", "кпа": "kPa",
    "па": "Pa", "вт": "W", "квт": "kW", "г": "g", "м": "m", "с": "s",
    "н": "N", "к": "K", "в": "V", "а": "A", "л": "L", "%": "%",
    "кг·м/с": "kg*m/s", "кг*м/с": "kg*m/s",
}

_ANSWER_SEG_RE = re.compile(
    r"(?:answer|ответ)\s*(?:is|=|:)?\s*[:＝:=]?\s*(.+?)(?:\n|$)", re.IGNORECASE,
)
_CONF_TAIL_RE = re.compile(
    r"(?:confidence|уверенность)\s*[:＝:=]?.*$", re.IGNORECASE,
)


def _unit_match_form(u: str) -> str:
    """Normalize a unit for matching: unify separators (·, × -> *), drop the
    spaces models put inside compound units ("mol / L" -> "mol/L"), rewrite
    negative-exponent notation ("mol L⁻¹" -> "mol/L"), and join the implicit
    multiplication of the momentum compound ("kg m/s" -> "kg*m/s")."""
    u = re.sub(r"([\w.]+)\s*(?:⁻¹|\⁻¹|\^-1)", r"/\1", u)
    u = re.sub(r"(\w)\s*([/*·×])\s*(\w)", r"\1\2\3", u)
    u = re.sub(r"\bkg\s+m/s\b", "kg*m/s", u)
    return u.replace("·", "*").replace("×", "*")


def answer_units(text: str) -> list[str]:
    """Unit tokens stated in the answer-scoped region of ``text``.

    Scope = the content after the last substantive final-answer marker
    (Answer:/Ответ:) whose segment contains a digit, falling back to the last
    non-empty line. Units stated elsewhere (restated problem, intermediate
    reasoning) must never influence scoring. Several tokens may coexist
    ("399000 J (399 кДж)" -> ["J", "kJ"]); Russian spellings map to canonical
    UNIT_WORDS entries. Returned tokens are separator-normalized ("kg·m/s" and
    "kg*m/s" both -> "kg*m/s").
    """
    if not text:
        return []
    norm = normalize_ws(normalize_cyrillic_letters(text))
    segments = _ANSWER_SEG_RE.findall(norm)
    seg = ""
    with_digit = [s for s in segments if re.search(r"\d", s)]
    if with_digit:
        seg = with_digit[-1]
    elif segments:
        seg = segments[-1]
    seg = _CONF_TAIL_RE.sub("", seg).strip()
    if not seg:
        lines = [ln.strip() for ln in norm.splitlines() if ln.strip()]
        seg = lines[-1] if lines else ""
    if not seg:
        return []
    seg_match = _unit_match_form(seg)
    found: list[str] = []
    for u in sorted(UNIT_WORDS, key=len, reverse=True):
        entry = _unit_match_form(u)
        if re.search(rf"(?<![\w.]){re.escape(entry)}(?![\w.])", seg_match, re.IGNORECASE):
            found.append(entry)
    low = _unit_match_form(seg.lower())
    for ru in sorted(RU_UNIT_ALIASES, key=len, reverse=True):
        if re.search(rf"(?<![\w.]){re.escape(_unit_match_form(ru))}(?![\w.])", low):
            canon = RU_UNIT_ALIASES[ru]
            if canon not in found:
                found.append(canon)
    return found


def normalize_exact(text: str) -> str:
    t = strip_punct(normalize_ws(normalize_cyrillic_letters(text or "")))
    t = t.rstrip(".;!")
    return t.lower()


def extract_exact_answer(text: str) -> str | None:
    """Prefer the content after 'Answer:'/'Ответ:'; else the last non-empty line."""
    if not text or not text.strip():
        return None
    norm = text.strip()
    m = re.search(
        r"(?:answer|ответ)\s*(?:is|=|:)?\s*[:＝:=]?\s*(.+?)(?:\n|$)",
        norm, re.IGNORECASE,
    )
    seg = m.group(1).strip() if m else ""
    seg = re.sub(
        r"(?:confidence|уверенность)\s*[:＝:=]?.*$", "", seg, flags=re.IGNORECASE
    ).strip()
    if seg:
        return seg
    lines = [ln.strip() for ln in norm.splitlines() if ln.strip()]
    return lines[-1] if lines else None
