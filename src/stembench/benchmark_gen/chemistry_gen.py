"""Original bilingual (ru/en) chemistry item generators.

Contains an embedded table of standard atomic weights (IUPAC 2021, abridged to
4-5 significant digits) and a small formula parser; ``verify.py`` recomputes
all molar-mass-dependent answers from an independently written table and
parser.  Units are untranslated ("g", "mol", "mol/L", "g/mol") and the decimal
separator is "." in both languages (see ``_core`` docstring).
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from stembench.schemas import AnswerType, Difficulty, Subject

from ._core import PairDraft, fmt, pick_distractors, sol_en, sol_ru
from .math_gen import _mc_numeric, _set_numeric, _std_pool, pick_distractors_str

SUBJECT = Subject.CHEMISTRY
PREFIX = "CHEM"

TOPICS: dict[str, str] = {
    "molar_mass": "molar mass computation",
    "stoich_mass": "stoichiometry: mass and moles",
    "molarity": "solution concentration (molarity)",
    "dilution": "dilution of solutions",
    "gas_moles": "ideal gas: amount of substance",
    "ph_strong": "pH of strong acids and bases",
    "percent_comp": "percent composition",
    "empirical": "empirical formula",
    "limiting": "limiting reagent",
    "econfig": "electron configuration and the periodic table",
    "reactions": "reaction products and precipitates",
    "balancing": "balancing chemical equations",
}

RUBRICS: dict[tuple[str, Difficulty], tuple[str, str]] = {
    ("molar_mass", Difficulty.SCHOOL): (
        "Summing atomic weights from the periodic table.",
        "Суммирование атомных масс по таблице Менделеева.",
    ),
    ("stoich_mass", Difficulty.SCHOOL): (
        "Single conversion between mass and moles via the molar mass.",
        "Однократный переход между массой и количеством вещества через молярную массу.",
    ),
    ("stoich_mass", Difficulty.UNIVERSITY): (
        "Mass-moles conversion with less rounded numbers.",
        "Переход между массой и количеством вещества с менее круглыми числами.",
    ),
    ("molarity", Difficulty.SCHOOL): (
        "Definition of molarity as moles per litre of solution.",
        "Определение молярной концентрации как количества вещества на литр раствора.",
    ),
    ("molarity", Difficulty.UNIVERSITY): (
        "Molarity from a weighed sample dissolved in a given volume.",
        "Молярная концентрация по навеске вещества в заданном объёме раствора.",
    ),
    ("dilution", Difficulty.UNIVERSITY): (
        "Conservation of moles on dilution (C1*V1 = C2*V2).",
        "Сохранение количества вещества при разбавлении (C1*V1 = C2*V2).",
    ),
    ("gas_moles", Difficulty.UNIVERSITY): (
        "Ideal gas law solved for n with R = 8.314 L*kPa/(mol*K).",
        "Уравнение Менделеева — Клапейрона относительно n при R = 8.314 L*kPa/(mol*K).",
    ),
    ("ph_strong", Difficulty.UNIVERSITY): (
        "pH of a fully dissociated strong acid or base at a power-of-ten concentration.",
        "pH полностью диссоциированной сильной кислоты или основания при концентрации, равной степени десяти.",
    ),
    ("percent_comp", Difficulty.SCHOOL): (
        "Mass fraction of an element from atomic weights.",
        "Массовая доля элемента по атомным массам.",
    ),
    ("empirical", Difficulty.UNIVERSITY): (
        "Deriving an empirical formula from percent composition.",
        "Вывод простейшей формулы по массовым долям.",
    ),
    ("limiting", Difficulty.OLYMPIAD): (
        "Limiting-reagent analysis with a full mole comparison.",
        "Определение лимитирующего реагента через сравнение количеств вещества.",
    ),
    ("econfig", Difficulty.SCHOOL): (
        "Reading ground-state electron configurations of main-group and d-block atoms.",
        "Чтение электронных конфигураций атомов в основном состоянии.",
    ),
    ("reactions", Difficulty.SCHOOL): (
        "Recognizing the precipitate or gas of a simple exchange or acid reaction.",
        "Определение осадка или газа в простой реакции обмена или кислотной реакции.",
    ),
    ("reactions", Difficulty.UNIVERSITY): (
        "Predicting reaction products beyond the basic solubility rules.",
        "Прогноз продуктов реакций, выходящих за базовые правила растворимости.",
    ),
    ("balancing", Difficulty.UNIVERSITY): (
        "Atom counting to recover the integer stoichiometric coefficients.",
        "Подбор целых стехиометрических коэффициентов подсчётом атомов.",
    ),
}

SPEC = [
    # topic_key, difficulty, count, answer_type  (chemistry: 180 pairs)
    ("molar_mass", Difficulty.SCHOOL, 8, AnswerType.NUMERIC),
    ("molar_mass", Difficulty.SCHOOL, 6, AnswerType.MC),
    ("stoich_mass", Difficulty.SCHOOL, 6, AnswerType.NUMERIC),
    ("stoich_mass", Difficulty.SCHOOL, 4, AnswerType.MC),
    ("molarity", Difficulty.SCHOOL, 8, AnswerType.NUMERIC),
    ("molarity", Difficulty.SCHOOL, 4, AnswerType.MC),
    ("percent_comp", Difficulty.SCHOOL, 6, AnswerType.NUMERIC),
    ("percent_comp", Difficulty.SCHOOL, 4, AnswerType.MC),
    ("percent_comp", Difficulty.SCHOOL, 4, AnswerType.EXACT),
    ("econfig", Difficulty.SCHOOL, 10, AnswerType.MC),
    ("econfig", Difficulty.SCHOOL, 4, AnswerType.EXACT),
    ("reactions", Difficulty.SCHOOL, 8, AnswerType.EXACT),
    ("reactions", Difficulty.SCHOOL, 6, AnswerType.MC),
    ("stoich_mass", Difficulty.UNIVERSITY, 6, AnswerType.NUMERIC),
    ("stoich_mass", Difficulty.UNIVERSITY, 4, AnswerType.MC),
    ("molarity", Difficulty.UNIVERSITY, 8, AnswerType.NUMERIC),
    ("dilution", Difficulty.UNIVERSITY, 8, AnswerType.NUMERIC),
    ("dilution", Difficulty.UNIVERSITY, 4, AnswerType.MC),
    ("gas_moles", Difficulty.UNIVERSITY, 8, AnswerType.NUMERIC),
    ("gas_moles", Difficulty.UNIVERSITY, 4, AnswerType.MC),
    ("ph_strong", Difficulty.UNIVERSITY, 8, AnswerType.NUMERIC),
    ("ph_strong", Difficulty.UNIVERSITY, 4, AnswerType.MC),
    ("empirical", Difficulty.UNIVERSITY, 4, AnswerType.EXACT),
    ("empirical", Difficulty.UNIVERSITY, 4, AnswerType.MC),
    ("balancing", Difficulty.UNIVERSITY, 10, AnswerType.MC),
    ("reactions", Difficulty.UNIVERSITY, 8, AnswerType.EXACT),
    ("reactions", Difficulty.UNIVERSITY, 4, AnswerType.MC),
    ("limiting", Difficulty.OLYMPIAD, 10, AnswerType.NUMERIC),
    ("limiting", Difficulty.OLYMPIAD, 6, AnswerType.MC),
    ("limiting", Difficulty.OLYMPIAD, 2, AnswerType.EXACT),
]

# IUPAC 2021 abridged atomic weights (g/mol).
ATOMIC_WEIGHTS: dict[str, float] = {
    "H": 1.008, "He": 4.003, "Li": 6.94, "Be": 9.012, "B": 10.81,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974,
    "S": 32.06, "Cl": 35.45, "Ar": 39.948, "K": 39.098, "Ca": 40.078,
    "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938, "Fe": 55.845,
    "Ni": 58.693, "Cu": 63.546, "Zn": 65.38, "Br": 79.904, "Ag": 107.868,
    "I": 126.904, "Ba": 137.327, "Pb": 207.2, "Sn": 118.71,
}

_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)|(\()|(\))(\d*)")  # documented format contract


def parse_formula(formula: str) -> dict[str, int]:
    """Atom counts for a formula like 'Ca(OH)2' or 'Fe2(SO4)3'."""
    stack: list[dict[str, int]] = [{}]
    pos = 0
    while pos < len(formula):
        ch = formula[pos]
        if ch == "(":
            stack.append({})
            pos += 1
        elif ch == ")":
            pos += 1
            m = re.match(r"\d*", formula[pos:])
            mult = int(m.group(0)) if m and m.group(0) else 1
            pos += len(m.group(0)) if m else 0
            group = stack.pop()
            for el, cnt in group.items():
                stack[-1][el] = stack[-1].get(el, 0) + cnt * mult
        else:
            m = re.match(r"[A-Z][a-z]?", formula[pos:])
            if not m:
                raise ValueError(f"bad formula {formula!r} at {pos}")
            el = m.group(0)
            pos += len(el)
            m2 = re.match(r"\d*", formula[pos:])
            cnt = int(m2.group(0)) if m2 and m2.group(0) else 1
            pos += len(m2.group(0)) if m2 else 0
            stack[-1][el] = stack[-1].get(el, 0) + cnt
    if len(stack) != 1:
        raise ValueError(f"unbalanced parentheses in {formula!r}")
    return stack[0]


def molar_mass(formula: str) -> float:
    return sum(ATOMIC_WEIGHTS[el] * cnt for el, cnt in parse_formula(formula).items())


# (formula, English name, Russian name)
COMPOUNDS: tuple[tuple[str, str, str], ...] = (
    ("CO2", "carbon dioxide", "углекислый газ"),
    ("H2O", "water", "вода"),
    ("NaCl", "sodium chloride", "хлорид натрия"),
    ("CaCO3", "calcium carbonate", "карбонат кальция"),
    ("H2SO4", "sulfuric acid", "серная кислота"),
    ("NH3", "ammonia", "аммиак"),
    ("CH4", "methane", "метан"),
    ("KBr", "potassium bromide", "бромид калия"),
    ("MgO", "magnesium oxide", "оксид магния"),
    ("Al2O3", "aluminium oxide", "оксид алюминия"),
    ("Fe2O3", "iron(III) oxide", "оксид железа(III)"),
    ("CuSO4", "copper(II) sulfate", "сульфат меди(II)"),
    ("ZnS", "zinc sulfide", "сульфид цинка"),
    ("HNO3", "nitric acid", "азотная кислота"),
    ("SO2", "sulfur dioxide", "оксид серы(IV)"),
    ("NaOH", "sodium hydroxide", "гидроксид натрия"),
    ("KCl", "potassium chloride", "хлорид калия"),
    ("CaCl2", "calcium chloride", "хлорид кальция"),
    ("Na2CO3", "sodium carbonate", "карбонат натрия"),
    ("C6H12O6", "glucose", "глюкоза"),
    ("KMnO4", "potassium permanganate", "перманганат калия"),
    ("NaHCO3", "sodium bicarbonate", "гидрокарбонат натрия"),
    ("H3PO4", "phosphoric acid", "ортофосфорная кислота"),
    ("MgCl2", "magnesium chloride", "хлорид магния"),
    ("ZnO", "zinc oxide", "оксид цинка"),
    ("C2H5OH", "ethanol", "этанол"),
    ("C12H22O11", "sucrose", "сахароза"),
    ("Fe(OH)3", "iron(III) hydroxide", "гидроксид железа(III)"),
    ("Cu(OH)2", "copper(II) hydroxide", "гидроксид меди(II)"),
    ("Al2(SO4)3", "aluminium sulfate", "сульфат алюминия"),
)

ECONFIGS: tuple[tuple[str, str], ...] = (
    ("O", "[He] 2s^2 2p^4"),
    ("N", "[He] 2s^2 2p^3"),
    ("Na", "[Ne] 3s^1"),
    ("Mg", "[Ne] 3s^2"),
    ("S", "[Ne] 3s^2 3p^4"),
    ("Cl", "[Ne] 3s^2 3p^5"),
    ("K", "[Ar] 4s^1"),
    ("Ca", "[Ar] 4s^2"),
    ("Fe", "[Ar] 3d^6 4s^2"),
    ("Zn", "[Ar] 3d^10 4s^2"),
    ("Cu", "[Ar] 3d^10 4s^1"),
    ("Cr", "[Ar] 3d^5 4s^1"),
    ("Ni", "[Ar] 3d^8 4s^2"),
    ("Mn", "[Ar] 3d^5 4s^2"),
)
ATOMIC_NUMBERS: dict[str, int] = {
    "He": 2, "Ne": 10, "Ar": 18,
    "O": 8, "N": 7, "Na": 11, "Mg": 12, "S": 16, "Cl": 17, "K": 19,
    "Ca": 20, "Fe": 26, "Zn": 30, "Cu": 29, "Cr": 24, "Ni": 28, "Mn": 25,
}

PRECIP_REACTIONS: tuple[tuple[str, str, str], ...] = (
    ("AgNO3", "NaCl", "AgCl"),
    ("BaCl2", "Na2SO4", "BaSO4"),
    ("Pb(NO3)2", "KI", "PbI2"),
    ("CuSO4", "NaOH", "Cu(OH)2"),
    ("FeCl3", "NaOH", "Fe(OH)3"),
    ("AgNO3", "KBr", "AgBr"),
    ("CaCl2", "Na2CO3", "CaCO3"),
    ("Ba(NO3)2", "K2SO4", "BaSO4"),
)

GAS_REACTIONS: tuple[tuple[str, str, str, str], ...] = (
    # reagent1, reagent2, gas product, en/ru description handled in template
    ("CaCO3", "HCl", "CO2", "acid_carbonate"),
    ("Na2CO3", "HCl", "CO2", "acid_carbonate"),
    ("Zn", "HCl", "H2", "metal_acid"),
    ("Mg", "H2SO4", "H2", "metal_acid"),
    ("CH4", "O2", "CO2", "combustion"),
    ("C", "O2", "CO2", "combustion"),
)

BALANCING_POOL: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...] = (
    # (equation, correct coefficients, perturbed coefficients)
    ("Fe2O3 + CO -> Fe + CO2", (1, 3, 2, 3), (1, 2, 2, 3)),
    ("H2 + O2 -> H2O", (2, 1, 2), (1, 1, 1)),
    ("CH4 + O2 -> CO2 + H2O", (1, 2, 1, 2), (1, 1, 1, 2)),
    ("Na + Cl2 -> NaCl", (2, 1, 2), (1, 1, 1)),
    ("Al + O2 -> Al2O3", (4, 3, 2), (2, 2, 1)),
    ("Mg + O2 -> MgO", (2, 1, 2), (1, 1, 1)),
    ("N2 + H2 -> NH3", (1, 3, 2), (1, 2, 2)),
    ("KClO3 -> KCl + O2", (2, 2, 3), (1, 1, 1)),
    ("Fe + O2 -> Fe3O4", (3, 2, 1), (2, 2, 1)),
    ("C2H6 + O2 -> CO2 + H2O", (2, 7, 4, 6), (1, 3, 2, 3)),
    ("Al + HCl -> AlCl3 + H2", (2, 6, 2, 3), (1, 3, 1, 2)),
    ("P4 + O2 -> P4O10", (1, 5, 1), (1, 3, 1)),
)

LIMITING_POOL: tuple[tuple[str, tuple[int, int, int], str], ...] = (
    # equation, (coeff reagent1, coeff reagent2, coeff product), product formula
    ("N2 + 3 H2 -> 2 NH3", (1, 3, 2), "NH3"),
    ("Zn + 2 HCl -> ZnCl2 + H2", (1, 2, 1), "H2"),
    ("CH4 + 2 O2 -> CO2 + 2 H2O", (1, 2, 1), "H2O"),
    ("2 H2 + O2 -> 2 H2O", (2, 1, 2), "H2O"),
    ("Fe2O3 + 3 CO -> 2 Fe + 3 CO2", (1, 3, 2), "Fe"),
    ("Mg + 2 HCl -> MgCl2 + H2", (1, 2, 1), "H2"),
    ("2 Al + 3 Cl2 -> 2 AlCl3", (2, 3, 2), "AlCl3"),
    ("4 Fe + 3 O2 -> 2 Fe2O3", (4, 3, 2), "Fe2O3"),
)

EMPIRICAL_POOL: tuple[tuple[dict[str, float], str], ...] = (
    ({"C": 40.0, "H": 6.7, "O": 53.3}, "CH2O"),
    ({"C": 27.3, "O": 72.7}, "CO2"),
    ({"N": 30.4, "O": 69.6}, "NO2"),
    ({"N": 82.4, "H": 17.6}, "NH3"),
    ({"N": 46.7, "O": 53.3}, "NO"),
    ({"Fe": 69.9, "O": 30.1}, "Fe2O3"),
    ({"S": 50.0, "O": 50.0}, "SO2"),
    ({"C": 75.0, "H": 25.0}, "CH4"),
    ({"C": 85.7, "H": 14.3}, "CH2"),
    ({"Mg": 60.3, "O": 39.7}, "MgO"),
    ({"Al": 52.9, "O": 47.1}, "Al2O3"),
    ({"Ca": 40.0, "C": 12.0, "O": 48.0}, "CaCO3"),
    ({"Na": 39.3, "Cl": 60.7}, "NaCl"),
)

RU_ELEMENT_NAMES: dict[str, str] = {
    "H": "водорода", "C": "углерода", "N": "азота", "O": "кислорода",
    "Na": "натрия", "Mg": "магния", "Al": "алюминия", "S": "серы",
    "Cl": "хлора", "K": "калия", "Ca": "кальция", "Fe": "железа",
    "Cu": "меди", "Zn": "цинка", "P": "фосфора", "Si": "кремния",
}
EN_ELEMENT_NAMES: dict[str, str] = {
    "H": "hydrogen", "C": "carbon", "N": "nitrogen", "O": "oxygen",
    "Na": "sodium", "Mg": "magnesium", "Al": "aluminium", "S": "sulfur",
    "Cl": "chlorine", "K": "potassium", "Ca": "calcium", "Fe": "iron",
    "Cu": "copper", "Zn": "zinc", "P": "phosphorus", "Si": "silicon",
}


def _compound(rng: np.random.Generator) -> tuple[str, str, str]:
    return COMPOUNDS[int(rng.integers(0, len(COMPOUNDS)))]


def _finish(d: PairDraft, key: str, difficulty: Difficulty) -> PairDraft:
    d.topic = TOPICS[key]
    d.topic_key = key
    d.difficulty = difficulty
    d.subject = SUBJECT
    return d


def _emit(
    d: PairDraft,
    rng: np.random.Generator,
    atype: AnswerType,
    value: float,
    units: str,
    steps_en: list[str],
    steps_ru: list[str],
    extras: list[tuple[float, str]],
    params: dict[str, Any],
    difficulty: Difficulty,
    key: str,
) -> PairDraft:
    if atype == AnswerType.NUMERIC:
        _set_numeric(d, value, units)
        d.solution_en = sol_en(steps_en, d.canonical, units)
        d.solution_ru = sol_ru(steps_ru, d.canonical, units)
    else:
        _mc_numeric(d, rng, value, _std_pool(value, extras), units)
        d.solution_en = sol_en(steps_en, fmt(value), units)
        d.solution_ru = sol_ru(steps_ru, fmt(value), units)
    d.params = params
    return _finish(d, key, difficulty)


# --------------------------------------------------------------------------- #
# Molar mass (school)
# --------------------------------------------------------------------------- #
def g_molar_mass(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    formula, en_name, ru_name = _compound(rng)
    mm = molar_mass(formula)
    q_en = f"Calculate the molar mass of {en_name} ({formula})."
    q_ru = f"Вычислите молярную массу {ru_name} ({formula})."
    parts_en = " + ".join(f"{cnt}*{ATOMIC_WEIGHTS[el]}" for el, cnt in parse_formula(formula).items())
    steps_en = [
        f"Atom counts in {formula}: {parse_formula(formula)}.",
        f"M = {parts_en} = {fmt(mm)} g/mol.",
    ]
    steps_ru = [
        f"Число атомов в {formula}: {parse_formula(formula)}.",
        f"M = {parts_en} = {fmt(mm)} g/mol.",
    ]
    # plausible error: miscounted atoms of one element
    extras: list[tuple[float, str]] = []
    for el, cnt in parse_formula(formula).items():
        if cnt > 1:
            wrong = mm - ATOMIC_WEIGHTS[el]
            extras.append((wrong, f"atom_count_error_{el}"))
    d = PairDraft(SUBJECT, "", "", Difficulty.SCHOOL, atype, "", question_en=q_en, question_ru=q_ru)
    return _emit(d, rng, atype, mm, "g/mol", steps_en, steps_ru, extras,
                 {"formula": formula, "expected": mm}, Difficulty.SCHOOL, "molar_mass")


# --------------------------------------------------------------------------- #
# Stoichiometry: mass <-> moles (school + university)
# --------------------------------------------------------------------------- #
def g_stoich_mass(rng: np.random.Generator, idx: int, atype: AnswerType, difficulty: Difficulty) -> PairDraft:
    formula, en_name, ru_name = _compound(rng)
    mm = molar_mass(formula)
    if bool(rng.integers(0, 2)):  # mass -> moles
        if difficulty == Difficulty.SCHOOL:
            n_target = float(rng.choice([0.1, 0.2, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5]))
        else:
            n_target = round(float(rng.uniform(0.1, 4.5)), 2)
        mass = round(n_target * mm, 2)
        n_val = mass / mm
        q_en = f"How many moles of {en_name} ({formula}) are contained in {fmt(mass)} g of it?"
        q_ru = f"Какое количество вещества (моль) содержится в {fmt(mass)} g {ru_name} ({formula})?"
        steps_en = [
            f"M({formula}) = {fmt(mm)} g/mol.",
            f"n = m / M = {fmt(mass)} / {fmt(mm)} = {fmt(n_val)} mol.",
        ]
        steps_ru = [
            f"M({formula}) = {fmt(mm)} g/mol.",
            f"n = m / M = {fmt(mass)} / {fmt(mm)} = {fmt(n_val)} mol.",
        ]
        value, units = n_val, "mol"
        extras = [(mass / mm * 2, "factor_of_2"), (mm / mass if mass else 0.0, "inverted_ratio")]
        params = {"formula": formula, "mass": mass, "ask": "moles", "expected": n_val}
    else:  # moles -> mass
        if difficulty == Difficulty.SCHOOL:
            n_val = float(rng.choice([0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3]))
        else:
            n_val = round(float(rng.uniform(0.2, 3.5)), 2)
        mass = round(n_val * mm, 2)
        q_en = f"What is the mass of {fmt(n_val)} mol of {en_name} ({formula})?"
        q_ru = f"Какова масса {fmt(n_val)} mol {ru_name} ({formula})?"
        steps_en = [
            f"M({formula}) = {fmt(mm)} g/mol.",
            f"m = n * M = {fmt(n_val)} * {fmt(mm)} = {fmt(mass)} g.",
        ]
        steps_ru = [
            f"M({formula}) = {fmt(mm)} g/mol.",
            f"m = n * M = {fmt(n_val)} * {fmt(mm)} = {fmt(mass)} g.",
        ]
        value, units = mass, "g"
        extras = [(mass * 2, "factor_of_2"), (n_val + mm, "added_values")]
        params = {"formula": formula, "moles": n_val, "ask": "mass", "expected": mass}
    d = PairDraft(SUBJECT, "", "", difficulty, atype, "", question_en=q_en, question_ru=q_ru)
    return _emit(d, rng, atype, value, units, steps_en, steps_ru, extras, params, difficulty, "stoich_mass")


def g_stoich_school(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    return g_stoich_mass(rng, idx, atype, Difficulty.SCHOOL)


def g_stoich_uni(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    return g_stoich_mass(rng, idx, atype, Difficulty.UNIVERSITY)


# --------------------------------------------------------------------------- #
# Molarity (school + university)
# --------------------------------------------------------------------------- #
def g_molarity(rng: np.random.Generator, idx: int, atype: AnswerType, difficulty: Difficulty) -> PairDraft:
    if difficulty == Difficulty.SCHOOL:
        v_lit = round(float(rng.choice([0.25, 0.5, 0.75, 1, 1.5, 2, 2.5])), 2)
        c_target = round(float(rng.choice([0.1, 0.2, 0.25, 0.4, 0.5, 0.8, 1, 1.2, 1.6, 2])), 2)
        n_val = c_target * v_lit
        q_en = f"A solution contains {fmt(n_val)} mol of solute in {fmt(v_lit)} L of solution. What is its molar concentration?"
        q_ru = f"В {fmt(v_lit)} L раствора содержится {fmt(n_val)} mol растворённого вещества. Чему равна молярная концентрация раствора?"
        steps_en = [f"c = n / V = {fmt(n_val)} / {fmt(v_lit)} = {fmt(c_target)} mol/L."]
        steps_ru = [f"c = n / V = {fmt(n_val)} / {fmt(v_lit)} = {fmt(c_target)} mol/L."]
        value, units = c_target, "mol/L"
        extras = [(n_val * v_lit, "multiplied_values"), (n_val + v_lit, "added_values")]
        params = {"n": n_val, "v": v_lit, "expected": c_target}
    else:
        formula, en_name, ru_name = _compound(rng)
        mm = molar_mass(formula)
        v_lit = round(float(rng.choice([0.5, 1, 1.5, 2, 2.5])), 2)
        c_target = round(float(rng.choice([0.2, 0.3, 0.5, 0.75, 1, 1.25, 1.5, 2])), 2)
        mass = round(c_target * v_lit * mm, 2)
        c_val = mass / (mm * v_lit)
        q_en = (
            f"{fmt(mass)} g of {en_name} ({formula}) is dissolved in water and the solution volume is "
            f"brought to {fmt(v_lit)} L. What is the molar concentration of the solution?"
        )
        q_ru = (
            f"{fmt(mass)} g {ru_name} ({formula}) растворили в воде, и объём раствора довели до "
            f"{fmt(v_lit)} L. Чему равна молярная концентрация полученного раствора?"
        )
        steps_en = [
            f"M({formula}) = {fmt(mm)} g/mol, so n = m / M = {fmt(mass)} / {fmt(mm)} = {fmt(mass / mm)} mol.",
            f"c = n / V = {fmt(mass / mm)} / {fmt(v_lit)} = {fmt(c_val)} mol/L.",
        ]
        steps_ru = [
            f"M({formula}) = {fmt(mm)} g/mol, поэтому n = m / M = {fmt(mass)} / {fmt(mm)} = {fmt(mass / mm)} mol.",
            f"c = n / V = {fmt(mass / mm)} / {fmt(v_lit)} = {fmt(c_val)} mol/L.",
        ]
        value, units = c_val, "mol/L"
        extras = [(mass / mm, "forgot_volume"), (mass / v_lit if v_lit else 0.0, "forgot_molar_mass")]
        params = {"formula": formula, "mass": mass, "v": v_lit, "expected": c_val}
    d = PairDraft(SUBJECT, "", "", difficulty, atype, "", question_en=q_en, question_ru=q_ru)
    return _emit(d, rng, atype, value, units, steps_en, steps_ru, extras, params, difficulty, "molarity")


def g_molarity_school(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    return g_molarity(rng, idx, atype, Difficulty.SCHOOL)


def g_molarity_uni(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    return g_molarity(rng, idx, atype, Difficulty.UNIVERSITY)


# --------------------------------------------------------------------------- #
# Dilution (university)
# --------------------------------------------------------------------------- #
def g_dilution(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    c1 = round(float(rng.choice([0.5, 0.8, 1, 1.2, 1.5, 2, 2.5, 3, 4])), 2)
    k = int(rng.choice([2, 3, 4, 5]))
    c2 = c1 / k
    v1 = round(float(rng.choice([0.2, 0.25, 0.5, 0.75, 1, 1.5, 2])), 2)
    v2 = c1 * v1 / c2
    if bool(rng.integers(0, 2)):
        q_en = (
            f"What volume of a {fmt(c2)} mol/L solution can be prepared from {fmt(v1)} L of a "
            f"{fmt(c1)} mol/L solution by dilution with water?"
        )
        q_ru = (
            f"Какой объём раствора с концентрацией {fmt(c2)} mol/L можно приготовить из {fmt(v1)} L "
            f"раствора с концентрацией {fmt(c1)} mol/L, разбавляя его водой?"
        )
        steps_en = [
            f"Moles of solute are conserved: C1*V1 = C2*V2.",
            f"V2 = {fmt(c1)} * {fmt(v1)} / {fmt(c2)} = {fmt(v2)} L.",
        ]
        steps_ru = [
            f"Количество растворённого вещества сохраняется: C1*V1 = C2*V2.",
            f"V2 = {fmt(c1)} * {fmt(v1)} / {fmt(c2)} = {fmt(v2)} L.",
        ]
        value, units = v2, "L"
        extras = [(v1 * k if k else 0.0, "inverted_ratio"), (v1 + c1 - c2, "delta_slip")]
        params = {"c1": c1, "v1": v1, "c2": c2, "ask": "v2", "expected": v2}
    else:
        water = v2 - v1
        q_en = (
            f"How much water must be added to {fmt(v1)} L of a {fmt(c1)} mol/L solution to lower its "
            f"concentration to {fmt(c2)} mol/L?"
        )
        q_ru = (
            f"Какой объём воды нужно добавить к {fmt(v1)} L раствора с концентрацией {fmt(c1)} mol/L, "
            f"чтобы концентрация стала {fmt(c2)} mol/L?"
        )
        steps_en = [
            f"C1*V1 = C2*V2 gives V2 = {fmt(c1)} * {fmt(v1)} / {fmt(c2)} = {fmt(v2)} L.",
            f"Added water: V2 - V1 = {fmt(v2)} - {fmt(v1)} = {fmt(water)} L.",
        ]
        steps_ru = [
            f"Из C1*V1 = C2*V2 получаем V2 = {fmt(c1)} * {fmt(v1)} / {fmt(c2)} = {fmt(v2)} L.",
            f"Добавленная вода: V2 - V1 = {fmt(v2)} - {fmt(v1)} = {fmt(water)} L.",
        ]
        value, units = water, "L"
        extras = [(v2, "forgot_subtraction"), (v1, "returned_initial_volume")]
        params = {"c1": c1, "v1": v1, "c2": c2, "ask": "water", "expected": water}
    d = PairDraft(SUBJECT, "", "", Difficulty.UNIVERSITY, atype, "", question_en=q_en, question_ru=q_ru)
    return _emit(d, rng, atype, value, units, steps_en, steps_ru, extras, params, Difficulty.UNIVERSITY, "dilution")


# --------------------------------------------------------------------------- #
# Ideal gas amount (university)
# --------------------------------------------------------------------------- #
R_GAS = 8.314


def g_gas_moles(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    for _ in range(100):
        n_target = round(float(rng.uniform(0.2, 4.0)), 2)
        t_kelvin = 5 * int(rng.integers(50, 81))  # 250..400 K
        v_lit = round(float(rng.uniform(2.0, 40.0)), 1)
        p_raw = n_target * R_GAS * t_kelvin / v_lit
        p_kpa = round(p_raw, 1)
        if 20 <= p_kpa <= 600:
            break
    else:
        n_target, t_kelvin, v_lit, p_kpa = 1.0, 300, 10.0, round(1.0 * R_GAS * 300 / 10, 1)
    n_val = p_kpa * v_lit / (R_GAS * t_kelvin)
    q_en = (
        f"What amount of substance (in moles) does a gas occupy if its volume is {fmt(v_lit)} L at a "
        f"pressure of {fmt(p_kpa)} kPa and temperature {t_kelvin} K (R = 8.314 L*kPa/(mol*K))?"
    )
    q_ru = (
        f"Какое количество вещества (моль) занимает газ, если его объём равен {fmt(v_lit)} L при давлении "
        f"{fmt(p_kpa)} kPa и температуре {t_kelvin} K (R = 8.314 L*kPa/(mol*K))?"
    )
    steps_en = [
        f"Ideal gas law: p*V = n*R*T.",
        f"n = p*V / (R*T) = {fmt(p_kpa)} * {fmt(v_lit)} / (8.314 * {t_kelvin}) = {fmt(n_val)} mol.",
    ]
    steps_ru = [
        f"Уравнение Менделеева — Клапейрона: p*V = n*R*T.",
        f"n = p*V / (R*T) = {fmt(p_kpa)} * {fmt(v_lit)} / (8.314 * {t_kelvin}) = {fmt(n_val)} mol.",
    ]
    value, units = n_val, "mol"
    extras = [(p_kpa * v_lit / (R_GAS + t_kelvin), "added_r_and_t"), (n_val * 2, "factor_of_2"), (p_kpa * v_lit * R_GAS * t_kelvin, "multiplied_everything")]
    params: dict[str, Any] = {"p": p_kpa, "v": v_lit, "T": t_kelvin, "expected": n_val}
    d = PairDraft(SUBJECT, "", "", Difficulty.UNIVERSITY, atype, "", question_en=q_en, question_ru=q_ru)
    return _emit(d, rng, atype, value, units, steps_en, steps_ru, extras, params, Difficulty.UNIVERSITY, "gas_moles")


# --------------------------------------------------------------------------- #
# pH of strong acids and bases (university)
# --------------------------------------------------------------------------- #
def g_ph_strong(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    k = int(rng.integers(2, 6))  # c = 10^-k
    is_acid = bool(rng.integers(0, 2))
    c_val = 10.0**-k
    if is_acid:
        acid, acid_ru = (
            ("HCl", "хлороводородной кислоты HCl")
            if bool(rng.integers(0, 2))
            else ("HNO3", "азотной кислоты HNO3")
        )
        ph = float(k)
        q_en = f"Calculate the pH of an aqueous solution of the strong acid {acid} with molar concentration c = {c_val:g} mol/L."
        q_ru = f"Вычислите pH водного раствора сильной {acid_ru} с молярной концентрацией c = {c_val:g} mol/L."
        steps_en = [
            f"Strong acid: [H+] = c = {c_val:g} mol/L.",
            f"pH = -log10([H+]) = {k}.",
        ]
        steps_ru = [
            f"Сильная кислота диссоциирует полностью: [H+] = c = {c_val:g} mol/L.",
            f"pH = -log10([H+]) = {k}.",
        ]
        extras = [(float(k + 1), "off_by_one"), (float(14 - k), "acid_base_confusion"), (float(k - 1), "off_by_one")]
        params = {"kind": "acid", "c_exp": -k, "expected": ph}
    else:
        base, base_ru = ("NaOH", "гидроксида натрия NaOH") if bool(rng.integers(0, 2)) else ("KOH", "гидроксида калия KOH")
        ph = float(14 - k)
        q_en = f"Calculate the pH of an aqueous solution of the strong base {base} with molar concentration c = {c_val:g} mol/L."
        q_ru = f"Вычислите pH водного раствора сильного основания {base_ru} с молярной концентрацией c = {c_val:g} mol/L."
        steps_en = [
            f"Strong base: [OH-] = c = {c_val:g} mol/L, pOH = {k}.",
            f"pH = 14 - pOH = 14 - {k} = {14 - k}.",
        ]
        steps_ru = [
            f"Сильное основание диссоциирует полностью: [OH-] = c = {c_val:g} mol/L, pOH = {k}.",
            f"pH = 14 - pOH = 14 - {k} = {14 - k}.",
        ]
        extras = [(float(k), "forgot_14_minus"), (float(14 - k + 1), "off_by_one"), (float(7 + k), "sign_error")]
        params = {"kind": "base", "c_exp": -k, "expected": ph}
    d = PairDraft(SUBJECT, "", "", Difficulty.UNIVERSITY, atype, "", question_en=q_en, question_ru=q_ru)
    return _emit(d, rng, atype, ph, "", steps_en, steps_ru, extras, params, Difficulty.UNIVERSITY, "ph_strong")


# --------------------------------------------------------------------------- #
# Percent composition (school)
# --------------------------------------------------------------------------- #
def g_percent_comp(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    if atype == AnswerType.EXACT:
        formula, en_name, ru_name = _compound(rng)
        mm = molar_mass(formula)
        counts = parse_formula(formula)
        best = max(counts, key=lambda el: ATOMIC_WEIGHTS[el] * counts[el])
        q_en = f"Which element has the highest mass fraction in {en_name} ({formula})?"
        q_ru = f"Какой элемент имеет наибольшую массовую долю в {ru_name} ({formula})?"
        steps_en = [
            "Mass contributions: "
            + ", ".join(f"{el}: {fmt(ATOMIC_WEIGHTS[el] * cnt)}" for el, cnt in counts.items())
            + f" out of M = {fmt(mm)} g/mol.",
            f"The largest contribution is that of {best}.",
        ]
        steps_ru = [
            "Вклады элементов: "
            + ", ".join(f"{el}: {fmt(ATOMIC_WEIGHTS[el] * cnt)}" for el, cnt in counts.items())
            + f" из M = {fmt(mm)} g/mol.",
            f"Наибольший вклад у элемента {best}.",
        ]
        d = PairDraft(SUBJECT, "", "", Difficulty.SCHOOL, atype, "", question_en=q_en, question_ru=q_ru)
        d.canonical = best
        d.solution_en = sol_en(steps_en, best, "")
        d.solution_ru = sol_ru(steps_ru, best, "")
        d.params = {"formula": formula, "expected_text": best, "kind": "argmax_element"}
        return _finish(d, "percent_comp", Difficulty.SCHOOL)
    formula, en_name, ru_name = _compound(rng)
    counts = parse_formula(formula)
    mm = molar_mass(formula)
    el = sorted(counts)[int(rng.integers(0, len(counts)))]
    pct = 100.0 * ATOMIC_WEIGHTS[el] * counts[el] / mm
    q_en = f"Calculate the mass fraction of {EN_ELEMENT_NAMES.get(el, el)} in {en_name} ({formula}), in percent."
    q_ru = f"Вычислите массовую долю {RU_ELEMENT_NAMES.get(el, el)} в {ru_name} ({formula}) в процентах."
    steps_en = [
        f"M({formula}) = {fmt(mm)} g/mol; the element contributes {fmt(ATOMIC_WEIGHTS[el] * counts[el])} g/mol.",
        f"w = {fmt(ATOMIC_WEIGHTS[el] * counts[el])} / {fmt(mm)} * 100% = {fmt(pct)}%.",
    ]
    steps_ru = [
        f"M({formula}) = {fmt(mm)} g/mol; вклад элемента составляет {fmt(ATOMIC_WEIGHTS[el] * counts[el])} g/mol.",
        f"w = {fmt(ATOMIC_WEIGHTS[el] * counts[el])} / {fmt(mm)} * 100% = {fmt(pct)}%.",
    ]
    value, units = pct, "%"
    others = [e for e in counts if e != el]
    extras = [(100.0 - pct, "complement_error")]
    for e in others:
        extras.append((100.0 * ATOMIC_WEIGHTS[e] * counts[e] / mm, "wrong_element"))
    d = PairDraft(SUBJECT, "", "", Difficulty.SCHOOL, atype, "", question_en=q_en, question_ru=q_ru)
    return _emit(d, rng, atype, value, units, steps_en, steps_ru, extras,
                 {"formula": formula, "element": el, "expected": pct}, Difficulty.SCHOOL, "percent_comp")


# --------------------------------------------------------------------------- #
# Empirical formula (university)
# --------------------------------------------------------------------------- #
def _formula_double(formula: str) -> str:
    counts = parse_formula(formula)
    return "".join(el + (str(2 * cnt) if 2 * cnt > 1 else "") for el, cnt in counts.items())


def _formula_shift(formula: str, delta: int) -> str:
    counts = parse_formula(formula)
    if delta < 0 and min(counts.values()) <= 1:
        return formula  # cannot decrement; caller drops this candidate
    out = []
    for el, cnt in counts.items():
        new = cnt + delta
        out.append(el + (str(new) if new > 1 else ""))
    return "".join(out)


def _formula_reverse(formula: str) -> str:
    counts = parse_formula(formula)
    items = list(counts.items())[::-1]
    return "".join(el + (str(cnt) if cnt > 1 else "") for el, cnt in items)


def g_empirical(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    pcts, formula = EMPIRICAL_POOL[int(rng.integers(0, len(EMPIRICAL_POOL)))]
    el_list = list(pcts)
    frac_en = ", ".join(f"{pcts[el]}% {EN_ELEMENT_NAMES.get(el, el)}" for el in el_list)
    frac_ru = ", ".join(f"{RU_ELEMENT_NAMES.get(el, el)} — {fmt(pcts[el])}%" for el in el_list)
    q_en = f"The mass fractions of the elements in a compound are: {frac_en}. What is its empirical formula?"
    q_ru = f"Массовые доли элементов в веществе равны: {frac_ru}. Какова его простейшая (эмпирическая) формула?"
    moles_str_en = ", ".join(f"{pcts[el]} / {ATOMIC_WEIGHTS[el]} = {fmt(pcts[el] / ATOMIC_WEIGHTS[el])}" for el in el_list)
    steps_en = [
        f"Divide the mass fractions by the atomic weights: {moles_str_en}.",
        f"Dividing by the smallest value gives the simple integer ratio, hence the empirical formula {formula}.",
    ]
    steps_ru = [
        f"Разделим массовые доли на атомные массы: {moles_str_en}.",
        f"Деление на наименьшее значение даёт простое целое отношение, отсюда простейшая формула {formula}.",
    ]
    doubled = _formula_double(formula)
    up = _formula_shift(formula, 1)
    down = _formula_shift(formula, -1)
    reversed_f = _formula_reverse(formula)
    wrongs = [
        (doubled, "molecular_instead_of_empirical") if doubled != formula else None,
        (up, "subscript_off_by_one") if up != formula else None,
        (down, "subscript_off_by_one") if down != formula else None,
        (reversed_f, "element_order_swap") if reversed_f != formula else None,
    ]
    wrongs = [(t_, tag) for t_, tag in wrongs if t_ and t_ != formula]
    if len(wrongs) < 3:
        wrongs.append(("H2O", "unrelated_formula"))
    d = PairDraft(SUBJECT, "", "", Difficulty.UNIVERSITY, atype, "", question_en=q_en, question_ru=q_ru)
    wrongs3 = pick_distractors_str(rng, formula, wrongs[:6])
    opts = [formula] + [w for w, _ in wrongs3]
    d.mc_en = tuple(opts)
    d.mc_ru = tuple(opts)
    d.distractor_tags = tuple(tag for _, tag in wrongs3)
    d.canonical = formula
    d.solution_en = sol_en(steps_en, formula, "")
    d.solution_ru = sol_ru(steps_ru, formula, "")
    d.params = {"pcts": pcts, "expected_text": formula, "kind": "empirical"}
    return _finish(d, "empirical", Difficulty.UNIVERSITY)


# --------------------------------------------------------------------------- #
# Electron configuration (school, MC/exact)
# --------------------------------------------------------------------------- #
def g_econfig(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    i = int(rng.integers(0, len(ECONFIGS)))
    symbol, config = ECONFIGS[i]
    pool = [s for s, _ in ECONFIGS if s != symbol]
    others = [str(s) for s in rng.choice(pool, size=6, replace=False)]
    q_en = f"The ground-state electron configuration of an atom is {config}. Which element is it?"
    q_ru = f"Электронная конфигурация основного состояния атома: {config}. Какому элементу она соответствует?"
    steps_en = [
        f"Count the electrons: {config} totals {ATOMIC_NUMBERS[symbol]} electrons.",
        f"The neutral atom with Z = {ATOMIC_NUMBERS[symbol]} is {symbol}.",
    ]
    steps_ru = [
        f"Подсчитаем электроны: в конфигурации {config} всего {ATOMIC_NUMBERS[symbol]} электронов.",
        f"Нейтральный атом с Z = {ATOMIC_NUMBERS[symbol]} — это {symbol}.",
    ]
    wrongs = [(str(s), "neighbor_configuration") for s in others]
    d = PairDraft(SUBJECT, "", "", Difficulty.SCHOOL, atype, "", question_en=q_en, question_ru=q_ru)
    if atype == AnswerType.EXACT:
        d.canonical = symbol
        d.solution_en = sol_en(steps_en, symbol, "")
        d.solution_ru = sol_ru(steps_ru, symbol, "")
        d.params = {"config": config, "expected_text": symbol, "kind": "econfig"}
        return _finish(d, "econfig", Difficulty.SCHOOL)
    wrongs3 = pick_distractors_str(rng, symbol, wrongs)
    opts = [symbol] + [w for w, _ in wrongs3]
    d.mc_en = tuple(opts)
    d.mc_ru = tuple(opts)
    d.distractor_tags = tuple(tag for _, tag in wrongs3)
    d.canonical = symbol
    d.solution_en = sol_en(steps_en, symbol, "")
    d.solution_ru = sol_ru(steps_ru, symbol, "")
    d.params = {"config": config, "expected_text": symbol, "kind": "econfig"}
    return _finish(d, "econfig", Difficulty.SCHOOL)


# --------------------------------------------------------------------------- #
# Reaction products / precipitates (school + university)
# --------------------------------------------------------------------------- #
SOLUBLE_BY_ANION = {
    "NO3": None,  # all nitrates soluble -> no precipitate
    "Cl": ["Ag", "Pb"],
    "Br": ["Ag", "Pb"],
    "I": ["Ag", "Pb"],
    "SO4": ["Ba", "Pb", "Ca"],
    "CO3": ["Na", "K"],  # carbonates insoluble except Na/K
    "OH": ["Na", "K"],  # hydroxides insoluble except Na/K
}


def g_reactions(rng: np.random.Generator, idx: int, atype: AnswerType, difficulty: Difficulty) -> PairDraft:
    if bool(rng.integers(0, 2)):  # precipitation
        r1, r2, precip = PRECIP_REACTIONS[int(rng.integers(0, len(PRECIP_REACTIONS)))]
        q_en = f"Aqueous solutions of {r1} and {r2} are mixed and a precipitate forms. Which substance precipitates?"
        q_ru = f"Смешивают водные растворы {r1} и {r2}, и образуется осадок. Какое вещество выпадает в осадок?"
        steps_en = [
            f"Ion exchange swaps the partners: the cations pair with the opposite anions.",
            f"By the solubility rules the insoluble product {precip} precipitates.",
        ]
        steps_ru = [
            f"Реакция обмена меняет партнёров: катионы соединяются с чужими анионами.",
            f"По правилам растворимости нерастворимый продукт {precip} выпадает в осадок.",
        ]
        ans = precip
        others = [r1, r2, "NaNO3", "KNO3", "NaCl", "H2O"]
        params = {"kind": "precip", "r1": r1, "r2": r2, "expected_text": precip}
    else:  # gas evolution
        r1, r2, gas, kind = GAS_REACTIONS[int(rng.integers(0, len(GAS_REACTIONS)))]
        if kind == "metal_acid":
            q_en = f"{r1} reacts with an excess of dilute {r2}. Which gas is released?"
            q_ru = f"{r1} реагирует с избытком разбавленной {r2}. Какой газ при этом выделяется?"
            steps_en = [
                f"A metal above hydrogen in the activity series displaces it from the acid.",
                f"The evolved gas is {gas}.",
            ]
            steps_ru = [
                f"Металл, стоящий в ряду активности до водорода, вытесняет его из кислоты.",
                f"Выделяющийся газ — это {gas}.",
            ]
        elif kind == "acid_carbonate":
            q_en = f"{r1} reacts with an excess of dilute {r2}. Which gas is released?"
            q_ru = f"{r1} реагирует с избытком разбавленной {r2}. Какой газ при этом выделяется?"
            steps_en = [
                f"An acid acting on a carbonate releases carbon dioxide.",
                f"The evolved gas is {gas}.",
            ]
            steps_ru = [
                f"При действии кислоты на карбонат выделяется углекислый газ.",
                f"Выделяющийся газ — это {gas}.",
            ]
        else:
            q_en = f"{r1} burns completely in {r2}. Which gas is the main combustion product?"
            q_ru = f"{r1} полностью сгорает в {r2}. Какой газ является основным продуктом горения?"
            steps_en = [
                f"Complete combustion of a hydrocarbon or carbon gives carbon dioxide and water.",
                f"The gaseous product is {gas}.",
            ]
            steps_ru = [
                f"Полное сгорание углеводорода или углерода даёт углекислый газ и воду.",
                f"Газообразный продукт — это {gas}.",
            ]
        ans = gas
        others = ["O2", "H2", "N2", "H2O", "CO"]
        params = {"kind": "gas", "r1": r1, "r2": r2, "expected_text": gas}
    d = PairDraft(SUBJECT, "", "", difficulty, atype, "", question_en=q_en, question_ru=q_ru)
    if atype == AnswerType.EXACT:
        d.canonical = ans
        d.solution_en = sol_en(steps_en, ans, "")
        d.solution_ru = sol_ru(steps_ru, ans, "")
        d.params = params
        return _finish(d, "reactions", difficulty)
    wrongs = [(o, "wrong_product") for o in others if o != ans]
    wrongs3 = pick_distractors_str(rng, ans, wrongs)
    opts = [ans] + [w for w, _ in wrongs3]
    d.mc_en = tuple(opts)
    d.mc_ru = tuple(opts)
    d.distractor_tags = tuple(tag for _, tag in wrongs3)
    d.canonical = ans
    d.solution_en = sol_en(steps_en, ans, "")
    d.solution_ru = sol_ru(steps_ru, ans, "")
    d.params = params
    return _finish(d, "reactions", difficulty)


def g_reactions_school(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    return g_reactions(rng, idx, atype, Difficulty.SCHOOL)


def g_reactions_uni(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    return g_reactions(rng, idx, atype, Difficulty.UNIVERSITY)


# --------------------------------------------------------------------------- #
# Balancing equations (university, MC)
# --------------------------------------------------------------------------- #
def g_balancing(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    eq, correct, wrong_seed = BALANCING_POOL[int(rng.integers(0, len(BALANCING_POOL)))]
    letters = "abcd"[: len(correct)]
    rhs = eq.split("->")[1].strip()
    lhs = eq.split("->")[0].strip()
    q_en = f"Balance the equation {eq}. Which set of coefficients ({', '.join(letters)}) makes it balanced?"
    q_ru = f"Уравняйте уравнение {eq}. Какой набор коэффициентов ({', '.join(letters)}) его уравнивает?"
    corr_str = "(" + ", ".join(str(x) for x in correct) + ")"
    wrongs: list[tuple[str, str]] = []
    for j in range(len(correct)):
        w = list(correct)
        w[j] = w[j] + 1
        wrongs.append(("(" + ", ".join(str(x) for x in w) + ")", "off_by_one"))
    w2 = list(correct)
    if len(w2) >= 2:
        w2[0], w2[1] = w2[1], w2[0]
        wrongs.append(("(" + ", ".join(str(x) for x in w2) + ")", "swapped_coefficients"))
    wrongs.append(("(" + ", ".join(str(x) for x in wrong_seed) + ")", "stoichiometry_slip"))
    steps_en = [
        f"Count the atoms of each element on both sides of {eq}.",
        f"Element-by-element balance requires the coefficients {corr_str}.",
    ]
    steps_ru = [
        f"Подсчитаем атомы каждого элемента в обеих частях уравнения {eq}.",
        f"Баланс по каждому элементу достигается коэффициентами {corr_str}.",
    ]
    d = PairDraft(SUBJECT, "", "", Difficulty.UNIVERSITY, atype, "", question_en=q_en, question_ru=q_ru)
    wrongs3 = pick_distractors_str(rng, corr_str, wrongs)
    opts = [corr_str] + [w for w, _ in wrongs3]
    d.mc_en = tuple(opts)
    d.mc_ru = tuple(opts)
    d.distractor_tags = tuple(tag for _, tag in wrongs3)
    d.canonical = corr_str
    d.solution_en = sol_en(steps_en, corr_str, "")
    d.solution_ru = sol_ru(steps_ru, corr_str, "")
    d.params = {
        "eq": eq, "expected_text": corr_str, "coeffs": list(correct), "kind": "balancing",
        "distractor_texts": [w for w, _ in wrongs3],
    }
    return _finish(d, "balancing", Difficulty.UNIVERSITY)


# --------------------------------------------------------------------------- #
# Limiting reagent (olympiad)
# --------------------------------------------------------------------------- #
def g_limiting(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    eq, coeffs, product = LIMITING_POOL[int(rng.integers(0, len(LIMITING_POOL)))]
    c1, c2, cp = coeffs
    lhs_parts = [re.sub(r"^[0-9]+\s*", "", p) for p in eq.split("->")[0].split(" + ")]
    m1_formula, m2_formula = lhs_parts[0], lhs_parts[1]
    mm1, mm2, mmp = molar_mass(m1_formula), molar_mass(m2_formula), molar_mass(product)
    n1 = float(rng.integers(1, 7)) + float(rng.choice([0.0, 0.25, 0.5]))
    n2 = float(rng.integers(1, 9)) + float(rng.choice([0.0, 0.25, 0.5]))
    mass1 = round(n1 * mm1, 2)
    mass2 = round(n2 * mm2, 2)
    n1p = mass1 / mm1
    n2p = mass2 / mm2
    lim_first = n1p / c1 <= n2p / c2
    limiting_formula = m1_formula if lim_first else m2_formula
    n_product = min(n1p / c1, n2p / c2) * cp
    mass_product = n_product * mmp
    q_en = (
        f"In the reaction {eq}, {fmt(mass1)} g of {m1_formula} is mixed with {fmt(mass2)} g of {m2_formula}. "
        f"What mass of {product} is formed?"
    )
    q_ru = (
        f"В реакции {eq} смешали {fmt(mass1)} g вещества {m1_formula} и {fmt(mass2)} g вещества {m2_formula}. "
        f"Какая масса {product} при этом образуется?"
    )
    steps_en = [
        f"n({m1_formula}) = {fmt(mass1)} / {fmt(mm1)} = {fmt(n1p)} mol; n({m2_formula}) = {fmt(mass2)} / {fmt(mm2)} = {fmt(n2p)} mol.",
        f"Divide by the coefficients: {fmt(n1p)} / {c1} = {fmt(n1p / c1)} and {fmt(n2p)} / {c2} = {fmt(n2p / c2)}; "
        f"the limiting reagent is {limiting_formula}.",
        f"n({product}) = {fmt(n_product)} mol, so m = {fmt(n_product)} * {fmt(mmp)} = {fmt(mass_product)} g.",
    ]
    steps_ru = [
        f"n({m1_formula}) = {fmt(mass1)} / {fmt(mm1)} = {fmt(n1p)} mol; n({m2_formula}) = {fmt(mass2)} / {fmt(mm2)} = {fmt(n2p)} mol.",
        f"Разделим на коэффициенты: {fmt(n1p)} / {c1} = {fmt(n1p / c1)} и {fmt(n2p)} / {c2} = {fmt(n2p / c2)}; "
        f"лимитирующий реагент — {limiting_formula}.",
        f"n({product}) = {fmt(n_product)} mol, поэтому m = {fmt(n_product)} * {fmt(mmp)} = {fmt(mass_product)} g.",
    ]
    d = PairDraft(SUBJECT, "", "", Difficulty.OLYMPIAD, atype, "", question_en=q_en, question_ru=q_ru)
    if atype == AnswerType.EXACT:
        d.canonical = limiting_formula
        d.solution_en = sol_en(steps_en[:2] + [f"The limiting reagent is {limiting_formula}."], limiting_formula, "")
        d.solution_ru = sol_ru(steps_ru[:2] + [f"Лимитирующий реагент — {limiting_formula}."], limiting_formula, "")
        d.params = {
            "eq": eq, "coeffs": coeffs, "mass1": mass1, "mass2": mass2,
            "f1": m1_formula, "f2": m2_formula, "expected_text": limiting_formula, "kind": "limiting_formula",
        }
        return _finish(d, "limiting", Difficulty.OLYMPIAD)
    if atype == AnswerType.MC:
        extras = [
            (n1p / c1 * cp * mmp, "assumed_first_limiting"),
            (n2p / c2 * cp * mmp, "assumed_second_limiting"),
        ]
        if abs(extras[0][0] - mass_product) < 1e-9:
            extras[0] = (mass_product * 1.5, "factor_of_1_5")
        if abs(extras[1][0] - mass_product) < 1e-9:
            extras[1] = (mass_product * 0.5, "factor_of_2_half")
        _mc_numeric(d, rng, mass_product, _std_pool(mass_product, extras), "g")
        d.solution_en = sol_en(steps_en, fmt(mass_product), "g")
        d.solution_ru = sol_ru(steps_ru, fmt(mass_product), "g")
        d.params = {
            "eq": eq, "coeffs": coeffs, "mass1": mass1, "mass2": mass2,
            "f1": m1_formula, "f2": m2_formula, "expected": mass_product, "kind": "product_mass",
        }
        return _finish(d, "limiting", Difficulty.OLYMPIAD)
    _set_numeric(d, mass_product, "g")
    d.solution_en = sol_en(steps_en, d.canonical, "g")
    d.solution_ru = sol_ru(steps_ru, d.canonical, "g")
    d.params = {
        "eq": eq, "coeffs": coeffs, "mass1": mass1, "mass2": mass2,
        "f1": m1_formula, "f2": m2_formula, "expected": mass_product, "kind": "product_mass",
    }
    return _finish(d, "limiting", Difficulty.OLYMPIAD)


GENERATORS: dict[str, Any] = {
    "molar_mass": g_molar_mass,
    "stoich_mass": g_stoich_school,
    "stoich_mass_uni": g_stoich_uni,
    "molarity": g_molarity_school,
    "molarity_uni": g_molarity_uni,
    "dilution": g_dilution,
    "gas_moles": g_gas_moles,
    "ph_strong": g_ph_strong,
    "percent_comp": g_percent_comp,
    "empirical": g_empirical,
    "econfig": g_econfig,
    "reactions": g_reactions_school,
    "reactions_uni": g_reactions_uni,
    "balancing": g_balancing,
    "limiting": g_limiting,
}

KEY_ALIASES: dict[str, str] = {
    "stoich_mass_uni": "stoich_mass",
    "molarity_uni": "molarity",
    "reactions_uni": "reactions",
}

__all__ = [
    "SUBJECT", "PREFIX", "TOPICS", "RUBRICS", "SPEC", "GENERATORS", "KEY_ALIASES",
    "ATOMIC_WEIGHTS", "parse_formula", "molar_mass",
]
