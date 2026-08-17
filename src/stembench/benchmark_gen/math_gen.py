"""Original bilingual (ru/en) mathematics item generators.

Every generator receives a per-pair ``numpy`` Generator (derived from the global
seed, subject, topic key and pair index) plus the requested answer type, and
returns a :class:`~stembench.benchmark_gen._core.PairDraft` whose answers are
computed in code.  ``verify.py`` re-derives every answer through a different
code path (exact integer arithmetic with ``fractions.Fraction``, root
substitution, brute-force enumeration, central-difference derivatives, ...).

Decimal separator "." and untranslated SI units are used in both languages
(see ``_core`` module docstring).
"""

from __future__ import annotations

import math
from fractions import Fraction
from itertools import combinations, product
from typing import Any

import numpy as np

from stembench.schemas import AnswerType, Difficulty, Subject

from ._core import (
    PairDraft,
    Person,
    PEOPLE,
    fmt,
    frac_str,
    poly_str,
    ru_past,
    ru_plural,
    ru_was,
    pick_distractors,
    sol_en,
    sol_ru,
)

SUBJECT = Subject.MATH
PREFIX = "MATH"

# Display topic per machine key.
TOPICS: dict[str, str] = {
    "arith_word": "arithmetic word problems",
    "linear_eq": "linear equations",
    "quad_eq": "quadratic equations",
    "percent": "percentages",
    "sequences": "arithmetic and geometric sequences",
    "derivatives": "derivatives of polynomials",
    "log_exp": "logarithms and exponentials",
    "numtheory": "number theory: remainders and divisibility",
    "geometry_area": "planar geometry: area and perimeter",
    "sys_lin2": "systems of two linear equations",
    "inequalities": "inequalities",
    "trig": "trigonometry basics",
    "prob_comb": "probability and combinatorics",
}

# Difficulty rubrics (one short sentence per topic x difficulty, both languages).
RUBRICS: dict[tuple[str, Difficulty], tuple[str, str]] = {
    ("arith_word", Difficulty.SCHOOL): (
        "Two arithmetic operations with whole numbers within a few thousand; standard school curriculum.",
        "Два арифметических действия с целыми числами в пределах нескольких тысяч; стандартная школьная программа.",
    ),
    ("linear_eq", Difficulty.SCHOOL): (
        "Single linear equation solved in one step; standard school algebra.",
        "Линейное уравнение, решаемое в одно-два действия; стандартная школьная алгебра.",
    ),
    ("quad_eq", Difficulty.UNIVERSITY): (
        "Quadratic with integer roots; requires the discriminant or Vieta formulas.",
        "Квадратное уравнение с целыми корнями; требуется дискриминант или теорема Виета.",
    ),
    ("percent", Difficulty.SCHOOL): (
        "Basic percent-of and percent-change computation on round numbers.",
        "Базовые вычисления с процентами на круглых числах.",
    ),
    ("sequences", Difficulty.UNIVERSITY): (
        "nth-term and partial-sum formulas for arithmetic and geometric progressions.",
        "Формулы n-го члена и суммы для арифметической и геометрической прогрессий.",
    ),
    ("prob_comb", Difficulty.SCHOOL): (
        "Classical probability with a small sample space or a direct counting rule.",
        "Классическое определение вероятности с малым пространством исходов или прямое правило подсчёта.",
    ),
    ("prob_comb", Difficulty.UNIVERSITY): (
        "Probability of a compound event or elementary combinatorial counting.",
        "Вероятность сложного события или элементарный комбинаторный подсчёт.",
    ),
    ("prob_comb", Difficulty.OLYMPIAD): (
        "Multi-step combinatorial count combining two binomial coefficients or repetitions.",
        "Многошаговый комбинаторный подсчёт с двумя биномиальными коэффициентами или с повторениями.",
    ),
    ("derivatives", Difficulty.UNIVERSITY): (
        "Derivative of a cubic polynomial evaluated at a point; power rule only.",
        "Производная кубического многочлена в точке; требуется только правило степени.",
    ),
    ("log_exp", Difficulty.UNIVERSITY): (
        "Logarithm or exponent equation with an integer answer on a power base.",
        "Логарифмическое или показательное уравнение с целым ответом на степенном основании.",
    ),
    ("numtheory", Difficulty.OLYMPIAD): (
        "Remainders, divisibility or the Chinese remainder theorem beyond the school syllabus.",
        "Остатки, делимость или китайская теорема об остатках за пределами школьной программы.",
    ),
    ("geometry_area", Difficulty.SCHOOL): (
        "Area or perimeter of a rectangle, square, circle or triangle from basic formulas.",
        "Площадь или периметр прямоугольника, квадрата, круга или треугольника по базовым формулам.",
    ),
    ("geometry_area", Difficulty.UNIVERSITY): (
        "Trapezoid area or a coordinate-triangle area via the shoelace computation.",
        "Площадь трапеции или треугольника на координатах через формулу площади.",
    ),
    ("sys_lin2", Difficulty.UNIVERSITY): (
        "System of two linear equations with integer solution; elimination or substitution.",
        "Система двух линейных уравнений с целым решением; метод сложения или подстановка.",
    ),
    ("inequalities", Difficulty.UNIVERSITY): (
        "Linear inequality requiring careful handling of the comparison direction.",
        "Линейное неравенство, требующее внимательной работы со знаком неравенства.",
    ),
    ("inequalities", Difficulty.OLYMPIAD): (
        "Quadratic inequality solved from the sign of the parabola between the roots.",
        "Квадратное неравенство, решаемое по расположению параболы между корнями.",
    ),
    ("trig", Difficulty.UNIVERSITY): (
        "Sine, cosine or tangent in a right triangle or at a special angle.",
        "Синус, косинус или тангенс в прямоугольном треугольнике или в особом угле.",
    ),
}

SPEC = [
    # topic_key, difficulty, count, answer_type  (math: 244 pairs)
    ("arith_word", Difficulty.SCHOOL, 20, AnswerType.NUMERIC),
    ("arith_word", Difficulty.SCHOOL, 6, AnswerType.MC),
    ("linear_eq", Difficulty.SCHOOL, 16, AnswerType.NUMERIC),
    ("linear_eq", Difficulty.SCHOOL, 2, AnswerType.MC),
    ("percent", Difficulty.SCHOOL, 16, AnswerType.NUMERIC),
    ("percent", Difficulty.SCHOOL, 6, AnswerType.MC),
    ("geometry_area", Difficulty.SCHOOL, 10, AnswerType.NUMERIC),
    ("geometry_area", Difficulty.SCHOOL, 4, AnswerType.MC),
    ("geometry_area", Difficulty.UNIVERSITY, 4, AnswerType.MC),
    ("prob_comb", Difficulty.SCHOOL, 8, AnswerType.MC),
    ("prob_comb", Difficulty.SCHOOL, 10, AnswerType.EXACT),
    ("quad_eq", Difficulty.UNIVERSITY, 10, AnswerType.MC),
    ("quad_eq", Difficulty.UNIVERSITY, 4, AnswerType.NUMERIC),
    ("sequences", Difficulty.UNIVERSITY, 12, AnswerType.NUMERIC),
    ("sequences", Difficulty.UNIVERSITY, 6, AnswerType.MC),
    ("derivatives", Difficulty.UNIVERSITY, 12, AnswerType.NUMERIC),
    ("derivatives", Difficulty.UNIVERSITY, 6, AnswerType.MC),
    ("log_exp", Difficulty.UNIVERSITY, 14, AnswerType.MC),
    ("log_exp", Difficulty.UNIVERSITY, 4, AnswerType.EXACT),
    ("sys_lin2", Difficulty.UNIVERSITY, 12, AnswerType.NUMERIC),
    ("sys_lin2", Difficulty.UNIVERSITY, 4, AnswerType.MC),
    ("trig", Difficulty.UNIVERSITY, 8, AnswerType.MC),
    ("trig", Difficulty.UNIVERSITY, 2, AnswerType.NUMERIC),
    ("inequalities", Difficulty.UNIVERSITY, 6, AnswerType.MC),
    ("prob_comb_uni", Difficulty.UNIVERSITY, 4, AnswerType.MC),
    ("prob_comb_uni", Difficulty.UNIVERSITY, 2, AnswerType.EXACT),
    ("numtheory", Difficulty.OLYMPIAD, 14, AnswerType.EXACT),
    ("numtheory", Difficulty.OLYMPIAD, 2, AnswerType.MC),
    ("prob_comb_olym", Difficulty.OLYMPIAD, 12, AnswerType.MC),
    ("prob_comb_olym", Difficulty.OLYMPIAD, 2, AnswerType.EXACT),
    ("inequalities_olym", Difficulty.OLYMPIAD, 6, AnswerType.MC),
]

SHOP_ITEMS: tuple[tuple[str, str, str, str, str], ...] = (
    # en plural, ru one, ru few, ru many, en singular-for-price
    ("notebooks", "тетрадь", "тетради", "тетрадей", "notebook"),
    ("pens", "ручка", "ручки", "ручек", "pen"),
    ("chocolates", "шоколадка", "шоколадки", "шоколадок", "chocolate bar"),
    ("books", "книга", "книги", "книг", "book"),
    ("folders", "папка", "папки", "папок", "folder"),
    ("markers", "фломастер", "фломастера", "фломастеров", "marker"),
    ("photo albums", "альбом", "альбома", "альбомов", "photo album"),
    ("rulers", "линейка", "линейки", "линеек", "ruler"),
    ("pencils", "карандаш", "карандаша", "карандашей", "pencil"),
    ("calculators", "калькулятор", "калькулятора", "калькуляторов", "calculator"),
)


def _person(rng: np.random.Generator) -> Person:
    return PEOPLE[int(rng.integers(0, len(PEOPLE)))]


def _set_numeric(
    draft: PairDraft,
    value: float,
    units: str,
) -> None:
    draft.canonical = fmt(value)
    draft.numeric_value = float(value)
    draft.units = units


def _mc_numeric(
    draft: PairDraft,
    rng: np.random.Generator,
    value: float,
    candidates: list[tuple[float, str]],
    units: str = "",
) -> None:
    wrongs = pick_distractors(rng, value, candidates)
    opts = [fmt(value)] + [fmt(w) for w, _ in wrongs]
    tail = f" {units}" if units else ""
    draft.mc_en = tuple(o + tail for o in opts)
    draft.mc_ru = tuple(o + tail for o in opts)  # numeric options: units not translated
    draft.distractor_tags = tuple(tag for _, tag in wrongs)
    draft.params["choice_values"] = [float(value)] + [float(w) for w, _ in wrongs]
    draft.numeric_value = float(value)
    draft.units = units


def _std_pool(v: float, extras: list[tuple[float, str]] | None = None) -> list[tuple[float, str]]:
    """Generic plausible-error candidates around v (never random junk)."""
    pool: list[tuple[float, str]] = [
        (v + 1, "off_by_one"),
        (v - 1, "off_by_one"),
        (2 * v, "factor_of_2"),
        (v + 2, "off_by_two"),
    ]
    if float(v).is_integer() and v >= 4 and int(v) % 2 == 0:
        pool.append((v / 2, "factor_of_2_half"))
    if float(v).is_integer() and v >= 20 and int(v) % 10 == 0:
        pool.append((v / 10, "unit_slip_x10"))
    if v != 0:
        pool.append((-v, "sign_error"))
    if extras:
        pool.extend(extras)
    return pool


def _finish(draft: PairDraft, subject_topic: str, key: str, difficulty: Difficulty) -> PairDraft:
    draft.topic = subject_topic
    draft.topic_key = key
    draft.difficulty = difficulty
    draft.subject = SUBJECT
    return draft


# --------------------------------------------------------------------------- #
# 1. Arithmetic word problems
# --------------------------------------------------------------------------- #
def g_arith_word(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    variant = int(rng.integers(0, 4))
    if variant == 0:
        p = _person(rng)
        en_pl, one, few, many, _en_single = SHOP_ITEMS[int(rng.integers(0, len(SHOP_ITEMS)))]
        k = int(rng.integers(5, 16))
        price = 5 * int(rng.integers(3, 20))
        note = int(rng.choice([1000, 2000, 5000]))
        while note <= k * price:
            note *= 2 if note < 5000 else 1
            if note > 5000:
                price = 5 * int(rng.integers(3, 10))
                note = 1000
        total = k * price
        change = note - total
        bought = ru_past(p.gender, "купил", "купила")
        paid = ru_past(p.gender, "заплатил", "заплатила")
        got = ru_past(p.gender, "получил", "получила")
        q_en = (
            f"{p.en} bought {k} {en_pl} at {price} rubles each and paid with a {note}-ruble note. "
            f"How much change did {p.en} receive?"
        )
        q_ru = (
            f"{p.ru_nom} {bought} {k} {ru_plural(k, one, few, many)} по {price} рублей за штуку "
            f"и {paid} купюрой в {note} рублей. Сколько рублей сдачи {got} {p.ru_nom}?"
        )
        s_en = [f"Cost of the purchase: {k} * {price} = {total} rubles.", f"Change: {note} - {total} = {change} rubles."]
        s_ru = [f"Стоимость покупки: {k} * {price} = {total} рублей.", f"Сдача: {note} - {total} = {change} рублей."]
        value, units = float(change), "RUB"
        params: dict[str, Any] = {"variant": "shop", "k": k, "price": price, "note": note, "expected": change}
        extras = [(float(total), "gave_total_instead_of_change"), (float(note), "returned_banknote")]
    elif variant == 1:
        p = _person(rng)
        f_cnt = int(rng.integers(3, 8))
        c = int(rng.integers(2, 10))
        r = int(rng.integers(0, f_cnt))
        n_all = c * f_cnt + r
        shared = ru_past(p.gender, "раздал", "раздала")
        q_en = (
            f"{p.en} has {n_all} candies and shares them equally between {f_cnt} friends. "
            f"How many candies does each friend receive?"
        )
        q_ru = (
            f"У {p.ru_gen} {n_all} конфет. {p.ru_nom} {shared} их {f_cnt} друзьям поровну. "
            f"Сколько конфет досталось каждому другу?"
        )
        s_en = [
            f"Division with remainder: {n_all} = {c} * {f_cnt} + {r}.",
            f"Each friend receives {c} candies; {r} candies are left over.",
        ]
        s_ru = [
            f"Деление с остатком: {n_all} = {c} * {f_cnt} + {r}.",
            f"Каждому другу достаётся {c} конфет; остаётся {r} конфет.",
        ]
        value, units = float(c), ""
        params = {"variant": "share", "n": n_all, "friends": f_cnt, "expected": c}
        extras = [(float(r), "returned_remainder"), (float(f_cnt), "used_divisor")]
    elif variant == 2:
        v = 5 * int(rng.integers(8, 19))  # 40..90 km/h
        t = int(rng.integers(2, 7))
        s = v * t
        q_en = f"A bus travels from town A to town B at a constant speed of {v} km/h. What distance does it cover in {t} h?"
        q_ru = f"Автобус движется из города А в город Б с постоянной скоростью {v} km/h. Какое расстояние он проедет за {t} h?"
        s_en = [f"Distance: s = v * t = {v} * {t} = {s} km."]
        s_ru = [f"Расстояние: s = v * t = {v} * {t} = {s} km."]
        value, units = float(s), "km"
        params = {"variant": "distance", "v": v, "t": t, "expected": s}
        extras = [(float(v + t), "added_speed_and_time"), (float(v), "forgot_time")]
    else:
        p = _person(rng)
        a = 50 * int(rng.integers(4, 11))  # 200..500 per day
        d = int(rng.integers(5, 21))
        earned = a * d
        b = 50 * int(rng.integers(2, max(3, earned // 50)))
        left = earned - b
        his = "у него" if p.gender == "m" else "у неё"
        earned_v = ru_past(p.gender, "заработал", "заработала")
        spent_v = ru_past(p.gender, "потратил", "потратила")
        q_en = (
            f"{p.en} earned {a} rubles per day for {d} days and then spent {b} rubles. "
            f"How much money does {p.en} have left?"
        )
        q_ru = (
            f"{p.ru_nom} {earned_v} {a} рублей в день в течение {d} дней, а затем {spent_v} {b} рублей. "
            f"Сколько денег {his} осталось?"
        )
        s_en = [f"Earnings: {a} * {d} = {earned} rubles.", f"Remaining: {earned} - {b} = {left} rubles."]
        s_ru = [f"Заработок: {a} * {d} = {earned} рублей.", f"Осталось: {earned} - {b} = {left} рублей."]
        value, units = float(left), "RUB"
        params = {"variant": "savings", "a": a, "d": d, "b": b, "expected": left}
        extras = [(float(earned), "forgot_spending"), (float(earned + b), "added_spending"), (float(a + d), "added_rates")]

    d_ = PairDraft(
        subject=SUBJECT, topic="", topic_key="", difficulty=Difficulty.SCHOOL, answer_type=atype,
        canonical="", question_en=q_en, question_ru=q_ru,
    )
    if atype == AnswerType.NUMERIC:
        _set_numeric(d_, value, units)
        d_.solution_en = sol_en(s_en, d_.canonical, units)
        d_.solution_ru = sol_ru(s_ru, d_.canonical, units)
    else:
        _mc_numeric(d_, rng, value, _std_pool(value, extras), units)
        d_.solution_en = sol_en(s_en, fmt(value), units)
        d_.solution_ru = sol_ru(s_ru, fmt(value), units)
    d_.params = params
    return _finish(d_, TOPICS["arith_word"], "arith_word", Difficulty.SCHOOL)


# --------------------------------------------------------------------------- #
# 2. Linear equations
# --------------------------------------------------------------------------- #
def g_linear_eq(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    form = int(rng.integers(0, 3))
    a = int(rng.integers(2, 10))
    x0 = int(rng.integers(2, 13))
    if form == 2:
        h = int(rng.integers(2, 10))
        c = a * (x0 + h)
        eq = f"{a}(x + {h}) = {c}"
        steps_en = [
            f"Expand the left side: {a}x + {a * h} = {c}.",
            f"Isolate the term with x: {a}x = {c} - {a * h} = {a * x0}.",
            f"x = {a * x0} / {a} = {x0}.",
        ]
        steps_ru = [
            f"Раскроем скобки: {a}x + {a * h} = {c}.",
            f"Перенесём свободный член: {a}x = {c} - {a * h} = {a * x0}.",
            f"x = {a * x0} / {a} = {x0}.",
        ]
        params: dict[str, Any] = {"a": a, "h": h, "c": c, "form": "paren", "expected": x0}
    else:
        b = int(rng.integers(-9, 10))
        if b == 0:
            b = 5
        c = a * x0 + b
        eq = f"{poly_str([(a, 'x'), (b, '')])} = {c}"
        steps_en = [
            f"Move the constant to the right: {a}x = {c} - ({b}) = {c - b}.",
            f"Divide by {a}: x = {c - b} / {a} = {x0}.",
        ]
        steps_ru = [
            f"Перенесём свободный член вправо: {a}x = {c} - ({b}) = {c - b}.",
            f"Разделим на {a}: x = {c - b} / {a} = {x0}.",
        ]
        params = {"a": a, "b": b, "c": c, "form": "simple", "expected": x0}
    phr = int(rng.integers(0, 3))
    if phr == 0:
        q_en = f"Given the equation {eq}, what is the value of x?"
        q_ru = f"Дано уравнение {eq}. Чему равно значение x?"
    elif phr == 1:
        q_en = f"What value of x satisfies the equation {eq}?"
        q_ru = f"Какое значение x удовлетворяет уравнению {eq}?"
    else:
        q_en = f"Solve the equation {eq} for x. What is x?"
        q_ru = f"Решите уравнение {eq}. Чему равен корень?"
    value = float(x0)
    pool = _std_pool(value, [(float(c) / a if a else 0.0, "forgot_constant")])
    if form != 2:
        pool += [
            (float(c + b) / a, "sign_error_moving_constant"),
            (float(c - b), "forgot_division"),
        ]
    else:
        pool += [(float(x0 + h), "sign_error"), (float(c / a), "forgot_subtraction")]
    d_ = PairDraft(
        SUBJECT, "", "", Difficulty.SCHOOL, atype, "",
        question_en=q_en, question_ru=q_ru,
    )
    if atype == AnswerType.NUMERIC:
        _set_numeric(d_, value, "")
        d_.solution_en = sol_en(steps_en, d_.canonical, "")
        d_.solution_ru = sol_ru(steps_ru, d_.canonical, "")
    else:
        _mc_numeric(d_, rng, value, pool, "")
        d_.solution_en = sol_en(steps_en, fmt(value), "")
        d_.solution_ru = sol_ru(steps_ru, fmt(value), "")
    d_.params = params
    return _finish(d_, TOPICS["linear_eq"], "linear_eq", Difficulty.SCHOOL)


# --------------------------------------------------------------------------- #
# 3. Quadratic equations
# --------------------------------------------------------------------------- #
def g_quad_eq(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    while True:
        r1 = int(rng.integers(-9, 6))
        r2 = int(rng.integers(r1 + 1, 10))
        if abs(r1 - r2) >= 1 and not (r1 == 0 and r2 == 0):
            break
    p = -(r1 + r2)
    q = r1 * r2
    disc = p * p - 4 * q
    if disc < 0 or int(math.isqrt(disc)) ** 2 != disc:  # cannot happen with integer roots
        raise ValueError("non-nice quadratic")
    eq = f"{poly_str([(1, 'x^2'), (p, 'x'), (q, '')])} = 0"
    sd = int(math.isqrt(disc))
    phr = int(rng.integers(0, 3))
    if phr == 0:
        q_en = f"What is the larger root of the equation {eq}?"
        q_ru = f"Чему равен больший корень уравнения {eq}?"
    elif phr == 1:
        q_en = f"The equation {eq} has two roots. Which of them is larger?"
        q_ru = f"Уравнение {eq} имеет два корня. Какой из них больше?"
    else:
        q_en = f"Solve the equation {eq}: what is its larger root?"
        q_ru = f"Решите уравнение {eq}: чему равен его больший корень?"
    steps_en = [
        f"Discriminant: D = ({p})^2 - 4*1*({q}) = {disc}.",
        f"sqrt(D) = {sd}, so x = ({-p} ± {sd}) / 2.",
        f"x1 = {r1}, x2 = {r2}; the larger root is {r2}.",
    ]
    steps_ru = [
        f"Дискриминант: D = ({p})^2 - 4*1*({q}) = {disc}.",
        f"sqrt(D) = {sd}, поэтому x = ({-p} ± {sd}) / 2.",
        f"x1 = {r1}, x2 = {r2}; больший корень равен {r2}.",
    ]
    value = float(r2)
    pool = _std_pool(value, [
        (float(r1), "smaller_root"),
        (float(-p), "returned_sum_of_roots"),
        (float(q), "returned_product_of_roots"),
        (float(-r2), "sign_error"),
    ])
    d_ = PairDraft(SUBJECT, "", "", Difficulty.UNIVERSITY, atype, "", question_en=q_en, question_ru=q_ru)
    if atype == AnswerType.NUMERIC:
        _set_numeric(d_, value, "")
        d_.solution_en = sol_en(steps_en, d_.canonical, "")
        d_.solution_ru = sol_ru(steps_ru, d_.canonical, "")
    else:
        _mc_numeric(d_, rng, value, pool, "")
        d_.solution_en = sol_en(steps_en, fmt(value), "")
        d_.solution_ru = sol_ru(steps_ru, fmt(value), "")
    d_.params = {"p": p, "q": q, "r1": r1, "r2": r2, "expected": r2}
    return _finish(d_, TOPICS["quad_eq"], "quad_eq", Difficulty.UNIVERSITY)


# --------------------------------------------------------------------------- #
# 4. Percentages
# --------------------------------------------------------------------------- #
def g_percent(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    variant = int(rng.integers(0, 4))
    base = 50 * int(rng.integers(2, 19))  # 100..900
    pct = int(rng.choice([5, 10, 15, 20, 25, 30, 40, 50, 60, 75]))
    if variant == 0:
        new = base * (100 + pct) // 100 if (base * pct) % 100 == 0 else base * (100 + pct) / 100
        q_en = f"The price of a jacket was {base} rubles and then increased by {pct}%. What is the new price in rubles?"
        q_ru = f"Цена куртки составляла {base} рублей, а затем выросла на {pct}%. Какой стала новая цена в рублях?"
        steps_en = [
            f"Increase factor: 100% + {pct}% = {100 + pct}%.",
            f"New price: {base} * {100 + pct} / 100 = {new} rubles.",
        ]
        steps_ru = [
            f"Коэффициент повышения: 100% + {pct}% = {100 + pct}%.",
            f"Новая цена: {base} * {100 + pct} / 100 = {new} рублей.",
        ]
        value, units = float(new), "RUB"
        params = {"variant": "increase", "base": base, "pct": pct, "expected": new}
        extras = [(float(base + pct), "added_percent_as_units"), (float(base * 2), "factor_of_2")]
    elif variant == 1:
        new = base * (100 - pct) // 100 if (base * pct) % 100 == 0 else base * (100 - pct) / 100
        q_en = f"A backpack costs {base} rubles. In a sale its price is reduced by {pct}%. How many rubles does it cost now?"
        q_ru = f"Рюкзак стоит {base} рублей. Во время распродажи его цена снижается на {pct}%. Сколько рублей он стоит теперь?"
        steps_en = [
            f"The discount amounts to {base} * {pct} / 100 = {base * pct / 100} rubles.",
            f"Sale price: {base} - {base * pct / 100} = {new} rubles.",
        ]
        steps_ru = [
            f"Скидка составляет {base} * {pct} / 100 = {base * pct / 100} рублей.",
            f"Цена со скидкой: {base} - {base * pct / 100} = {new} рублей.",
        ]
        value, units = float(new), "RUB"
        params = {"variant": "discount", "base": base, "pct": pct, "expected": new}
        extras = [(float(base + pct), "added_percent_as_units"), (float(base * pct / 100), "returned_discount")]
    elif variant == 2:
        whole = 20 * int(rng.integers(3, 22))  # 60..420
        part = whole * pct // 100 if (whole * pct) % 100 == 0 else whole * pct / 100
        q_en = f"What percent of the number {whole} is the number {part}?"
        q_ru = f"Какой процент от числа {whole} составляет число {part}?"
        steps_en = [
            f"Ratio: {part} / {whole} = {part / whole:.4f}.",
            f"In percent: {part / whole:.4f} * 100 = {pct}%.",
        ]
        steps_ru = [
            f"Отношение: {part} / {whole} = {part / whole:.4f}.",
            f"В процентах: {part / whole:.4f} * 100 = {pct}%.",
        ]
        value, units = float(pct), "%"
        params = {"variant": "what_percent", "whole": whole, "part": part, "expected": pct}
        extras = [(float(100 - pct), "complement"), (float(pct * 2), "factor_of_2")]
    else:
        final = base * (100 + pct) // 100 if (base * pct) % 100 == 0 else base * (100 + pct) / 100
        q_en = (
            f"After a {pct}% increase the price of a tablet became {final} rubles. "
            f"What was the original price in rubles?"
        )
        q_ru = (
            f"После повышения цены на {pct}% планшет стал стоить {final} рублей. "
            f"Какой была первоначальная цена в рублях?"
        )
        steps_en = [
            f"The new price is {100 + pct}% of the original: original * {100 + pct} / 100 = {final}.",
            f"Original price: {final} * 100 / {100 + pct} = {base} rubles.",
        ]
        steps_ru = [
            f"Новая цена составляет {100 + pct}% от первоначальной: первоначальная * {100 + pct} / 100 = {final}.",
            f"Первоначальная цена: {final} * 100 / {100 + pct} = {base} рублей.",
        ]
        value, units = float(base), "RUB"
        params = {"variant": "reverse", "final": final, "pct": pct, "expected": base}
        extras = [(float(final - final * pct / 100), "subtracted_percent"), (float(final), "returned_final")]
    d_ = PairDraft(SUBJECT, "", "", Difficulty.SCHOOL, atype, "", question_en=q_en, question_ru=q_ru)
    if atype == AnswerType.NUMERIC:
        _set_numeric(d_, value, units)
        d_.solution_en = sol_en(steps_en, d_.canonical, units)
        d_.solution_ru = sol_ru(steps_ru, d_.canonical, units)
    else:
        _mc_numeric(d_, rng, value, _std_pool(value, extras), units)
        d_.solution_en = sol_en(steps_en, fmt(value), units)
        d_.solution_ru = sol_ru(steps_ru, fmt(value), units)
    d_.params = params
    return _finish(d_, TOPICS["percent"], "percent", Difficulty.SCHOOL)


# --------------------------------------------------------------------------- #
# 5. Sequences
# --------------------------------------------------------------------------- #
def g_sequences(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    variant = int(rng.integers(0, 5))
    if variant in (0, 1, 2):  # arithmetic
        a1 = int(rng.integers(-10, 16))
        dd = int(rng.choice([2, 3, 4, 5, 6, 7, 8, 9, -2, -3, -4, -5]))
        n = int(rng.integers(5, 31))
        an = a1 + (n - 1) * dd
        ssum = n * (2 * a1 + (n - 1) * dd) / 2
        if ssum != int(ssum):
            n += 1
            an = a1 + (n - 1) * dd
            ssum = n * (2 * a1 + (n - 1) * dd) / 2
        if variant == 0:
            q_en = f"In an arithmetic progression the first term is {a1} and the common difference is {dd}. What is the {n}th term?"
            q_ru = f"В арифметической прогрессии первый член равен {a1}, разность равна {dd}. Чему равен {n}-й член прогрессии?"
            steps_en = [f"a_n = a_1 + (n - 1)*d = {a1} + {n - 1}*({dd}) = {an}."]
            steps_ru = [f"a_n = a_1 + (n - 1)*d = {a1} + {n - 1}*({dd}) = {an}."]
            value, units = float(an), ""
            params = {"variant": "arith_nth", "a1": a1, "d": dd, "n": n, "expected": an}
            extras = [(float(an - dd), "off_by_one_index"), (float(an + dd), "off_by_one_index")]
        elif variant == 1:
            q_en = (
                f"What is the sum of the first {n} terms of an arithmetic progression "
                f"with first term {a1} and common difference {dd}?"
            )
            q_ru = (
                f"Чему равна сумма первых {n} членов арифметической прогрессии "
                f"с первым членом {a1} и разностью {dd}?"
            )
            last = a1 + (n - 1) * dd
            steps_en = [
                f"The {n}th term: a_n = {a1} + {n - 1}*({dd}) = {last}.",
                f"S_n = n*(a_1 + a_n)/2 = {n}*({a1} + {last})/2 = {int(ssum)}.",
            ]
            steps_ru = [
                f"{n}-й член: a_n = {a1} + {n - 1}*({dd}) = {last}.",
                f"S_n = n*(a_1 + a_n)/2 = {n}*({a1} + {last})/2 = {int(ssum)}.",
            ]
            value, units = float(ssum), ""
            params = {"variant": "arith_sum", "a1": a1, "d": dd, "n": n, "expected": ssum}
            extras = [(float(n * (a1 + an)), "forgot_half"), (float(ssum - an), "dropped_last_term")]
        else:
            m = int(rng.integers(2, 6))
            k = m + int(rng.integers(2, 6))
            am = a1 + (m - 1) * dd
            ak = a1 + (k - 1) * dd
            q_en = f"In an arithmetic progression a_{m} = {am} and a_{k} = {ak}. What is the common difference?"
            q_ru = f"В арифметической прогрессии a_{m} = {am}, a_{k} = {ak}. Чему равна разность прогрессии?"
            steps_en = [
                f"a_{k} - a_{m} = ({k} - {m})*d, so {ak} - {am} = {k - m}*d.",
                f"d = ({ak} - {am}) / {k - m} = {dd}.",
            ]
            steps_ru = [
                f"a_{k} - a_{m} = ({k} - {m})*d, поэтому {ak} - {am} = {k - m}*d.",
                f"d = ({ak} - {am}) / {k - m} = {dd}.",
            ]
            value, units = float(dd), ""
            params = {"variant": "arith_diff", "m": m, "k": k, "am": am, "ak": ak, "expected": dd}
            extras = [(float(ak - am), "forgot_division"), (float(-dd), "sign_error")]
    else:  # geometric
        b1 = int(rng.integers(2, 10))
        qq = int(rng.choice([2, 3]))
        n = int(rng.integers(3, 8))
        bn = b1 * qq ** (n - 1)
        gsum = b1 * (qq**n - 1) / (qq - 1)
        if variant == 3:
            q_en = f"In a geometric progression the first term is {b1} and the common ratio is {qq}. What is the {n}th term?"
            q_ru = f"В геометрической прогрессии первый член равен {b1}, знаменатель равен {qq}. Чему равен {n}-й член прогрессии?"
            steps_en = [f"b_n = b_1 * q^(n - 1) = {b1} * {qq}^{n - 1} = {bn}."]
            steps_ru = [f"b_n = b_1 * q^(n - 1) = {b1} * {qq}^{n - 1} = {bn}."]
            value, units = float(bn), ""
            params = {"variant": "geo_nth", "b1": b1, "q": qq, "n": n, "expected": bn}
            extras = [(float(b1 * qq**n), "off_by_one_index"), (float(b1 * qq ** (n - 2)), "off_by_one_index")]
        else:
            q_en = (
                f"What is the sum of the first {n} terms of a geometric progression "
                f"with first term {b1} and common ratio {qq}?"
            )
            q_ru = (
                f"Чему равна сумма первых {n} членов геометрической прогрессии "
                f"с первым членом {b1} и знаменателем {qq}?"
            )
            steps_en = [
                f"The {n}th term: b_n = {b1} * {qq}^{n - 1} = {bn}.",
                f"S_n = b_1*(q^n - 1)/(q - 1) = {b1}*({qq}^{n} - 1)/{qq - 1} = {int(gsum)}.",
            ]
            steps_ru = [
                f"{n}-й член: b_n = {b1} * {qq}^{n - 1} = {bn}.",
                f"S_n = b_1*(q^n - 1)/(q - 1) = {b1}*({qq}^{n} - 1)/{qq - 1} = {int(gsum)}.",
            ]
            value, units = float(gsum), ""
            params = {"variant": "geo_sum", "b1": b1, "q": qq, "n": n, "expected": gsum}
            extras = [(float(b1 * (qq**n - 1)), "forgot_division"), (float(bn), "returned_nth_term")]
    d_ = PairDraft(SUBJECT, "", "", Difficulty.UNIVERSITY, atype, "", question_en=q_en, question_ru=q_ru)
    if atype == AnswerType.NUMERIC:
        _set_numeric(d_, value, units)
        d_.solution_en = sol_en(steps_en, d_.canonical, units)
        d_.solution_ru = sol_ru(steps_ru, d_.canonical, units)
    else:
        _mc_numeric(d_, rng, value, _std_pool(value, extras), units)
        d_.solution_en = sol_en(steps_en, fmt(value), units)
        d_.solution_ru = sol_ru(steps_ru, fmt(value), units)
    d_.params = params
    return _finish(d_, TOPICS["sequences"], "sequences", Difficulty.UNIVERSITY)


# --------------------------------------------------------------------------- #
# 6. Derivatives
# --------------------------------------------------------------------------- #
def g_derivatives(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    a = int(rng.choice([n for n in range(-5, 6) if n != 0]))
    b = int(rng.integers(-9, 10))
    c = int(rng.integers(-9, 10))
    x0 = int(rng.integers(1, 5))
    fp = 3 * a * x0**2 + 2 * b * x0 + c
    f_str = poly_str([(a, "x^3"), (b, "x^2"), (c, "x")])
    df_str = poly_str([(3 * a, "x^2"), (2 * b, "x"), (c, "")])
    q_en = f"Find the value of the derivative of f(x) = {f_str} at the point x0 = {x0}."
    q_ru = f"Найдите значение производной функции f(x) = {f_str} в точке x0 = {x0}."
    fx0 = a * x0**3 + b * x0**2 + c * x0
    steps_en = [
        f"Power rule: f'(x) = {df_str}.",
        f"Substitute x0 = {x0}: f'({x0}) = {fp}.",
    ]
    steps_ru = [
        f"По правилу степени: f'(x) = {df_str}.",
        f"Подставим x0 = {x0}: f'({x0}) = {fp}.",
    ]
    value, units = float(fp), ""
    pool = _std_pool(value, [
        (float(fx0), "did_not_differentiate"),
        (float(a * x0**2 + b * x0 + c), "power_not_reduced"),
        (float(3 * a * x0 + 2 * b), "degree_slip"),
    ])
    d_ = PairDraft(SUBJECT, "", "", Difficulty.UNIVERSITY, atype, "", question_en=q_en, question_ru=q_ru)
    if atype == AnswerType.NUMERIC:
        _set_numeric(d_, value, units)
        d_.solution_en = sol_en(steps_en, d_.canonical, units)
        d_.solution_ru = sol_ru(steps_ru, d_.canonical, units)
    else:
        _mc_numeric(d_, rng, value, pool, units)
        d_.solution_en = sol_en(steps_en, fmt(value), units)
        d_.solution_ru = sol_ru(steps_ru, fmt(value), units)
    d_.params = {"a": a, "b": b, "c": c, "x0": x0, "expected": fp}
    return _finish(d_, TOPICS["derivatives"], "derivatives", Difficulty.UNIVERSITY)


# --------------------------------------------------------------------------- #
# 7. Logarithms and exponentials
# --------------------------------------------------------------------------- #
def g_log_exp(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    variant = int(rng.integers(0, 4))
    base = int(rng.choice([2, 3, 5, 10]))
    k = int(rng.integers(2, 7))
    x = base**k
    if variant == 0:
        q_en = f"What is the value of log_{base}({x})?"
        q_ru = f"Чему равно значение log_{base}({x})?"
        steps_en = [f"{base}^{k} = {x}, therefore log_{base}({x}) = {k}."]
        steps_ru = [f"{base}^{k} = {x}, следовательно log_{base}({x}) = {k}."]
    elif variant == 1:
        q_en = f"Solve the equation {base}^x = {x}. What is x?"
        q_ru = f"Решите уравнение {base}^x = {x}. Чему равно x?"
        steps_en = [f"{x} = {base}^{k}, so the equation holds for x = {k}."]
        steps_ru = [f"{x} = {base}^{k}, поэтому уравнение выполняется при x = {k}."]
    elif variant == 2:
        e_k = k + 2
        q_en = f"What is the value of ln(e^{e_k})?"
        q_ru = f"Чему равно значение ln(e^{e_k})?"
        base, k, x = 0, e_k, 0  # e-base item; verifier uses exp()
        steps_en = [f"The natural logarithm inverts exp: ln(e^{e_k}) = {e_k}."]
        steps_ru = [f"Натуральный логарифм обратен экспоненте: ln(e^{e_k}) = {e_k}."]
    else:
        x = 10**k
        base = 10
        q_en = f"What is the value of log_10({x})?"
        q_ru = f"Чему равно значение log_10({x})?"
        steps_en = [f"{x} = 10^{k}, therefore log_10({x}) = {k}."]
        steps_ru = [f"{x} = 10^{k}, следовательно log_10({x}) = {k}."]
    value, units = float(k), ""
    d_ = PairDraft(SUBJECT, "", "", Difficulty.UNIVERSITY, atype, "", question_en=q_en, question_ru=q_ru)
    if atype == AnswerType.EXACT:
        d_.canonical = str(k)
        d_.solution_en = sol_en(steps_en, d_.canonical, "")
        d_.solution_ru = sol_ru(steps_ru, d_.canonical, "")
    else:
        if variant == 2:  # ln(e^k): base/argument distractors are meaningless
            pool = _std_pool(value)
        else:
            pool = _std_pool(value, [
                (float(x), "returned_argument"),
                (float(base), "returned_base"),
            ])
        _mc_numeric(d_, rng, value, pool, "")
        d_.solution_en = sol_en(steps_en, fmt(value), "")
        d_.solution_ru = sol_ru(steps_ru, fmt(value), "")
    d_.params = {"variant": variant, "base": base, "x": x, "expected": k}
    return _finish(d_, TOPICS["log_exp"], "log_exp", Difficulty.UNIVERSITY)


# --------------------------------------------------------------------------- #
# 8. Number theory (olympiad)
# --------------------------------------------------------------------------- #
PRIMES_SMALL = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
CRT_PAIRS = [(3, 4), (3, 5), (3, 7), (3, 8), (4, 5), (4, 7), (5, 6), (5, 7), (5, 8), (6, 7)]


def _crt(m1: int, m2: int, r1: int, r2: int) -> int:
    for n in range(1, m1 * m2 + 1):
        if n % m1 == r1 and n % m2 == r2:
            return n
    raise ValueError("no CRT solution")


def g_numtheory(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    if atype == AnswerType.MC:
        big_n = int(rng.integers(60, 301))
        ddiv = int(rng.integers(3, 10))
        cnt = big_n // ddiv
        q_en = f"How many integers from 1 to {big_n} are divisible by {ddiv}?"
        q_ru = f"Сколько натуральных чисел от 1 до {big_n} делятся на {ddiv} без остатка?"
        steps_en = [
            f"These are {ddiv}, {2 * ddiv}, ..., {cnt * ddiv} <= {big_n}.",
            f"The count is {big_n} / {ddiv} rounded down = {cnt}.",
        ]
        steps_ru = [
            f"Это числа {ddiv}, {2 * ddiv}, ..., {cnt * ddiv} <= {big_n}.",
            f"Их количество равно целой части {big_n} / {ddiv} = {cnt}.",
        ]
        value = float(cnt)
        pool = _std_pool(value, [(float(big_n // 10), "unit_slip_x10")])
        params = {"variant": "count_multiples", "N": big_n, "d": ddiv, "expected": cnt}
    else:
        variant = int(rng.integers(0, 4))
        if variant == 0:
            a = int(rng.integers(100, 1000))
            m = int(rng.integers(3, 13))
            r = a % m
            q_en = f"What is the remainder when {a} is divided by {m}?"
            q_ru = f"Каков остаток от деления числа {a} на {m}?"
            steps_en = [f"{a} = {a // m} * {m} + {a % m}.", f"The remainder is {r}."]
            steps_ru = [f"{a} = {a // m} * {m} + {a % m}.", f"Остаток равен {r}."]
            ans, steps_en_v, steps_ru_v = str(r), steps_en, steps_ru
            params = {"variant": "remainder", "a": a, "m": m, "expected": r}
        elif variant == 1:
            ps = sorted(int(v) for v in rng.choice(len(PRIMES_SMALL), size=2, replace=False))
            p1, p2 = PRIMES_SMALL[ps[0]], PRIMES_SMALL[ps[1]]
            n_all = p1 * p2
            ans_v = p2
            q_en = f"What is the largest prime that divides {n_all} exactly?"
            q_ru = f"Каково наибольшее простое число, на которое число {n_all} делится нацело?"
            steps_en = [f"Factorization: {n_all} = {p1} * {p2}.", f"The largest prime factor is {p2}."]
            steps_ru = [f"Разложение: {n_all} = {p1} * {p2}.", f"Наибольший простой делитель равен {p2}."]
            ans, steps_en_v, steps_ru_v = str(ans_v), steps_en, steps_ru
            params = {"variant": "largest_prime", "n": n_all, "expected": p2}
        elif variant == 2:
            m1, m2 = CRT_PAIRS[int(rng.integers(0, len(CRT_PAIRS)))]
            r1 = int(rng.integers(1, m1))
            r2 = int(rng.integers(1, m2))
            n_all = _crt(m1, m2, r1, r2)
            q_en = (
                f"Find the smallest positive integer that leaves remainder {r1} when divided by {m1} "
                f"and remainder {r2} when divided by {m2}. What is it?"
            )
            q_ru = (
                f"Найдите наименьшее натуральное число, которое при делении на {m1} даёт остаток {r1}, "
                f"а при делении на {m2} — остаток {r2}. Каково это число?"
            )
            steps_en = [
                f"Numbers of the form {m1}*t + {r1}: {r1}, {m1 + r1}, {2 * m1 + r1}, ...",
                f"The first one congruent to {r2} modulo {m2} is {n_all}.",
            ]
            steps_ru = [
                f"Числа вида {m1}*t + {r1}: {r1}, {m1 + r1}, {2 * m1 + r1}, ...",
                f"Первое число, дающее остаток {r2} при делении на {m2}, — это {n_all}.",
            ]
            ans, steps_en_v, steps_ru_v = str(n_all), steps_en, steps_ru
            params = {"variant": "crt", "m1": m1, "m2": m2, "r1": r1, "r2": r2, "expected": n_all}
        else:
            a = int(rng.integers(2, 10))
            k = int(rng.integers(3, 16))
            dig = pow(a, k, 10)
            q_en = f"What is the last digit of the number {a}^{k}?"
            q_ru = f"Какова последняя цифра числа {a}^{k}?"
            steps_en = [
                f"Last digits of powers of {a} repeat with period {1 if a in (0, 1, 5, 6) else 2 if a in (4, 9) else 4}.",
                f"{a}^{k} ends in {dig}.",
            ]
            steps_ru = [
                f"Последние цифры степеней числа {a} повторяются с периодом {1 if a in (0, 1, 5, 6) else 2 if a in (4, 9) else 4}.",
                f"Число {a}^{k} оканчивается цифрой {dig}.",
            ]
            ans, steps_en_v, steps_ru_v = str(dig), steps_en, steps_ru
            params = {"variant": "last_digit", "a": a, "k": k, "expected": dig}
        d_ = PairDraft(SUBJECT, "", "", Difficulty.OLYMPIAD, atype, "", question_en=q_en, question_ru=q_ru)
        d_.canonical = ans
        d_.solution_en = sol_en(steps_en_v, ans, "")
        d_.solution_ru = sol_ru(steps_ru_v, ans, "")
        d_.params = params
        return _finish(d_, TOPICS["numtheory"], "numtheory", Difficulty.OLYMPIAD)
    d_ = PairDraft(SUBJECT, "", "", Difficulty.OLYMPIAD, atype, "", question_en=q_en, question_ru=q_ru)
    _mc_numeric(d_, rng, value, pool, "")
    d_.solution_en = sol_en(steps_en, fmt(value), "")
    d_.solution_ru = sol_ru(steps_ru, fmt(value), "")
    d_.params = params
    return _finish(d_, TOPICS["numtheory"], "numtheory", Difficulty.OLYMPIAD)


# --------------------------------------------------------------------------- #
# 9. Planar geometry
# --------------------------------------------------------------------------- #
def g_geometry_area(rng: np.random.Generator, idx: int, atype: AnswerType, difficulty: Difficulty = Difficulty.SCHOOL) -> PairDraft:
    uni = difficulty == Difficulty.UNIVERSITY
    if uni:
        variant = int(rng.integers(0, 2))
    else:
        variant = int(rng.integers(0, 4))
    if uni and variant == 0:
        a = int(rng.integers(6, 21))
        b = int(rng.choice([n for n in range(6, 21) if (n + a) % 2 == 0]))
        h = int(rng.integers(4, 13))
        area = (a + b) // 2 * h
        q_en = f"The parallel sides of a trapezoid are {a} m and {b} m and its height is {h} m. What is its area?"
        q_ru = f"Основания трапеции равны {a} m и {b} m, высота — {h} m. Чему равна площадь трапеции?"
        steps_en = [f"S = (a + b)/2 * h = ({a} + {b})/2 * {h} = {area} m^2."]
        steps_ru = [f"S = (a + b)/2 * h = ({a} + {b})/2 * {h} = {area} m^2."]
        value, units = float(area), "m^2"
        pool = _std_pool(value, [(float((a + b) * h), "forgot_half"), (float(a * b), "multiplied_bases")])
        params = {"variant": "trapezoid", "a": a, "b": b, "h": h, "expected": area}
    elif uni:
        x1 = int(rng.integers(4, 16))
        x2 = int(rng.integers(1, x1))
        y2 = 2 * int(rng.integers(2, 8))
        area = x1 * y2 // 2
        q_en = f"The vertices of a triangle are A(0, 0), B({x1}, 0) and C({x2}, {y2}). What is the area of the triangle?"
        q_ru = f"Вершины треугольника имеют координаты A(0, 0), B({x1}, 0), C({x2}, {y2}). Чему равна площадь треугольника?"
        steps_en = [
            f"Base AB = {x1}; the height is the ordinate of C: {y2}.",
            f"S = {x1} * {y2} / 2 = {area}.",
        ]
        steps_ru = [
            f"Основание AB = {x1}; высота равна ординате точки C: {y2}.",
            f"S = {x1} * {y2} / 2 = {area}.",
        ]
        value, units = float(area), ""
        pool = _std_pool(value, [(float(x1 * y2), "forgot_half"), (float((x1 + x2) * y2 / 2), "wrong_base")])
        params = {"variant": "coord_triangle", "x1": x1, "x2": x2, "y2": y2, "expected": area}
    elif variant == 0:
        a = int(rng.integers(4, 25))
        b = int(rng.integers(3, 25))
        ask_area = bool(rng.integers(0, 2))
        if ask_area:
            value, units = float(a * b), "m^2"
            q_en = f"The sides of a rectangle are {a} m and {b} m. What is the area of the rectangle?"
            q_ru = f"Стороны прямоугольника равны {a} m и {b} m. Чему равна площадь прямоугольника?"
            steps_en = [f"S = a * b = {a} * {b} = {a * b} m^2."]
            steps_ru = [f"S = a * b = {a} * {b} = {a * b} m^2."]
            pool = _std_pool(value, [(float(2 * (a + b)), "perimeter_confusion"), (float(a + b), "added_sides")])
        else:
            value, units = float(2 * (a + b)), "m"
            q_en = f"The sides of a rectangle are {a} m and {b} m. What is the perimeter of the rectangle?"
            q_ru = f"Стороны прямоугольника равны {a} m и {b} m. Чему равен периметр прямоугольника?"
            steps_en = [f"P = 2*(a + b) = 2*({a} + {b}) = {2 * (a + b)} m."]
            steps_ru = [f"P = 2*(a + b) = 2*({a} + {b}) = {2 * (a + b)} m."]
            pool = _std_pool(value, [(float(a * b), "area_confusion"), (float(a + b), "forgot_factor_2")])
        params = {"variant": "rect_area" if ask_area else "rect_perim", "a": a, "b": b, "expected": value}
    elif variant == 1:
        s = int(rng.integers(4, 21))
        ask_area = bool(rng.integers(0, 2))
        if ask_area:
            value, units = float(s * s), "m^2"
            q_en = f"A square has side {s} m. What is its area?"
            q_ru = f"Сторона квадрата равна {s} m. Чему равна площадь квадрата?"
            steps_en = [f"S = s^2 = {s}^2 = {s * s} m^2."]
            steps_ru = [f"S = s^2 = {s}^2 = {s * s} m^2."]
            pool = _std_pool(value, [(float(4 * s), "perimeter_confusion")])
        else:
            value, units = float(4 * s), "m"
            q_en = f"A square has side {s} m. What is its perimeter?"
            q_ru = f"Сторона квадрата равна {s} m. Чему равен периметр квадрата?"
            steps_en = [f"P = 4*s = 4*{s} = {4 * s} m."]
            steps_ru = [f"P = 4*s = 4*{s} = {4 * s} m."]
            pool = _std_pool(value, [(float(s * s), "area_confusion"), (float(2 * s), "forgot_factor_2")])
        params = {"variant": "square", "s": s, "ask": "area" if ask_area else "perimeter", "expected": value}
    elif variant == 2:
        r = int(rng.choice([2, 4, 5, 10, 12, 15, 20, 25, 30, 50]))
        ask_area = bool(rng.integers(0, 2))
        if ask_area:
            value = 3.14 * r * r
            q_en = f"The radius of a circle is {r} m. What is the area of the circle (use pi = 3.14)?"
            q_ru = f"Радиус круга равен {r} m. Чему равна площадь круга, если считать pi = 3.14?"
            steps_en = [f"S = pi * r^2 = 3.14 * {r}^2 = {fmt(value)} m^2."]
            steps_ru = [f"S = pi * r^2 = 3.14 * {r}^2 = {fmt(value)} m^2."]
            pool = _std_pool(value, [(float(2 * 3.14 * r), "circumference_confusion"), (3.14 * 2 * r * r, "factor_of_2")])
            units = "m^2"
        else:
            value = 2 * 3.14 * r
            q_en = f"The radius of a circle is {r} m. What is the circumference of the circle (use pi = 3.14)?"
            q_ru = f"Радиус круга равен {r} m. Чему равна длина окружности, если считать pi = 3.14?"
            steps_en = [f"C = 2 * pi * r = 2 * 3.14 * {r} = {fmt(value)} m."]
            steps_ru = [f"C = 2 * pi * r = 2 * 3.14 * {r} = {fmt(value)} m."]
            pool = _std_pool(value, [(float(3.14 * r * r), "area_confusion"), (3.14 * r, "forgot_factor_2")])
            units = "m"
        params = {"variant": "circle", "r": r, "ask": "area" if ask_area else "circumference", "expected": value}
    else:
        b = 2 * int(rng.integers(3, 16))
        h = int(rng.integers(5, 21))
        value, units = float(b * h // 2), "m^2"
        q_en = f"The base of a triangle is {b} m and the height drawn to it is {h} m. What is the area of the triangle?"
        q_ru = f"Основание треугольника равно {b} m, а высота, проведённая к нему, — {h} m. Чему равна площадь треугольника?"
        steps_en = [f"S = b * h / 2 = {b} * {h} / 2 = {b * h // 2} m^2."]
        steps_ru = [f"S = b * h / 2 = {b} * {h} / 2 = {b * h // 2} m^2."]
        pool = _std_pool(value, [(float(b * h), "forgot_half"), (float((b + h) * 2), "perimeter_confusion")])
        params = {"variant": "triangle", "b": b, "h": h, "expected": b * h // 2}
    d_ = PairDraft(SUBJECT, "", "", difficulty, atype, "", question_en=q_en, question_ru=q_ru)
    if atype == AnswerType.NUMERIC:
        _set_numeric(d_, value, units)
        d_.solution_en = sol_en(steps_en, d_.canonical, units)
        d_.solution_ru = sol_ru(steps_ru, d_.canonical, units)
    else:
        _mc_numeric(d_, rng, value, pool, units)
        d_.solution_en = sol_en(steps_en, fmt(value), units)
        d_.solution_ru = sol_ru(steps_ru, fmt(value), units)
    d_.params = params
    return _finish(d_, TOPICS["geometry_area"], "geometry_area", difficulty)


def g_geometry_area_school(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    return g_geometry_area(rng, idx, atype, Difficulty.SCHOOL)


def g_geometry_area_uni(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    return g_geometry_area(rng, idx, atype, Difficulty.UNIVERSITY)


# --------------------------------------------------------------------------- #
# 10. Systems of two linear equations
# --------------------------------------------------------------------------- #
def g_sys_lin2(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    while True:
        x0 = int(rng.integers(-6, 10))
        y0 = int(rng.integers(-6, 10))
        a1 = int(rng.integers(-5, 6))
        b1 = int(rng.integers(-5, 6))
        if (a1, b1) != (0, 0):
            break
    a2, b2 = b1, -a1  # guarantees determinant -(a1^2 + b1^2) != 0
    c1 = a1 * x0 + b1 * y0
    c2 = a2 * x0 + b2 * y0
    sys_str = f"{poly_str([(a1, 'x'), (b1, 'y')])} = {c1}, {poly_str([(a2, 'x'), (b2, 'y')])} = {c2}"
    det = a1 * b2 - b1 * a2
    if atype == AnswerType.MC:
        q_en = f"Solve the system of equations {sys_str}. Which ordered pair (x, y) is the solution?"
        q_ru = f"Решите систему уравнений: {sys_str}. Какая пара (x, y) является решением системы?"
        cands = [
            (f"({x0}, {y0})", "correct"),
            (f"({y0}, {x0})", "swapped_unknowns"),
            (f"({-x0}, {y0})", "sign_error_x"),
            (f"({x0}, {-y0})", "sign_error_y"),
            (f"({-x0}, {-y0})", "sign_error_xy"),
            (f"({y0}, {-x0})", "swapped_and_sign"),
        ]
        seen: list[tuple[str, str]] = []
        for txt, tag in cands:
            if txt == f"({x0}, {y0})":
                seen.append((txt, tag))
        for txt, tag in cands[1:]:
            if all(txt != t for t, _ in seen) and len(seen) < 4:
                seen.append((txt, tag))
        correct = f"({x0}, {y0})"
        wrongs = [(t, tag) for t, tag in seen[1:]]
        wrongs3 = pick_distractors_str(rng, correct, wrongs)
        opts = [correct] + [w for w, _ in wrongs3]
        steps_en = [
            f"Eliminate y: multiply the equations suitably and add; x = {x0}.",
            f"Substitute back: {poly_str([(a1, 'x'), (b1, 'y')])} gives y = {y0}.",
        ]
        steps_ru = [
            f"Исключим y: умножим уравнения на подходящие числа и сложим; x = {x0}.",
            f"Подставим обратно: из {poly_str([(a1, 'x'), (b1, 'y')])} = {c1} получаем y = {y0}.",
        ]
        d_ = PairDraft(SUBJECT, "", "", Difficulty.UNIVERSITY, atype, "", question_en=q_en, question_ru=q_ru)
        d_.mc_en = tuple(opts)
        d_.mc_ru = tuple(opts)
        d_.distractor_tags = tuple(tag for _, tag in wrongs3)
        d_.params = {
            "a1": a1, "b1": b1, "c1": c1, "a2": a2, "b2": b2, "c2": c2,
            "expected_x": x0, "expected_y": y0, "expected_text": correct, "det": det,
        }
        d_.canonical = correct
        d_.solution_en = sol_en(steps_en, correct, "")
        d_.solution_ru = sol_ru(steps_ru, correct, "")
        return _finish(d_, TOPICS["sys_lin2"], "sys_lin2", Difficulty.UNIVERSITY)
    q_en = f"Solve the system of equations {sys_str}. What is the value of x?"
    q_ru = f"Решите систему уравнений: {sys_str}. Чему равно значение x?"
    steps_en = [
        f"By elimination/substitution the solution is x = {x0}, y = {y0}.",
        f"Substitution check: {a1}*({x0}) + {b1}*({y0}) = {c1}.",
    ]
    steps_ru = [
        f"Методом сложения или подстановки получаем x = {x0}, y = {y0}.",
        f"Проверка: {a1}*({x0}) + {b1}*({y0}) = {c1}.",
    ]
    value = float(x0)
    pool = _std_pool(value, [(float(y0), "returned_y_instead_of_x"), (float(c1), "returned_constant")])
    d_ = PairDraft(SUBJECT, "", "", Difficulty.UNIVERSITY, atype, "", question_en=q_en, question_ru=q_ru)
    _set_numeric(d_, value, "")
    d_.solution_en = sol_en(steps_en, d_.canonical, "")
    d_.solution_ru = sol_ru(steps_ru, d_.canonical, "")
    d_.params = {
        "a1": a1, "b1": b1, "c1": c1, "a2": a2, "b2": b2, "c2": c2,
        "expected_x": x0, "expected_y": y0, "expected": x0, "det": det,
    }
    return _finish(d_, TOPICS["sys_lin2"], "sys_lin2", Difficulty.UNIVERSITY)


def pick_distractors_str(
    rng: np.random.Generator, correct: str, wrongs: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    """Pick 3 distinct textual distractors (normalized text must differ)."""
    uniq: list[tuple[str, str]] = []
    for txt, tag in wrongs:
        if txt != correct and all(txt != t for t, _ in uniq):
            uniq.append((txt, tag))
    if len(uniq) < 3:
        raise ValueError(f"not enough distinct textual distractors for {correct!r}")
    idx = rng.choice(len(uniq), size=3, replace=False)
    return [uniq[int(i)] for i in idx]


# --------------------------------------------------------------------------- #
# 11. Inequalities
# --------------------------------------------------------------------------- #
def g_inequalities(rng: np.random.Generator, idx: int, atype: AnswerType, difficulty: Difficulty) -> PairDraft:
    if difficulty == Difficulty.UNIVERSITY:
        a = int(rng.integers(2, 7))
        a2 = a - 1
        b = int(rng.integers(-9, 10))
        c = int(rng.integers(-9, 10))
        t = c - b
        lhs = poly_str([(a, "x"), (b, "")])
        rhs = poly_str([(a2, "x"), (c, "")])
        ineq = f"{lhs} > {rhs}"
        correct = f"x > {t}"
        options = [
            (correct, "correct"),
            (f"x < {t}", "direction_flip"),
            (f"x >= {t}", "boundary_slip"),
            (f"x > {t + 1}", "off_by_one"),
            (f"x > {-t}", "sign_error"),
        ]
        steps_en = [
            f"Move the x-terms left and constants right: {a - a2}x > {c} - ({b}) = {t}.",
            f"x > {t}.",
        ]
        steps_ru = [
            f"Перенесём слагаемые с x влево, числа вправо: {a - a2}x > {c} - ({b}) = {t}.",
            f"x > {t}.",
        ]
        params = {"a": a, "a2": a2, "b": b, "c": c, "t": t, "kind": "linear"}
    else:
        while True:
            r1 = int(rng.integers(-7, 6))
            r2 = int(rng.integers(r1 + 2, 8))
            if r2 > r1 + 1:
                break
        p = -(r1 + r2)
        q = r1 * r2
        poly = f"{poly_str([(1, 'x^2'), (p, 'x'), (q, '')])}"
        correct = f"{r1} < x < {r2}"
        options = [
            (correct, "correct"),
            (f"x < {r1} or x > {r2}", "inverted_region"),
            (f"{r1} <= x <= {r2}", "boundary_slip"),
            (f"x < {r2}", "half_interval"),
        ]
        steps_en = [
            f"The roots of {poly} = 0 are {r1} and {r2}.",
            f"The parabola opens upward, so the product is negative strictly between the roots: {correct}.",
        ]
        steps_ru = [
            f"Корни уравнения {poly} = 0 равны {r1} и {r2}.",
            f"Парабола направлена ветвями вверх, поэтому выражение отрицательно строго между корнями: {correct}.",
        ]
        params = {"r1": r1, "r2": r2, "kind": "quadratic"}
        ineq = poly
    q_en = f"Solve the inequality {ineq} < 0. Which of the following describes all solutions?" if difficulty != Difficulty.UNIVERSITY else f"Solve the inequality {ineq}. Which of the following describes all solutions?"
    q_ru = f"Решите неравенство {ineq} < 0. Какое из указанных множеств является множеством решений?" if difficulty != Difficulty.UNIVERSITY else f"Решите неравенство {ineq}. Какое из указанных множеств является его решением?"
    wrongs = [(t_, tag) for t_, tag in options[1:]]
    wrongs3 = pick_distractors_str(rng, correct, wrongs)
    opts = [correct] + [w for w, _ in wrongs3]
    d_ = PairDraft(SUBJECT, "", "", difficulty, atype, "", question_en=q_en, question_ru=q_ru)
    d_.mc_en = tuple(opts)
    d_.mc_ru = tuple(opts)
    d_.distractor_tags = tuple(tag for _, tag in wrongs3)
    d_.canonical = correct
    d_.solution_en = sol_en(steps_en, correct, "")
    d_.solution_ru = sol_ru(steps_ru, correct, "")
    d_.params = {**params, "expected_text": correct}
    return _finish(d_, TOPICS["inequalities"], "inequalities", difficulty)


def g_ineq_uni(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    return g_inequalities(rng, idx, atype, Difficulty.UNIVERSITY)


def g_ineq_olym(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    return g_inequalities(rng, idx, atype, Difficulty.OLYMPIAD)


# --------------------------------------------------------------------------- #
# 12. Trigonometry basics
# --------------------------------------------------------------------------- #
TRIPLES = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (9, 40, 41), (20, 21, 29)]
SPECIALS = [
    ("sin", 30, 0.5), ("cos", 60, 0.5), ("sin", 60, 0.866), ("cos", 30, 0.866),
    ("sin", 45, 0.7071), ("cos", 45, 0.7071), ("tan", 45, 1.0),
    ("tan", 30, 0.5774), ("tan", 60, 1.7321), ("sin", 90, 1.0), ("cos", 0, 1.0),
]


def g_trig(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    variant = int(rng.integers(0, 3))
    if variant == 0:
        a, b, c = TRIPLES[int(rng.integers(0, len(TRIPLES)))]
        ask = int(rng.integers(0, 2))
        if ask == 0:
            correct = frac_str(a, c)
            q_en = f"In a right triangle the legs are {a} and {b} and the hypotenuse is {c}. What is the sine of the angle opposite the leg of length {a}?"
            q_ru = f"В прямоугольном треугольнике катеты равны {a} и {b}, гипотенуза — {c}. Чему равен синус угла, противолежащего катету {a}?"
            steps_en = [f"sin = opposite / hypotenuse = {a} / {c} = {correct}."]
            steps_ru = [f"sin = противолежащий катет / гипотенуза = {a} / {c} = {correct}."]
        else:
            correct = frac_str(b, c)
            q_en = f"In a right triangle the legs are {a} and {b} and the hypotenuse is {c}. What is the cosine of the angle adjacent to the leg of length {b}?"
            q_ru = f"В прямоугольном треугольнике катеты равны {a} и {b}, гипотенуза — {c}. Чему равен косинус угла, прилежащего к катету {b}?"
            steps_en = [f"cos = adjacent / hypotenuse = {b} / {c} = {correct}."]
            steps_ru = [f"cos = прилежащий катет / гипотенуза = {b} / {c} = {correct}."]
        pool = [
            (frac_str(b, c) if ask == 0 else frac_str(a, c), "confused_sin_cos"),
            (frac_str(a, b), "tangent_confusion"),
            (frac_str(c, a if ask == 0 else b), "reciprocal"),
        ]
        wrongs3 = pick_distractors_str(rng, correct, [(t_, tag) for t_, tag in pool])
        opts = [correct] + [w for w, _ in wrongs3]
        d_ = PairDraft(SUBJECT, "", "", Difficulty.UNIVERSITY, atype, "", question_en=q_en, question_ru=q_ru)
        d_.mc_en = tuple(opts)
        d_.mc_ru = tuple(opts)
        d_.distractor_tags = tuple(tag for _, tag in wrongs3)
        d_.canonical = correct
        d_.solution_en = sol_en(steps_en, correct, "")
        d_.solution_ru = sol_ru(steps_ru, correct, "")
        d_.params = {"a": a, "b": b, "c": c, "ask": ask, "expected_text": correct, "kind": "ratio"}
        return _finish(d_, TOPICS["trig"], "trig", Difficulty.UNIVERSITY)
    if variant == 1:
        fn, deg, val = SPECIALS[int(rng.integers(0, len(SPECIALS)))]
        correct = fmt(val)
        q_en = f"What is the exact decimal value of {fn}({deg}°)?"
        q_ru = f"Чему равно значение {fn}({deg}°)?"
        steps_en = [f"From the special-angle table: {fn}({deg}°) = {correct}."]
        steps_ru = [f"Из таблицы значений: {fn}({deg}°) = {correct}."]
        pool = [
            ("0.5", "special_angle_confusion"),
            ("0.866", "special_angle_confusion"),
            ("0.7071", "special_angle_confusion"),
            ("1", "unit_confusion"),
            ("1.7321", "tangent_slip"),
            ("0.5774", "tangent_slip"),
        ]
        pool = [(t_, tag) for t_, tag in pool if t_ != correct]
        wrongs3 = pick_distractors_str(rng, correct, pool)
        opts = [correct] + [w for w, _ in wrongs3]
        d_ = PairDraft(SUBJECT, "", "", Difficulty.UNIVERSITY, atype, "", question_en=q_en, question_ru=q_ru)
        d_.mc_en = tuple(opts)
        d_.mc_ru = tuple(opts)
        d_.distractor_tags = tuple(tag for _, tag in wrongs3)
        d_.canonical = correct
        d_.solution_en = sol_en(steps_en, correct, "")
        d_.solution_ru = sol_ru(steps_ru, correct, "")
        d_.params = {"fn": fn, "deg": deg, "expected_text": correct, "kind": "special"}
        return _finish(d_, TOPICS["trig"], "trig", Difficulty.UNIVERSITY)
    fn, deg, val = [("sin", 30, 0.5), ("cos", 60, 0.5), ("tan", 45, 1.0), ("sin", 90, 1.0)][int(rng.integers(0, 4))]
    q_en = f"Given {fn}(x) = {fmt(val)} and 0° <= x <= 90°, what is x in degrees?"
    q_ru = f"Известно, что {fn}(x) = {fmt(val)} и 0° <= x <= 90°. Чему равен x в градусах?"
    steps_en = [f"The angle with {fn} equal to {fmt(val)} in this range is {deg}°."]
    steps_ru = [f"Угол, при котором {fn} равен {fmt(val)} в этом диапазоне, составляет {deg}°."]
    d_ = PairDraft(SUBJECT, "", "", Difficulty.UNIVERSITY, atype, "", question_en=q_en, question_ru=q_ru)
    _set_numeric(d_, float(deg), "")
    d_.solution_en = sol_en(steps_en, d_.canonical, "")
    d_.solution_ru = sol_ru(steps_ru, d_.canonical, "")
    d_.params = {"fn": fn, "value": val, "expected": deg, "kind": "inverse"}
    return _finish(d_, TOPICS["trig"], "trig", Difficulty.UNIVERSITY)


# --------------------------------------------------------------------------- #
# 13. Probability and combinatorics
# --------------------------------------------------------------------------- #
def _mc_text(
    draft: PairDraft,
    rng: np.random.Generator,
    correct: str,
    wrongs: list[tuple[str, str]],
    params_extra: dict[str, Any],
) -> None:
    wrongs3 = pick_distractors_str(rng, correct, wrongs)
    opts = [correct] + [w for w, _ in wrongs3]
    draft.mc_en = tuple(opts)
    draft.mc_ru = tuple(opts)
    draft.distractor_tags = tuple(tag for _, tag in wrongs3)
    draft.canonical = correct
    draft.params.update({"expected_text": correct, **params_extra})


def g_prob_comb(rng: np.random.Generator, idx: int, atype: AnswerType, difficulty: Difficulty) -> PairDraft:
    d_ = PairDraft(SUBJECT, "", "", difficulty, atype, "", question_en="", question_ru="")
    if difficulty == Difficulty.SCHOOL:
        if atype == AnswerType.MC:
            r = int(rng.integers(5, 10))
            g = int(rng.integers(5, 10))
            b = int(rng.integers(5, 10))
            t = r + g + b
            correct = frac_str(r, t)
            q_en = (
                f"An urn contains {r} red, {g} green and {b} blue balls. One ball is drawn at random. "
                f"What is the probability that it is red?"
            )
            q_ru = (
                f"В урне лежат {r} красных, {g} зелёных и {b} синих шаров. Из урны наугад достают один шар. "
                f"Какова вероятность того, что этот шар окажется красным?"
            )
            steps_en = [f"Total balls: {r} + {g} + {b} = {t}.", f"P(red) = {r} / {t} = {correct}."]
            steps_ru = [f"Всего шаров: {r} + {g} + {b} = {t}.", f"P(красный) = {r} / {t} = {correct}."]
            wrongs = [
                (frac_str(r + 1, t), "off_by_one"),
                (frac_str(t - r, t), "complement_error"),
                (frac_str(t, r), "reciprocal"),
                (frac_str(r, t - r), "wrong_total"),
            ]
            _mc_text(d_, rng, correct, wrongs, {"r": r, "g": g, "b": b, "kind": "urn"})
        else:
            kind = int(rng.integers(0, 2))
            if kind == 0:
                n = int(rng.integers(5, 11))
                k = int(rng.integers(2, min(4, n - 1)))
                val = math.comb(n, k)
                q_en = f"In how many ways can {k} students on duty be chosen from a class of {n} students?"
                q_ru = f"Сколькими способами можно выбрать {k} дежурных из {n} учеников класса?"
                steps_en = [
                    f"The order does not matter, so we count combinations: C({n}, {k}) = {n}! / ({k}! * {n - k}!).",
                    f"C({n}, {k}) = {val}.",
                ]
                steps_ru = [
                    f"Порядок не важен, поэтому считаем сочетания: C({n}, {k}) = {n}! / ({k}! * {n - k}!).",
                    f"C({n}, {k}) = {val}.",
                ]
                d_.params = {"kind": "comb", "n": n, "k": k, "expected": val}
            else:
                a = int(rng.integers(3, 7))
                bb = int(rng.integers(3, 7))
                val = a * bb
                q_en = f"A cafeteria offers {a} main dishes and {bb} drinks. In how many ways can one choose a dish and a drink?"
                q_ru = f"В столовой в меню {a} первых блюд и {bb} напитков. Сколькими способами можно выбрать одно блюдо и один напиток?"
                steps_en = [f"Product rule: {a} * {bb} = {val} ways."]
                steps_ru = [f"Правило произведения: {a} * {bb} = {val} способов."]
                d_.params = {"kind": "product_rule", "a": a, "b": bb, "expected": val}
            d_.canonical = str(val)
            d_.question_en = q_en
            d_.question_ru = q_ru
            d_.solution_en = sol_en(steps_en, d_.canonical, "")
            d_.solution_ru = sol_ru(steps_ru, d_.canonical, "")
    elif difficulty == Difficulty.UNIVERSITY:
        kind = int(rng.integers(0, 2))
        if kind == 0:
            s = int(rng.integers(4, 11))
            cnt = 6 - abs(s - 7)
            correct = frac_str(cnt, 36)
            q_en = f"Two fair dice are rolled. What is the probability that the sum of the points is exactly {s}?"
            q_ru = f"Бросают две игральные кости. Какова вероятность того, что сумма выпавших очков равна {s}?"
            steps_en = [
                f"Favorable outcomes: {cnt} (out of 6 * 6 = 36 equiprobable outcomes).",
                f"P = {cnt} / 36 = {correct}.",
            ]
            steps_ru = [
                f"Благоприятных исходов: {cnt} (из 6 * 6 = 36 равновозможных).",
                f"P = {cnt} / 36 = {correct}.",
            ]
            wrongs = [
                (frac_str(cnt + 1, 36), "off_by_one"),
                (frac_str(cnt, 12), "wrong_denominator"),
                (frac_str(s, 36), "used_sum_as_count"),
                (frac_str(cnt, 30), "wrong_denominator"),
            ]
            _mc_text(d_, rng, correct, wrongs, {"s": s, "kind": "dice"})
        else:
            t1 = int(rng.integers(3, 7))
            r1 = int(rng.integers(1, t1))
            t2 = int(rng.integers(3, 7))
            r2 = int(rng.integers(1, t2))
            correct = frac_str(r1 * r2, t1 * t2)
            q_en = (
                f"One ball is drawn from each of two urns: the first holds {r1} red balls out of {t1}, "
                f"the second {r2} red balls out of {t2}. What is the probability that both drawn balls are red?"
            )
            q_ru = (
                f"Из каждой из двух урн достают по одному шару: в первой урне {r1} красных шаров из {t1}, "
                f"во второй — {r2} красных из {t2}. Какова вероятность, что оба вынутых шара окажутся красными?"
            )
            steps_en = [
                f"Independence: P = ({r1}/{t1}) * ({r2}/{t2}) = {correct}.",
            ]
            steps_ru = [
                f"Независимость событий: P = ({r1}/{t1}) * ({r2}/{t2}) = {correct}.",
            ]
            sum_frac = frac_str(r1 * t2 + r2 * t1, t1 * t2)
            wrongs = [
                (sum_frac, "added_probabilities"),
                (frac_str(r1, t1), "ignored_second_event"),
                (frac_str(r1 * r2, t1 + t2), "wrong_denominator"),
                (frac_str(r2, t2), "ignored_first_event"),
            ]
            _mc_text(d_, rng, correct, wrongs, {"t1": t1, "r1": r1, "t2": t2, "r2": r2, "kind": "independent"})
    else:
        if atype == AnswerType.MC:
            kind = int(rng.integers(0, 2))
            if kind == 0:
                ng = int(rng.integers(4, 8))
                nb = int(rng.integers(4, 8))
                k = int(rng.integers(2, 4))
                j = int(rng.integers(2, 4))
                val = math.comb(ng, k) * math.comb(nb, j)
                q_en = f"A club has {ng} girls and {nb} boys. In how many ways can a team of {k} girls and {j} boys be formed?"
                q_ru = f"В кружке занимаются {ng} девочек и {nb} мальчиков. Сколькими способами можно составить команду из {k} девочек и {j} мальчиков?"
                steps_en = [
                    f"Choose the girls: C({ng}, {k}) = {math.comb(ng, k)}.",
                    f"Choose the boys: C({nb}, {j}) = {math.comb(nb, j)}; product = {val}.",
                ]
                steps_ru = [
                    f"Выберем девочек: C({ng}, {k}) = {math.comb(ng, k)}.",
                    f"Выберем мальчиков: C({nb}, {j}) = {math.comb(nb, j)}; произведение = {val}.",
                ]
                wrongs = [
                    (str(math.comb(ng + nb, k + j)), "combined_pool_error"),
                    (str(math.comb(ng, k) + math.comb(nb, j)), "added_instead_of_multiplied"),
                    (str(val + math.comb(ng, k)), "extra_term"),
                    (str(2 * val), "factor_of_2"),
                ]
                _mc_text(d_, rng, str(val), wrongs, {"ng": ng, "nb": nb, "k": k, "j": j, "kind": "team"})
            else:
                a = int(rng.integers(3, 7))
                bb = int(rng.integers(3, 7))
                val = math.comb(a + bb, a)
                q_en = (
                    f"How many distinct sequences can be made by arranging {a} identical red balls "
                    f"and {bb} identical blue balls in a row?"
                )
                q_ru = (
                    f"Сколькими способами можно выложить в ряд {a} одинаковых красных и {bb} одинаковых синих шаров?"
                )
                steps_en = [
                    f"Choose the positions of the red balls among {a + bb} places: C({a + bb}, {a}).",
                    f"C({a + bb}, {a}) = {val}.",
                ]
                steps_ru = [
                    f"Выберем позиции красных шаров среди {a + bb} мест: C({a + bb}, {a}).",
                    f"C({a + bb}, {a}) = {val}.",
                ]
                wrongs = [
                    (str(math.factorial(a + bb)), "treated_balls_as_distinct"),
                    (str(math.factorial(a) * math.factorial(bb)), "multiplied_factorials"),
                    (str(math.comb(a + bb - 1, a)), "off_by_one"),
                ]
                _mc_text(d_, rng, str(val), wrongs, {"a": a, "b": bb, "kind": "sequences"})
        else:
            n = int(rng.integers(8, 13))
            k = int(rng.integers(3, 5))
            val = math.comb(n, k)
            q_en = f"In how many ways can {k} winners be selected from {n} contest participants (order does not matter)?"
            q_ru = f"Сколькими способами можно выбрать {k} победителей из {n} участников конкурса (порядок не важен)?"
            steps_en = [f"C({n}, {k}) = {n}! / ({k}! * {n - k}!) = {val}."]
            steps_ru = [f"C({n}, {k}) = {n}! / ({k}! * {n - k}!) = {val}."]
            d_.canonical = str(val)
            d_.question_en = q_en
            d_.question_ru = q_ru
            d_.solution_en = sol_en(steps_en, d_.canonical, "")
            d_.solution_ru = sol_ru(steps_ru, d_.canonical, "")
            d_.params = {"kind": "comb", "n": n, "k": k, "expected": val}
    if not d_.question_en:
        d_.question_en = q_en
        d_.question_ru = q_ru
        d_.solution_en = sol_en(steps_en, d_.canonical, "")
        d_.solution_ru = sol_ru(steps_ru, d_.canonical, "")
    return _finish(d_, TOPICS["prob_comb"], "prob_comb", difficulty)


def g_prob_school(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    return g_prob_comb(rng, idx, atype, Difficulty.SCHOOL)


def g_prob_uni(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    return g_prob_comb(rng, idx, atype, Difficulty.UNIVERSITY)


def g_prob_olym(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    return g_prob_comb(rng, idx, atype, Difficulty.OLYMPIAD)


GENERATORS: dict[str, Any] = {
    "arith_word": g_arith_word,
    "linear_eq": g_linear_eq,
    "quad_eq": g_quad_eq,
    "percent": g_percent,
    "sequences": g_sequences,
    "derivatives": g_derivatives,
    "log_exp": g_log_exp,
    "numtheory": g_numtheory,
    "geometry_area": g_geometry_area_school,  # school rows
    "geometry_area_uni": g_geometry_area_uni,
    "sys_lin2": g_sys_lin2,
    "inequalities": g_ineq_uni,
    "inequalities_olym": g_ineq_olym,
    "trig": g_trig,
    "prob_comb": g_prob_school,
    "prob_comb_uni": g_prob_uni,
    "prob_comb_olym": g_prob_olym,
}

# topic_key used for verifier dispatch: unify school/uni aliases
KEY_ALIASES: dict[str, str] = {
    "geometry_area_uni": "geometry_area",
    "inequalities_olym": "inequalities",
    "prob_comb_uni": "prob_comb",
    "prob_comb_olym": "prob_comb",
}

__all__ = ["SUBJECT", "PREFIX", "TOPICS", "RUBRICS", "SPEC", "GENERATORS", "KEY_ALIASES", "Fraction", "product", "combinations"]
