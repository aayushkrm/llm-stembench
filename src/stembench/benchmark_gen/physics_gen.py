"""Original bilingual (ru/en) physics item generators.

All prompts state the numeric conventions they rely on (g = 9.8 or 10 m/s^2,
c_water = 4200 J/(kg*K), pi = 3.14) so that every item is self-contained.
Units are untranslated SI symbols in both languages; the decimal separator is
"." in both languages (see ``_core`` docstring).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from stembench.schemas import AnswerType, Difficulty, Subject

from ._core import (
    PairDraft,
    fmt,
    sol_en,
    sol_ru,
)
from .math_gen import _mc_numeric, _set_numeric, _std_pool

SUBJECT = Subject.PHYSICS
PREFIX = "PHYS"

TOPICS: dict[str, str] = {
    "kinem_const": "kinematics: constant velocity",
    "kinem_accel": "kinematics: constant acceleration",
    "newton2": "Newton's second law",
    "work_energy": "work-energy theorem",
    "momentum": "momentum and its conservation",
    "ohm_law": "Ohm's law and series resistors",
    "ohm_circuits": "series and parallel resistor networks",
    "power": "mechanical and electrical power",
    "heat_q": "thermal energy: Q = m*c*dT",
    "gas_law": "ideal gas law and isoprocesses",
    "hydrostatic": "hydrostatic pressure",
    "lenses": "thin lenses",
    "circular": "uniform circular motion",
    "projectile": "projectile motion",
}

RUBRICS: dict[tuple[str, Difficulty], tuple[str, str]] = {
    ("kinem_const", Difficulty.SCHOOL): (
        "Single application of v = s/t with round numbers.",
        "Однократное применение формулы v = s/t с круглыми числами.",
    ),
    ("kinem_accel", Difficulty.UNIVERSITY): (
        "Uniformly accelerated motion; requires v = v0 + a*t or s = v0*t + a*t^2/2.",
        "Равноускоренное движение; нужны формулы v = v0 + a*t или s = v0*t + a*t^2/2.",
    ),
    ("newton2", Difficulty.SCHOOL): (
        "Direct application of F = m*a in one of its three forms.",
        "Прямое применение формулы F = m*a в одной из трёх форм.",
    ),
    ("work_energy", Difficulty.UNIVERSITY): (
        "Work of a force at an angle to displacement or kinetic-energy change.",
        "Работа силы под углом к перемещению либо изменение кинетической энергии.",
    ),
    ("momentum", Difficulty.UNIVERSITY): (
        "Momentum of a body or perfectly inelastic collision conservation.",
        "Импульс тела или сохранение импульса при неупругом столкновении.",
    ),
    ("ohm_law", Difficulty.SCHOOL): (
        "Ohm's law for a single resistor or a series pair.",
        "Закон Ома для одного резистора или последовательной пары.",
    ),
    ("ohm_circuits", Difficulty.UNIVERSITY): (
        "Mixed series-parallel network; combine resistances first, then use Ohm's law.",
        "Смешанное соединение резисторов: сначала свести сопротивления, затем закон Ома.",
    ),
    ("power", Difficulty.SCHOOL): (
        "Power from work over time or from voltage and current.",
        "Мощность через работу и время либо через напряжение и силу тока.",
    ),
    ("heat_q", Difficulty.SCHOOL): (
        "One application of Q = m*c*dT with water.",
        "Однократное применение формулы Q = m*c*dT для воды.",
    ),
    ("heat_q", Difficulty.UNIVERSITY): (
        "Inverting Q = m*c*dT to recover the specific heat capacity.",
        "Обращение формулы Q = m*c*dT для нахождения удельной теплоёмкости.",
    ),
    ("gas_law", Difficulty.UNIVERSITY): (
        "Isothermal or isobaric ideal-gas process with the combined law.",
        "Изотермический или изобарный процесс идеального газа по газовым законам.",
    ),
    ("hydrostatic", Difficulty.UNIVERSITY): (
        "Hydrostatic pressure P = rho*g*h or the force on a wall area.",
        "Гидростатическое давление P = rho*g*h или сила давления на площадку.",
    ),
    ("lenses", Difficulty.UNIVERSITY): (
        "Thin-lens equation 1/f = 1/d_o + 1/d_i with real image formation.",
        "Формула тонкой линзы 1/f = 1/d_o + 1/d_i с действительным изображением.",
    ),
    ("lenses", Difficulty.OLYMPIAD): (
        "A two-lens system requires sequential imaging, separation geometry, and combined magnification.",
        "Система двух линз требует последовательного построения изображений, учёта расстояния и общего увеличения.",
    ),
    ("circular", Difficulty.OLYMPIAD): (
        "Couples centripetal dynamics with energy conservation or limiting friction on a banked curve.",
        "Сочетает центростремительную динамику с сохранением энергии или предельным трением на вираже.",
    ),
    ("projectile", Difficulty.OLYMPIAD): (
        "Combines two-dimensional motion with a nonzero launch height or a two-angle trajectory constraint.",
        "Сочетает двумерное движение с ненулевой высотой старта или условием двух углов траектории.",
    ),
}

SPEC = [
    # topic_key, difficulty, count, answer_type  (physics: 200 pairs)
    ("kinem_const", Difficulty.SCHOOL, 8, AnswerType.NUMERIC),
    ("kinem_const", Difficulty.SCHOOL, 6, AnswerType.MC),
    ("kinem_const", Difficulty.SCHOOL, 4, AnswerType.EXACT),
    ("newton2", Difficulty.SCHOOL, 8, AnswerType.NUMERIC),
    ("newton2", Difficulty.SCHOOL, 6, AnswerType.MC),
    ("newton2", Difficulty.SCHOOL, 2, AnswerType.EXACT),
    ("ohm_law", Difficulty.SCHOOL, 10, AnswerType.NUMERIC),
    ("ohm_law", Difficulty.SCHOOL, 6, AnswerType.MC),
    ("power", Difficulty.SCHOOL, 8, AnswerType.NUMERIC),
    ("power", Difficulty.SCHOOL, 4, AnswerType.MC),
    ("heat_q", Difficulty.SCHOOL, 8, AnswerType.NUMERIC),
    ("heat_q", Difficulty.SCHOOL, 6, AnswerType.MC),
    ("heat_q", Difficulty.SCHOOL, 4, AnswerType.EXACT),
    ("kinem_accel", Difficulty.UNIVERSITY, 8, AnswerType.NUMERIC),
    ("kinem_accel", Difficulty.UNIVERSITY, 6, AnswerType.MC),
    ("work_energy", Difficulty.UNIVERSITY, 8, AnswerType.NUMERIC),
    ("work_energy", Difficulty.UNIVERSITY, 4, AnswerType.MC),
    ("momentum", Difficulty.UNIVERSITY, 4, AnswerType.NUMERIC),
    ("momentum", Difficulty.UNIVERSITY, 4, AnswerType.MC),
    ("momentum", Difficulty.UNIVERSITY, 4, AnswerType.EXACT),
    ("ohm_circuits", Difficulty.UNIVERSITY, 6, AnswerType.NUMERIC),
    ("ohm_circuits", Difficulty.UNIVERSITY, 8, AnswerType.MC),
    ("gas_law", Difficulty.UNIVERSITY, 6, AnswerType.NUMERIC),
    ("gas_law", Difficulty.UNIVERSITY, 4, AnswerType.MC),
    ("hydrostatic", Difficulty.UNIVERSITY, 4, AnswerType.NUMERIC),
    ("hydrostatic", Difficulty.UNIVERSITY, 4, AnswerType.MC),
    ("hydrostatic", Difficulty.UNIVERSITY, 2, AnswerType.EXACT),
    ("lenses", Difficulty.UNIVERSITY, 2, AnswerType.NUMERIC),
    ("lenses", Difficulty.UNIVERSITY, 6, AnswerType.MC),
    ("lenses", Difficulty.UNIVERSITY, 2, AnswerType.EXACT),
    ("heat_q_uni", Difficulty.UNIVERSITY, 4, AnswerType.NUMERIC),
    ("heat_q_uni", Difficulty.UNIVERSITY, 4, AnswerType.MC),
    ("circular", Difficulty.OLYMPIAD, 2, AnswerType.NUMERIC),
    ("circular", Difficulty.OLYMPIAD, 8, AnswerType.MC),
    ("projectile", Difficulty.OLYMPIAD, 6, AnswerType.NUMERIC),
    ("projectile", Difficulty.OLYMPIAD, 4, AnswerType.MC),
    ("projectile", Difficulty.OLYMPIAD, 4, AnswerType.EXACT),
    ("lenses_olym", Difficulty.OLYMPIAD, 6, AnswerType.MC),
]

G98 = 9.8
PARALLEL_PAIRS = [(30, 60), (20, 30), (6, 3), (12, 4), (60, 20), (15, 10), (24, 12), (10, 40)]
# All Russian nouns are masculine so the question tails («он поднимется») agree.
PROJECTILE_OBJECTS = (("ball", "мяч"), ("stone", "камень"), ("marble", "шарик"), ("cube", "кубик"))


def _finish(d: PairDraft, key: str, difficulty: Difficulty) -> PairDraft:
    # Every benchmark prompt is phrased as a direct question.  A few of the
    # hydrostatics templates historically ended the interrogative sentence
    # with a full stop; normalize the punctuation at the subject boundary so
    # both language variants stay parallel.
    for attr in ("question_en", "question_ru"):
        question = getattr(d, attr).rstrip()
        if question.endswith("."):
            question = question[:-1] + "?"
        elif not question.endswith("?"):
            question += "?"
        setattr(d, attr, question)
    base = key.replace("_olym", "")
    d.topic = TOPICS[base if base in TOPICS else key]
    d.topic_key = "lenses" if key.startswith("lenses") else base
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
    pool: list[tuple[float, str]] | None,
    params: dict[str, Any],
) -> PairDraft:
    if len(steps_en) == 1:
        steps_en = [
            "Select the governing relation and isolate the requested physical quantity.",
            *steps_en,
        ]
    if len(steps_ru) == 1:
        steps_ru = [
            "Выберем определяющее соотношение и выразим из него искомую физическую величину.",
            *steps_ru,
        ]
    if atype == AnswerType.NUMERIC:
        _set_numeric(d, value, units)
        d.solution_en = sol_en(steps_en, d.canonical, units)
        d.solution_ru = sol_ru(steps_ru, d.canonical, units)
    else:
        _mc_numeric(d, rng, value, _std_pool(value, pool or []), units)
        d.solution_en = sol_en(steps_en, fmt(value), units)
        d.solution_ru = sol_ru(steps_ru, fmt(value), units)
    d.params = params
    return d


def _emit_exact(
    d: PairDraft,
    ans: str,
    steps_en: list[str],
    steps_ru: list[str],
    params: dict[str, Any],
    ans_ru: str | None = None,
) -> PairDraft:
    if len(steps_en) == 1:
        steps_en = [
            "Compare the governing relation before and after the stated change.",
            *steps_en,
        ]
    if len(steps_ru) == 1:
        steps_ru = [
            "Сравним определяющее соотношение до и после указанного изменения.",
            *steps_ru,
        ]
    d.canonical = ans
    d.solution_en = sol_en(steps_en, ans, "")
    d.solution_ru = sol_ru(steps_ru, ans_ru or ans, "")
    d.params = params
    return d


# --------------------------------------------------------------------------- #
# Kinematics: constant velocity (school)
# --------------------------------------------------------------------------- #
KINEM_AGENTS = [
    ("cyclist", "велосипедист", "он", "его"),
    ("walker", "пешеход", "он", "его"),
    ("car", "автомобиль", "он", "его"),
    ("train", "поезд", "он", "его"),
]


def g_kinem_const(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    agent_en, agent_ru, _, _ = KINEM_AGENTS[int(rng.integers(0, len(KINEM_AGENTS)))]
    if atype == AnswerType.EXACT:
        ans_ru: str | None = None
        if bool(rng.integers(0, 2)):
            v1 = int(rng.integers(1, 4))
            v2 = v1 + int(rng.integers(1, 4))
            d0 = 5 * int(rng.integers(2, 19))
            t_big = int(rng.choice([5, 10, 15, 20, 30, 40, 60]))
            closing = v2 - v1
            catch = d0 / closing <= t_big
            ans = "yes" if catch else "no"
            ans_ru = "да" if catch else "нет"
            q_en = (
                f"Boris is {d0} m behind Anna on a straight road. Anna walks at {v1} m/s and Boris "
                f"at {v2} m/s in the same direction. Will Boris catch up with Anna within {t_big} s?"
            )
            q_ru = (
                f"Борис идёт по прямой дороге в {d0} m позади Анны. Анна идёт со скоростью {v1} m/s, "
                f"Борис — со скоростью {v2} m/s в том же направлении. Догонит ли Борис Анну за {t_big} s?"
            )
            steps_en = [
                f"Closing speed: {v2} - {v1} = {closing} m/s.",
                f"Time needed: {d0} / {closing} = {d0 / closing:g} s, which is "
                f"{'at most' if catch else 'more than'} {t_big} s.",
            ]
            steps_ru = [
                f"Скорость сближения: {v2} - {v1} = {closing} m/s.",
                f"Необходимое время: {d0} / {closing} = {d0 / closing:g} s, это "
                f"{'не больше' if catch else 'больше'} {t_big} s.",
            ]
            exact_params = {"kind": "catch", "v1": v1, "v2": v2, "d": d0, "T": t_big,
                            "expected_text": ans}
        else:
            k = int(rng.choice([2, 3, 4]))
            v1 = int(rng.integers(1, 5))
            v2 = k * v1
            ans = str(k)
            if bool(rng.integers(0, 2)):
                q_en = (
                    f"A high-speed train travels at {v2} m/s while a freight train travels at {v1} m/s "
                    f"on a parallel track. How many times is the speed of the high-speed train greater?"
                )
                q_ru = (
                    f"Скорый поезд движется со скоростью {v2} m/s, а грузовой поезд по параллельному пути — "
                    f"со скоростью {v1} m/s. Во сколько раз скорость скорого поезда больше?"
                )
            else:
                q_en = (
                    f"A motorbike moves at {v2} m/s and a cyclist at {v1} m/s along the same road. "
                    f"How many times faster is the motorbike?"
                )
                q_ru = (
                    f"Мотоциклист движется со скоростью {v2} m/s, а велосипедист по той же дороге — "
                    f"{v1} m/s. Во сколько раз скорость мотоциклиста больше?"
                )
            steps_en = [f"Ratio of speeds: {v2} / {v1} = {k}."]
            steps_ru = [f"Отношение скоростей: {v2} / {v1} = {k}."]
            exact_params = {"kind": "ratio", "v1": v1, "v2": v2, "expected": k}
        d = PairDraft(SUBJECT, "", "", Difficulty.SCHOOL, atype, "", question_en=q_en, question_ru=q_ru)
        return _finish(
            _emit_exact(d, ans, steps_en, steps_ru, exact_params, ans_ru=ans_ru),
            "kinem_const",
            Difficulty.SCHOOL,
        )
    v = 2 * int(rng.integers(1, 16))  # 2..30 m/s
    t = int(rng.integers(2, 13))
    s = v * t
    mode = int(rng.integers(0, 3))
    if mode == 0:
        q_en = f"A {agent_en} moving at a constant speed covers {s} m in {t} s. What is the speed?"
        q_ru = f"{agent_ru.capitalize()} движется равномерно и проходит {s} m за {t} s. Чему равна скорость?"
        steps_en = [f"v = s / t = {s} / {t} = {v} m/s."]
        steps_ru = [f"v = s / t = {s} / {t} = {v} m/s."]
        value, units = float(v), "m/s"
        extras = [(float(s + t), "added_values"), (float(s / t / 2) if t else 0.0, "factor_of_2_half")]
    elif mode == 1:
        q_en = f"A {agent_en} moving at a constant speed of {v} m/s travels for {t} s. What distance is covered?"
        q_ru = f"{agent_ru.capitalize()} движется равномерно со скоростью {v} m/s в течение {t} s. Какое расстояние будет пройдено?"
        steps_en = [f"s = v * t = {v} * {t} = {s} m."]
        steps_ru = [f"s = v * t = {v} * {t} = {s} m."]
        value, units = float(s), "m"
        extras = [(float(v + t), "added_values"), (float(s * 2), "factor_of_2")]
    else:
        q_en = f"How much time does a {agent_en} moving at a constant speed of {v} m/s need to cover {s} m?"
        q_ru = (
            f"За какое время {agent_ru} при равномерном движении со скоростью "
            f"{v} m/s преодолеет {s} m?"
        )
        steps_en = [f"t = s / v = {s} / {v} = {t} s."]
        steps_ru = [f"t = s / v = {s} / {v} = {t} s."]
        value, units = float(t), "s"
        extras = [(float(s + v), "added_values"), (float(s * v), "multiplied_values")]
    d = PairDraft(SUBJECT, "", "", Difficulty.SCHOOL, atype, "", question_en=q_en, question_ru=q_ru)
    return _finish(
        _emit(d, rng, atype, value, units, steps_en, steps_ru, extras,
              {"v": v, "t": t, "s": s, "expected": value, "mode": mode}),
        "kinem_const", Difficulty.SCHOOL,
    )


# --------------------------------------------------------------------------- #
# Newton's second law (school)
# --------------------------------------------------------------------------- #
def g_newton2(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    if atype == AnswerType.EXACT:
        k = int(rng.choice([2, 3, 4]))
        phr = int(rng.integers(0, 2))
        if phr == 0:
            q_en = (
                f"If the net force acting on a body is increased by a factor of {k} while its mass stays "
                f"the same, by what factor does the acceleration change?"
            )
            q_ru = (
                f"Если равнодействующую силу, действующую на тело, увеличить в {k} раза при неизменной массе, "
                f"во сколько раз изменится ускорение?"
            )
            steps_en = [f"a = F / m is directly proportional to F, so the acceleration grows by a factor of {k}."]
            steps_ru = [f"a = F / m прямо пропорциональна F, поэтому ускорение вырастет в {k} раза."]
        else:
            q_en = (
                f"The mass of a body is kept unchanged while the net force acting on it is made {k} times "
                f"larger. By what factor does the magnitude of the acceleration grow?"
            )
            q_ru = (
                f"Массу тела не меняют, а модуль действующей на него равнодействующей силы увеличивают "
                f"в {k} раза. Во сколько раз вырастет модуль ускорения тела?"
            )
            steps_en = [f"Proportionality a ~ F/m: with m fixed, multiplying F by {k} multiplies a by {k}."]
            steps_ru = [f"Пропорциональность a ~ F/m: при неизменном m умножение F на {k} умножает a на {k}."]
        d = PairDraft(SUBJECT, "", "", Difficulty.SCHOOL, atype, "", question_en=q_en, question_ru=q_ru)
        return _finish(_emit_exact(d, str(k), steps_en, steps_ru, {"k": k, "expected": k}), "newton2", Difficulty.SCHOOL)
    m = int(rng.integers(2, 51))
    a = float(rng.choice([0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6]))
    f_val = m * a
    mode = int(rng.integers(0, 3))
    if mode == 0:
        q_en = f"A constant force gives a body of mass {m} kg an acceleration of {fmt(a)} m/s^2. What is the magnitude of the force?"
        q_ru = f"Постоянная сила сообщает телу массой {m} kg ускорение {fmt(a)} m/s^2. Чему равен модуль этой силы?"
        steps_en = [f"F = m * a = {m} * {fmt(a)} = {fmt(f_val)} N."]
        steps_ru = [f"F = m * a = {m} * {fmt(a)} = {fmt(f_val)} N."]
        value, units = f_val, "N"
        extras = [(m + a, "added_values"), (m / a if a else 0.0, "divided_wrong_way")]
    elif mode == 1:
        q_en = f"A force of {fmt(f_val)} N acts on a body of mass {m} kg. What is the acceleration of the body?"
        q_ru = f"На тело массой {m} kg действует сила {fmt(f_val)} N. Чему равно ускорение тела?"
        steps_en = [f"a = F / m = {fmt(f_val)} / {m} = {fmt(a)} m/s^2."]
        steps_ru = [f"a = F / m = {fmt(f_val)} / {m} = {fmt(a)} m/s^2."]
        value, units = a, "m/s^2"
        extras = [(f_val * m, "multiplied_values"), (f_val + m, "added_values")]
    else:
        q_en = f"A force of {fmt(f_val)} N gives a body an acceleration of {fmt(a)} m/s^2. What is the mass of the body?"
        q_ru = f"Сила {fmt(f_val)} N сообщает телу ускорение {fmt(a)} m/s^2. Чему равна масса тела?"
        steps_en = [f"m = F / a = {fmt(f_val)} / {fmt(a)} = {m} kg."]
        steps_ru = [f"m = F / a = {fmt(f_val)} / {fmt(a)} = {m} kg."]
        value, units = float(m), "kg"
        extras = [(f_val * a, "multiplied_values"), (f_val + a, "added_values")]
    d = PairDraft(SUBJECT, "", "", Difficulty.SCHOOL, atype, "", question_en=q_en, question_ru=q_ru)
    return _finish(
        _emit(d, rng, atype, value, units, steps_en, steps_ru, extras,
              {"m": m, "a": a, "F": f_val, "expected": value, "mode": mode}),
        "newton2", Difficulty.SCHOOL,
    )


# --------------------------------------------------------------------------- #
# Ohm's law (school)
# --------------------------------------------------------------------------- #
def g_ohm_law(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    mode = int(rng.integers(0, 4))
    if mode == 3:
        i_cur = int(rng.integers(1, 5))
        r_tot = 10 * int(rng.integers(2, 11))
        r1 = 5 * int(rng.integers(1, r_tot // 5))
        r2 = r_tot - r1
        if r1 == 0 or r2 == 0:
            r1, r2 = r_tot // 2, r_tot - r_tot // 2
        u = i_cur * r_tot
        q_en = (
            f"Two resistors of {r1} Ohm and {r2} Ohm are connected in series across a {u} V source. "
            f"What is the current in the circuit?"
        )
        q_ru = (
            f"Два резистора {r1} Ohm и {r2} Ohm соединены последовательно и подключены к источнику "
            f"напряжением {u} V. Чему равна сила тока в цепи?"
        )
        steps_en = [
            f"Series resistance: R = {r1} + {r2} = {r_tot} Ohm.",
            f"I = U / R = {u} / {r_tot} = {i_cur} A.",
        ]
        steps_ru = [
            f"Общее сопротивление: R = {r1} + {r2} = {r_tot} Ohm.",
            f"I = U / R = {u} / {r_tot} = {i_cur} A.",
        ]
        value, units = float(i_cur), "A"
        extras = [(u / r1 if r1 else 0.0, "ignored_second_resistor"), (u + r_tot, "added_values")]
        params = {"mode": "series", "r1": r1, "r2": r2, "u": u, "expected": i_cur}
    else:
        i_cur = float(rng.choice([0.5, 1, 1.5, 2, 2.5, 3, 4, 5]))
        r_res = 5 * int(rng.integers(1, 41))
        u = i_cur * r_res
        if mode == 0:
            q_en = f"A current of {fmt(i_cur)} A flows through a {r_res} Ohm resistor. What is the voltage across the resistor?"
            q_ru = f"Через резистор сопротивлением {r_res} Ohm протекает постоянный ток {fmt(i_cur)} A. Чему равно напряжение на резисторе?"
            steps_en = [f"U = I * R = {fmt(i_cur)} * {r_res} = {fmt(u)} V."]
            steps_ru = [f"U = I * R = {fmt(i_cur)} * {r_res} = {fmt(u)} V."]
            value, units = u, "V"
            extras = [(i_cur + r_res, "added_values"), (u * 2, "factor_of_2")]
        elif mode == 1:
            q_en = f"A voltage of {fmt(u)} V is applied to a {r_res} Ohm resistor. What is the current through it?"
            q_ru = f"К резистору сопротивлением {r_res} Ohm приложено напряжение {fmt(u)} V. Чему равна сила тока в резисторе?"
            steps_en = [f"I = U / R = {fmt(u)} / {r_res} = {fmt(i_cur)} A."]
            steps_ru = [f"I = U / R = {fmt(u)} / {r_res} = {fmt(i_cur)} A."]
            value, units = i_cur, "A"
            extras = [(u * r_res, "multiplied_values"), (u + r_res, "added_values")]
        else:
            q_en = f"When {fmt(u)} V is applied to a resistor the current is {fmt(i_cur)} A. What is the resistance?"
            q_ru = f"При напряжении {fmt(u)} V на резисторе сила тока равна {fmt(i_cur)} A. Чему равно сопротивление резистора?"
            steps_en = [f"R = U / I = {fmt(u)} / {fmt(i_cur)} = {r_res} Ohm."]
            steps_ru = [f"R = U / I = {fmt(u)} / {fmt(i_cur)} = {r_res} Ohm."]
            value, units = float(r_res), "Ohm"
            extras = [(u + i_cur, "added_values"), (r_res * 2, "factor_of_2")]
        params = {"mode": f"single{mode}", "i": i_cur, "r": r_res, "u": u, "expected": value}
    d = PairDraft(SUBJECT, "", "", Difficulty.SCHOOL, atype, "", question_en=q_en, question_ru=q_ru)
    return _finish(_emit(d, rng, atype, value, units, steps_en, steps_ru, extras, params), "ohm_law", Difficulty.SCHOOL)


# --------------------------------------------------------------------------- #
# Power (school)
# --------------------------------------------------------------------------- #
def g_power(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    if bool(rng.integers(0, 2)):
        t = int(rng.integers(2, 61))
        p = 5 * int(rng.integers(1, 61))  # 5..300 W
        w = p * t
        q_en = f"An electric motor performs {w} J of work in {t} s. What is its power?"
        q_ru = f"Электродвигатель совершает работу {w} J за {t} s. Чему равна его мощность?"
        steps_en = [f"P = W / t = {w} / {t} = {p} W."]
        steps_ru = [f"P = W / t = {w} / {t} = {p} W."]
        value, units = float(p), "W"
        extras = [(w + t, "added_values"), (w * t, "multiplied_values")]
        params = {"kind": "mech", "w": w, "t": t, "expected": p}
    else:
        u = int(rng.integers(6, 49))
        i_cur = float(rng.choice([0.5, 1, 1.5, 2, 3, 4, 5]))
        p = u * i_cur
        q_en = f"An appliance operates at {u} V and draws a current of {fmt(i_cur)} A. What is the power it consumes?"
        q_ru = f"Электрический прибор работает при напряжении {u} V и потребляет ток {fmt(i_cur)} A. Чему равна потребляемая мощность?"
        steps_en = [f"P = U * I = {u} * {fmt(i_cur)} = {fmt(p)} W."]
        steps_ru = [f"P = U * I = {u} * {fmt(i_cur)} = {fmt(p)} W."]
        value, units = p, "W"
        extras = [(u + i_cur, "added_values"), (u / i_cur if i_cur else 0.0, "divided_wrong_way")]
        params = {"kind": "el", "u": u, "i": i_cur, "expected": p}
    d = PairDraft(SUBJECT, "", "", Difficulty.SCHOOL, atype, "", question_en=q_en, question_ru=q_ru)
    return _finish(_emit(d, rng, atype, value, units, steps_en, steps_ru, extras, params), "power", Difficulty.SCHOOL)


# --------------------------------------------------------------------------- #
# Thermal energy (school + university)
# --------------------------------------------------------------------------- #
def g_heat_q(rng: np.random.Generator, idx: int, atype: AnswerType, difficulty: Difficulty) -> PairDraft:
    if atype == AnswerType.EXACT:
        k = int(rng.choice([2, 3, 4]))
        phr = int(rng.integers(0, 4))
        if phr == 0:
            q_en = (
                f"The same amount of heat is supplied to two samples of water, and the second sample has "
                f"{k} times the mass of the first. How many times smaller is its temperature change?"
            )
            q_ru = (
                f"Двум порциям воды сообщают одинаковое количество теплоты, при этом масса второй порции "
                f"в {k} раза больше массы первой. Во сколько раз меньше её изменение температуры?"
            )
            steps_en = [f"dT = Q / (m*c): with the same Q the temperature change is inversely proportional to the mass, i.e. {k} times smaller."]
            steps_ru = [f"dT = Q / (m*c): при том же Q изменение температуры обратно пропорционально массе, то есть в {k} раза меньше."]
        elif phr == 1:
            q_en = (
                f"Equal amounts of heat are delivered to two identical burners with {k} times as much water "
                f"on the second one. How many times smaller is the temperature rise of the second portion?"
            )
            q_ru = (
                f"Две одинаковые горелки отдают одинаковое количество теплоты, но на второй находится "
                f"в {k} раза больше воды. Во сколько раз меньше она нагреется?"
            )
            steps_en = [f"For a fixed Q, dT is inversely proportional to m, hence the factor {k}."]
            steps_ru = [f"При фиксированном Q величина dT обратно пропорциональна m, то есть множитель {k}."]
        elif phr == 2:
            q_en = (
                f"An electric kettle heats {k} times more water than usual while transferring the same "
                f"amount of heat as before. By what factor is the water's temperature rise reduced?"
            )
            q_ru = (
                f"Электрический чайник нагревает в {k} раза больше воды, чем обычно, передавая то же "
                f"количество теплоты, что и раньше. Во сколько раз уменьшится нагрев воды?"
            )
            steps_en = [f"dT = Q / (m*c): with Q fixed and m multiplied by {k}, the rise dT is divided by {k}."]
            steps_ru = [f"dT = Q / (m*c): при умножении m на {k} и неизменном Q величина dT уменьшается в {k} раз."]
        else:
            q_en = (
                f"Two vessels with water receive identical heat inputs from identical heaters, but the "
                f"second vessel holds {k} times the mass of water. By what factor is its temperature "
                f"rise smaller?"
            )
            q_ru = (
                f"Два сосуда с водой получают от одинаковых нагревателей одинаковое количество теплоты, "
                f"но во втором сосуде в {k} раза больше воды. Во сколько раз меньше он нагреется?"
            )
            steps_en = [f"Inverse proportionality of dT to m gives the factor {k}."]
            steps_ru = [f"Обратная пропорциональность dT массе даёт множитель {k}."]
        d = PairDraft(SUBJECT, "", "", Difficulty.SCHOOL, atype, "", question_en=q_en, question_ru=q_ru)
        return _finish(_emit_exact(d, str(k), steps_en, steps_ru, {"k": k, "expected": k}), "heat_q", Difficulty.SCHOOL)
    m_tenths = int(rng.integers(1, 31))  # 0.1..3.0 kg
    m = m_tenths / 10
    if difficulty == Difficulty.UNIVERSITY:
        c = float(rng.choice([460, 900, 2100, 4200]))
        dt = int(rng.integers(10, 51, endpoint=False)) or 10
        if bool(rng.integers(0, 2)):
            dt = 2 * (dt // 2)
        q_heat = m * c * dt
        q_en = (
            f"Heating a {fmt(m)} kg sample of an unknown substance from {20} °C to {20 + dt} °C "
            f"required {fmt(q_heat)} J of heat. What is the substance's specific heat capacity?"
        )
        q_ru = (
            f"Для нагревания образца неизвестного вещества массой {fmt(m)} kg от {20} °C "
            f"до {20 + dt} °C потребовалось {fmt(q_heat)} J теплоты. Чему равна удельная "
            f"теплоёмкость этого вещества?"
        )
        steps_en = [
            f"Temperature change: dT = {dt} K.",
            f"c = Q / (m * dT) = {fmt(q_heat)} / ({fmt(m)} * {dt}) = {fmt(c)} J/(kg*K).",
        ]
        steps_ru = [
            f"Изменение температуры: dT = {dt} K.",
            f"c = Q / (m * dT) = {fmt(q_heat)} / ({fmt(m)} * {dt}) = {fmt(c)} J/(kg*K).",
        ]
        value, units = c, "J/(kg*K)"
        extras = [(q_heat / m if m else 0.0, "forgot_dT"), (q_heat * m, "multiplied_extra")]
        params = {"kind": "find_c", "m": m, "dT": dt, "c": c, "Q": q_heat, "expected": c}
        d = PairDraft(SUBJECT, "", "", Difficulty.UNIVERSITY, atype, "", question_en=q_en, question_ru=q_ru)
        return _finish(_emit(d, rng, atype, value, units, steps_en, steps_ru, extras, params), "heat_q", Difficulty.UNIVERSITY)
    dt = 5 * int(rng.integers(1, 13))  # 5..60
    t1 = 5 * int(rng.integers(2, 9))
    t2 = t1 + dt
    q_heat = m * 4200 * dt
    q_en = (
        f"How much heat is needed to heat {fmt(m)} kg of water from {t1} °C to {t2} °C "
        f"(specific heat of water c = 4200 J/(kg*K))?"
    )
    q_ru = (
        f"Какое количество теплоты необходимо, чтобы нагреть {fmt(m)} kg воды от {t1} °C до {t2} °C "
        f"(удельная теплоёмкость воды c = 4200 J/(kg*K))?"
    )
    steps_en = [
        f"Temperature change: dT = {t2} - {t1} = {dt} K.",
        f"Q = m * c * dT = {fmt(m)} * 4200 * {dt} = {fmt(q_heat)} J.",
    ]
    steps_ru = [
        f"Изменение температуры: dT = {t2} - {t1} = {dt} K.",
        f"Q = m * c * dT = {fmt(m)} * 4200 * {dt} = {fmt(q_heat)} J.",
    ]
    value, units = q_heat, "J"
    extras = [(m * 4200 * dt / 2, "factor_of_2_half"), (m * 4200, "forgot_dT"), (4200 * dt, "forgot_mass")]
    params = {"kind": "find_q", "m": m, "dT": dt, "c": 4200, "Q": q_heat, "expected": q_heat}
    d = PairDraft(SUBJECT, "", "", Difficulty.SCHOOL, atype, "", question_en=q_en, question_ru=q_ru)
    return _finish(_emit(d, rng, atype, value, units, steps_en, steps_ru, extras, params), "heat_q", Difficulty.SCHOOL)


def g_heat_school(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    return g_heat_q(rng, idx, atype, Difficulty.SCHOOL)


def g_heat_uni(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    return g_heat_q(rng, idx, atype, Difficulty.UNIVERSITY)


# --------------------------------------------------------------------------- #
# Kinematics with acceleration (university)
# --------------------------------------------------------------------------- #
def g_kinem_accel(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    v0 = 2 * int(rng.integers(0, 11))  # 0..20
    a = float(rng.choice([0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6]))
    t = int(rng.integers(2, 11))
    v = v0 + a * t
    mode = int(rng.integers(0, 3))
    if mode in (0, 2):
        s = v0 * t + a * t * t / 2
        if mode == 0:
            q_en = (
                f"A car speeds up from {v0} m/s with a constant acceleration of {fmt(a)} m/s^2 for {t} s. "
                f"What is its final speed?"
            )
            q_ru = (
                f"Автомобиль движется со скоростью {v0} m/s и разгоняется с постоянным ускорением {fmt(a)} m/s^2 "
                f"в течение {t} s. Чему равна его конечная скорость?"
            )
            steps_en = [f"v = v0 + a*t = {v0} + {fmt(a)}*{t} = {fmt(v)} m/s."]
            steps_ru = [f"v = v0 + a*t = {v0} + {fmt(a)}*{t} = {fmt(v)} m/s."]
            value, units = v, "m/s"
            extras = [(v0 - a * t, "sign_error"), (v0 + a, "forgot_time"), (v0 * a * t, "multiplied_extra")]
        else:
            q_en = (
                f"A body starts moving at {v0} m/s with a constant acceleration of {fmt(a)} m/s^2. "
                f"What distance does it cover in the first {t} s?"
            )
            q_ru = (
                f"Тело движется со скоростью {v0} m/s и постоянным ускорением {fmt(a)} m/s^2. "
                f"Какой путь оно пройдёт за первые {t} s?"
            )
            steps_en = [
                f"s = v0*t + a*t^2/2 = {v0}*{t} + {fmt(a)}*{t}^2/2 = {fmt(v0 * t)} + {fmt(a * t * t / 2)} = {fmt(s)} m.",
            ]
            steps_ru = [
                f"s = v0*t + a*t^2/2 = {v0}*{t} + {fmt(a)}*{t}^2/2 = {fmt(v0 * t)} + {fmt(a * t * t / 2)} = {fmt(s)} m.",
            ]
            value, units = s, "m"
            extras = [(v0 * t, "forgot_acceleration"), (v0 * t * 2, "factor_of_2"), (v * t, "used_final_speed")]
    else:
        q_en = (
            f"The speed of a body increases uniformly from {v0} m/s to {fmt(v)} m/s with an acceleration "
            f"of {fmt(a)} m/s^2. How long does the acceleration take?"
        )
        q_ru = (
            f"Скорость тела равномерно возрастает с {v0} m/s до {fmt(v)} m/s при ускорении {fmt(a)} m/s^2. "
            f"Сколько времени длится разгон?"
        )
        steps_en = [f"t = (v - v0) / a = ({fmt(v)} - {v0}) / {fmt(a)} = {t} s."]
        steps_ru = [f"t = (v - v0) / a = ({fmt(v)} - {v0}) / {fmt(a)} = {t} s."]
        value, units = float(t), "s"
        extras = [((v - v0) * a, "multiplied_instead_of_dividing"), (float(t) + 1, "off_by_one")]
    d = PairDraft(SUBJECT, "", "", Difficulty.UNIVERSITY, atype, "", question_en=q_en, question_ru=q_ru)
    return _finish(
        _emit(d, rng, atype, value, units, steps_en, steps_ru, extras,
              {"v0": v0, "a": a, "t": t, "v": v, "expected": value, "mode": mode}),
        "kinem_accel", Difficulty.UNIVERSITY,
    )


# --------------------------------------------------------------------------- #
# Work-energy (university)
# --------------------------------------------------------------------------- #
def g_work_energy(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    if bool(rng.integers(0, 2)):
        f_val = 10 * int(rng.integers(1, 21))  # 10..200 N
        d_dist = int(rng.integers(2, 31))
        theta = int(rng.choice([0, 60]))
        cos_t = 1.0 if theta == 0 else 0.5
        w = f_val * d_dist * cos_t
        hint = " (cos 60° = 0.5)" if theta == 60 else ""
        hint_ru = " (cos 60° = 0.5)" if theta == 60 else ""
        q_en = (
            f"A sled is pulled along a horizontal path by a force of {f_val} N directed at {theta}° to the "
            f"displacement over a distance of {d_dist} m{hint}. What is the work done by the force?"
        )
        q_ru = (
            f"Сани тянут по горизонтальному пути силой {f_val} N, направленной под углом {theta}° к перемещению, "
            f"на пути {d_dist} m{hint_ru}. Чему равна работа этой силы?"
        )
        steps_en = [
            f"W = F * d * cos({theta}°) = {f_val} * {d_dist} * {cos_t} = {fmt(w)} J.",
        ]
        steps_ru = [
            f"W = F * d * cos({theta}°) = {f_val} * {d_dist} * {cos_t} = {fmt(w)} J.",
        ]
        value, units = w, "J"
        extras = [(f_val * d_dist * 2, "factor_of_2"), (f_val + d_dist, "added_values"), (f_val * d_dist * 0.866, "wrong_angle")]
        params = {"kind": "work", "F": f_val, "d": d_dist, "theta": theta, "expected": w}
    else:
        m = int(rng.integers(2, 21))
        v0 = int(rng.integers(0, 8))
        v1 = v0 + int(rng.integers(1, 6))
        dke = m * (v1 * v1 - v0 * v0) / 2
        q_en = (
            f"The speed of a body of mass {m} kg increases from {v0} m/s to {v1} m/s. "
            f"What is the change in its kinetic energy?"
        )
        q_ru = (
            f"Скорость тела массой {m} kg увеличивается с {v0} m/s до {v1} m/s. "
            f"Чему равно изменение его кинетической энергии?"
        )
        steps_en = [
            f"dKE = m*(v^2 - v0^2)/2 = {m}*({v1 * v1} - {v0 * v0})/2 = {fmt(dke)} J.",
        ]
        steps_ru = [
            f"dKE = m*(v^2 - v0^2)/2 = {m}*({v1 * v1} - {v0 * v0})/2 = {fmt(dke)} J.",
        ]
        value, units = dke, "J"
        extras = [(m * (v1 * v1 + v0 * v0) / 2, "added_speeds"), (m * (v1 - v0) / 2, "forgot_squares"), (m * v1 * v1 / 2, "ignored_initial")]
        params = {"kind": "dke", "m": m, "v0": v0, "v1": v1, "expected": dke}
    d = PairDraft(SUBJECT, "", "", Difficulty.UNIVERSITY, atype, "", question_en=q_en, question_ru=q_ru)
    return _finish(_emit(d, rng, atype, value, units, steps_en, steps_ru, extras, params), "work_energy", Difficulty.UNIVERSITY)


# --------------------------------------------------------------------------- #
# Momentum (university)
# --------------------------------------------------------------------------- #
def g_momentum(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    if atype == AnswerType.EXACT:
        k = int(rng.choice([2, 3, 4]))
        phr = int(rng.integers(0, 4))
        if phr == 0:
            q_en = (
                f"By what factor does the momentum of a body increase if its speed grows {k}-fold "
                f"while its mass stays constant?"
            )
            q_ru = (
                f"Во сколько раз увеличится импульс тела, если его скорость возрастёт в {k} раза "
                f"при неизменной массе?"
            )
            steps_en = [f"p = m*v is proportional to v, so the momentum increases by a factor of {k}."]
            steps_ru = [f"p = m*v пропорционален скорости, поэтому импульс увеличится в {k} раза."]
        elif phr == 1:
            q_en = (
                f"The speed of a cart is raised {k}-fold and its load (mass) is not changed. By what factor "
                f"does the magnitude of the cart's momentum grow?"
            )
            q_ru = (
                f"Скорость тележки увеличили в {k} раза, а её загрузку (массу) не меняли. Во сколько раз "
                f"возрос модуль импульса тележки?"
            )
            steps_en = [f"p = m*v with constant m: the factor is exactly {k}."]
            steps_ru = [f"p = m*v при постоянной m: множитель равен {k}."]
        elif phr == 2:
            q_en = (
                f"A truck travels along a highway; if its speed is made {k} times larger and its cargo is "
                f"unchanged, how many times larger is its momentum?"
            )
            q_ru = (
                f"Грузовик едет по шоссе; если его скорость станет в {k} раза больше, а груз не изменится, "
                f"во сколько раз вырастет его импульс?"
            )
            steps_en = [f"Momentum is proportional to speed at fixed mass: factor {k}."]
            steps_ru = [f"Импульс пропорционален скорости при неизменной массе: множитель {k}."]
        else:
            q_en = (
                f"During a manoeuvre the velocity of a spacecraft grows by a factor of {k} while its mass "
                f"remains the same. By what factor does |m*v| increase?"
            )
            q_ru = (
                f"При манёвре скорость космического аппарата возрастает в {k} раза, а масса остаётся "
                f"прежней. Во сколько раз увеличится модуль импульса |m*v|?"
            )
            steps_en = [f"|m*v| grows in proportion to |v|: the factor is {k}."]
            steps_ru = [f"|m*v| растёт пропорционально |v|: множитель равен {k}."]
        d = PairDraft(SUBJECT, "", "", Difficulty.UNIVERSITY, atype, "", question_en=q_en, question_ru=q_ru)
        return _finish(_emit_exact(d, str(k), steps_en, steps_ru, {"k": k, "expected": k}), "momentum", Difficulty.UNIVERSITY)
    mode = int(rng.integers(0, 2))
    if mode == 0:
        m = int(rng.integers(1, 51))
        v = int(rng.integers(2, 31))
        p = m * v
        q_en = f"A body of mass {m} kg moves in a straight line at {v} m/s. What is the magnitude of its momentum?"
        q_ru = f"Тело массой {m} kg движется прямолинейно со скоростью {v} m/s. Чему равен модуль импульса тела?"
        steps_en = [f"p = m * v = {m} * {v} = {p} kg*m/s."]
        steps_ru = [f"p = m * v = {m} * {v} = {p} kg*m/s."]
        value, units = float(p), "kg*m/s"
        extras = [(m + v, "added_values"), (m / v if v else 0.0, "divided_instead_of_multiplying")]
        params = {"kind": "p", "m": m, "v": v, "expected": p}
    else:
        k = int(rng.integers(1, 4))
        u = int(rng.integers(2, 6))
        v_stick = k * u
        v1 = u * (k + 1)
        m2 = int(rng.integers(1, 9))
        m1 = k * m2
        q_en = (
            f"A body of mass {m1} kg moving at {v1} m/s collides with a body of mass {m2} kg at rest and "
            f"they stick together. What is their common speed after the collision?"
        )
        q_ru = (
            f"Тело массой {m1} kg, движущееся со скоростью {v1} m/s, сталкивается с покоящимся телом "
            f"массой {m2} kg, после чего тела движутся вместе. Чему равна их общая скорость после столкновения?"
        )
        steps_en = [
            f"Momentum conservation: {m1} * {v1} = ({m1} + {m2}) * v.",
            f"v = {m1 * v1} / {m1 + m2} = {v_stick} m/s.",
        ]
        steps_ru = [
            f"Сохранение импульса: {m1} * {v1} = ({m1} + {m2}) * v.",
            f"v = {m1 * v1} / {m1 + m2} = {v_stick} m/s.",
        ]
        value, units = float(v_stick), "m/s"
        extras = [(m1 * v1 / m2 if m2 else 0.0, "ignored_second_mass"), (float(v1), "ignored_collision"), (v_stick + m2, "added_values")]
        params = {"kind": "inelastic", "m1": m1, "v1": v1, "m2": m2, "expected": v_stick}
    d = PairDraft(SUBJECT, "", "", Difficulty.UNIVERSITY, atype, "", question_en=q_en, question_ru=q_ru)
    return _finish(_emit(d, rng, atype, value, units, steps_en, steps_ru, extras, params), "momentum", Difficulty.UNIVERSITY)


# --------------------------------------------------------------------------- #
# Series/parallel networks (university)
# --------------------------------------------------------------------------- #
def g_ohm_circuits(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    r2, r3 = PARALLEL_PAIRS[int(rng.integers(0, len(PARALLEL_PAIRS)))]
    r_par = r2 * r3 / (r2 + r3)
    r1 = 5 * int(rng.integers(2, 13))  # 10..60
    r_tot = r1 + r_par
    i_cur = float(rng.choice([1, 2, 3, 4]))
    u = i_cur * r_tot
    if bool(rng.integers(0, 2)):
        q_en = (
            f"A {r1} Ohm resistor is connected in series with two resistors of {r2} Ohm and {r3} Ohm "
            f"connected in parallel; the network is attached to a {fmt(u)} V source. What is the total current?"
        )
        q_ru = (
            f"Резистор {r1} Ohm соединён последовательно с двумя параллельно соединёнными резисторами "
            f"{r2} Ohm и {r3} Ohm; цепь подключена к источнику с напряжением {fmt(u)} V. Чему равна сила тока в неразветвлённой части цепи?"
        )
        steps_en = [
            f"Parallel part: R_p = {r2}*{r3}/({r2} + {r3}) = {fmt(r_par)} Ohm.",
            f"Total: R = {r1} + {fmt(r_par)} = {fmt(r_tot)} Ohm; I = U / R = {fmt(u)} / {fmt(r_tot)} = {fmt(i_cur)} A.",
        ]
        steps_ru = [
            f"Параллельный участок: R_p = {r2}*{r3}/({r2} + {r3}) = {fmt(r_par)} Ohm.",
            f"Всё сопротивление: R = {r1} + {fmt(r_par)} = {fmt(r_tot)} Ohm; I = U / R = {fmt(u)} / {fmt(r_tot)} = {fmt(i_cur)} A.",
        ]
        value, units = i_cur, "A"
        extras = [(u / (r1 + r2 + r3), "treated_as_series"), (u / r_par if r_par else 0.0, "ignored_series_part")]
        params = {"r1": r1, "r2": r2, "r3": r3, "u": u, "ask": "current", "expected": i_cur}
    else:
        q_en = (
            f"A {r1} Ohm resistor is connected in series with two resistors of {r2} Ohm and {r3} Ohm "
            f"connected in parallel. What is the equivalent resistance of the whole network?"
        )
        q_ru = (
            f"Резистор {r1} Ohm соединён последовательно с двумя параллельно соединёнными резисторами "
            f"{r2} Ohm и {r3} Ohm. Чему равно полное сопротивление цепи?"
        )
        steps_en = [
            f"Parallel part: R_p = {r2}*{r3}/({r2} + {r3}) = {fmt(r_par)} Ohm.",
            f"R = {r1} + {fmt(r_par)} = {fmt(r_tot)} Ohm.",
        ]
        steps_ru = [
            f"Параллельный участок: R_p = {r2}*{r3}/({r2} + {r3}) = {fmt(r_par)} Ohm.",
            f"R = {r1} + {fmt(r_par)} = {fmt(r_tot)} Ohm.",
        ]
        value, units = r_tot, "Ohm"
        extras = [(r1 + r2 + r3, "treated_as_series"), (r_par, "returned_parallel_part"), (r1 + r2, "ignored_one_resistor")]
        params = {"r1": r1, "r2": r2, "r3": r3, "u": u, "ask": "resistance", "expected": r_tot}
    d = PairDraft(SUBJECT, "", "", Difficulty.UNIVERSITY, atype, "", question_en=q_en, question_ru=q_ru)
    return _finish(_emit(d, rng, atype, value, units, steps_en, steps_ru, extras, params), "ohm_circuits", Difficulty.UNIVERSITY)


# --------------------------------------------------------------------------- #
# Ideal gas processes (university)
# --------------------------------------------------------------------------- #
def g_gas_law(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    if bool(rng.integers(0, 2)):  # Boyle (isothermal)
        p1 = 10 * int(rng.integers(2, 31))  # 20..300 kPa
        ratio_num, ratio_den = [(1, 2), (2, 1), (1, 3), (3, 1), (1, 4), (4, 1), (2, 3), (3, 2)][int(rng.integers(0, 8))]
        v1 = ratio_den * int(rng.integers(1, 6))
        v2 = ratio_num * (v1 // ratio_den)
        p2 = p1 * v1 // v2 if (p1 * v1) % v2 == 0 else p1 * v1 / v2
        if v2 < v1:
            q_en = (
                f"A gas is compressed isothermally from {v1} L to {v2} L. If the initial pressure was {p1} kPa, "
                f"what is the final pressure?"
            )
            q_ru = (
                f"Газ изотермически сжимают с объёма {v1} L до {v2} L. Каким станет давление газа, "
                f"если первоначальное давление было {p1} kPa?"
            )
        else:
            q_en = (
                f"A gas expands isothermally from {v1} L to {v2} L. If the initial pressure was {p1} kPa, "
                f"what is the final pressure?"
            )
            q_ru = (
                f"Газ изотермически расширяют с объёма {v1} L до {v2} L. Каким станет давление газа, "
                f"если первоначальное давление было {p1} kPa?"
            )
        steps_en = [
            "Boyle's law: p1 * V1 = p2 * V2.",
            f"p2 = {p1} * {v1} / {v2} = {fmt(p2)} kPa.",
        ]
        steps_ru = [
            "Закон Бойля — Мариотта: p1 * V1 = p2 * V2.",
            f"p2 = {p1} * {v1} / {v2} = {fmt(p2)} kPa.",
        ]
        value, units = float(p2), "kPa"
        extras = [(p1 * v2 / v1 if v1 else 0.0, "inverted_ratio"), (p1 + v2 - v1, "delta_slip"), (p1 * 2, "factor_of_2")]
        params = {"kind": "boyle", "p1": p1, "v1": v1, "v2": v2, "expected": p2}
    else:  # Charles (isobaric)
        t1 = 10 * int(rng.integers(20, 41))  # 200..400 K
        ratio_num, ratio_den = [(3, 2), (4, 3), (5, 4), (2, 1), (3, 1)][int(rng.integers(0, 5))]
        scale = int(rng.integers(2, 7))
        v1 = ratio_den * scale
        v2 = ratio_num * scale
        t2 = t1 * v2 // v1 if (t1 * v2) % v1 == 0 else t1 * v2 / v1
        q_en = (
            f"A gas is heated at constant pressure from {t1} K to {fmt(t2)} K, and its volume becomes {v2} L. "
            f"What was the initial volume?"
        )
        q_ru = (
            f"Газ изобарно нагревают от {t1} K до {fmt(t2)} K, и его объём становится равным {v2} L. "
            f"Каким был начальный объём газа?"
        )
        steps_en = [
            "Charles's law: V1 / T1 = V2 / T2.",
            f"V1 = V2 * T1 / T2 = {v2} * {t1} / {fmt(t2)} = {fmt(v1)} L.",
        ]
        steps_ru = [
            "Закон Гей-Люссака: V1 / T1 = V2 / T2.",
            f"V1 = V2 * T1 / T2 = {v2} * {t1} / {fmt(t2)} = {fmt(v1)} L.",
        ]
        value, units = float(v1), "L"
        extras = [(v2 * t2 / t1 if t1 else 0.0, "inverted_ratio"), (v2 + t2 - t1, "delta_slip"), (v2 * 2, "factor_of_2")]
        params = {"kind": "charles", "t1": t1, "t2": t2, "v1": v1, "v2": v2, "expected": v1}
    d = PairDraft(SUBJECT, "", "", Difficulty.UNIVERSITY, atype, "", question_en=q_en, question_ru=q_ru)
    return _finish(_emit(d, rng, atype, value, units, steps_en, steps_ru, extras, params), "gas_law", Difficulty.UNIVERSITY)


# --------------------------------------------------------------------------- #
# Hydrostatic pressure (university)
# --------------------------------------------------------------------------- #
def g_hydrostatic(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    if atype == AnswerType.EXACT:
        k = int(rng.choice([2, 3, 4, 5]))
        phr = int(rng.integers(0, 2))
        if phr == 0:
            q_en = f"By what factor does the hydrostatic pressure of water grow if the depth increases {k}-fold?"
            q_ru = (
                f"Во сколько раз вырастет гидростатическое давление воды, если глубина погружения "
                f"увеличится в {k} раза?"
            )
            steps_en = [f"P = rho*g*h is proportional to the depth h, so the pressure grows {k}-fold."]
            steps_ru = [f"P = rho*g*h пропорциональна глубине h, поэтому давление вырастет в {k} раза."]
        else:
            q_en = (
                f"A diver measures gauge pressure in the water at two depths, the second one being "
                f"{k} times deeper. How many times larger is the gauge-pressure reading there?"
            )
            q_ru = (
                f"Аквалангист измеряет избыточное давление в воде на двух глубинах, вторая "
                f"из которых в {k} раза больше первой. Во сколько раз больше будет второе показание?"
            )
            steps_en = [f"Gauge pressure rho*g*h grows linearly with depth, so the ratio is {k}."]
            steps_ru = [f"Избыточное давление rho*g*h растёт линейно с глубиной, поэтому отношение равно {k}."]
        d = PairDraft(SUBJECT, "", "", Difficulty.UNIVERSITY, atype, "", question_en=q_en, question_ru=q_ru)
        return _finish(_emit_exact(d, str(k), steps_en, steps_ru, {"k": k, "expected": k}), "hydrostatic", Difficulty.UNIVERSITY)
    h = float(rng.choice([2.5, 3, 4, 5, 7.5, 10, 12.5, 15, 20, 25, 30, 40, 50]))
    if bool(rng.integers(0, 2)):
        press = 1000 * G98 * h
        q_en = f"Find the gauge pressure of water at a depth of {fmt(h)} m (rho = 1000 kg/m^3, g = 9.8 m/s^2)."
        q_ru = f"Вычислите избыточное давление воды на глубине {fmt(h)} m (rho = 1000 kg/m^3, g = 9.8 m/s^2)."
        steps_en = [f"P = rho * g * h = 1000 * 9.8 * {fmt(h)} = {fmt(press)} Pa."]
        steps_ru = [f"P = rho * g * h = 1000 * 9.8 * {fmt(h)} = {fmt(press)} Pa."]
        value, units = press, "Pa"
        extras = [(1000 * h, "forgot_g"), (G98 * h, "forgot_rho"), (press / 2, "factor_of_2_half")]
        params = {"kind": "pressure", "h": h, "expected": press}
    else:
        area = float(rng.choice([0.02, 0.04, 0.05, 0.1, 0.2, 0.25, 0.5]))
        press = 1000 * G98 * h
        force = press * area
        q_en = (
            f"A horizontal porthole of area {fmt(area)} m^2 is located at a depth of {fmt(h)} m in fresh water "
            f"(rho = 1000 kg/m^3, g = 9.8 m/s^2). What is the force of the water on it?"
        )
        q_ru = (
            f"Горизонтальный иллюминатор площадью {fmt(area)} m^2 находится на глубине {fmt(h)} m в пресной воде "
            f"(rho = 1000 kg/m^3, g = 9.8 m/s^2). Чему равна действующая на него сила давления воды?"
        )
        steps_en = [
            f"P = rho * g * h = 1000 * 9.8 * {fmt(h)} = {fmt(press)} Pa.",
            f"F = P * S = {fmt(press)} * {fmt(area)} = {fmt(force)} N.",
        ]
        steps_ru = [
            f"P = rho * g * h = 1000 * 9.8 * {fmt(h)} = {fmt(press)} Pa.",
            f"F = P * S = {fmt(press)} * {fmt(area)} = {fmt(force)} N.",
        ]
        value, units = force, "N"
        extras = [(press + area, "added_values"), (force * 2, "factor_of_2"), (press / area if area else 0.0, "divided_instead_of_multiplying")]
        params = {"kind": "force", "h": h, "area": area, "expected": force}
    d = PairDraft(SUBJECT, "", "", Difficulty.UNIVERSITY, atype, "", question_en=q_en, question_ru=q_ru)
    return _finish(_emit(d, rng, atype, value, units, steps_en, steps_ru, extras, params), "hydrostatic", Difficulty.UNIVERSITY)


# --------------------------------------------------------------------------- #
# Thin lenses (university + olympiad)
# --------------------------------------------------------------------------- #
def _lens_triple(rng: np.random.Generator) -> tuple[int, int, int]:
    """Nice (f, d_o, d_i) with d_o > f so the real image distance is an integer."""
    for _ in range(200):
        f = int(rng.integers(4, 31))
        d_o = int(rng.integers(f + 1, 4 * f + 20))
        num = f * d_o
        if num % (d_o - f) == 0:
            d_i = num // (d_o - f)
            if d_i <= 120 and d_i != d_o:
                return f, d_o, d_i
    return 10, 15, 30


def g_lenses(rng: np.random.Generator, idx: int, atype: AnswerType, difficulty: Difficulty) -> PairDraft:
    if atype == AnswerType.EXACT:
        f = int(rng.integers(4, 31))
        d_o = f + int(rng.integers(1, max(2, f // 2)))
        if bool(rng.integers(0, 2)):
            d_o = f + int(rng.integers(1, max(2, f // 2)))  # d_o > f -> real
            ans = "real"
        else:
            d_o = max(2, f - int(rng.integers(1, max(2, f // 2))))  # d_o < f -> virtual
            ans = "virtual"
        q_en = (
            f"An object stands {d_o} cm from a converging lens with focal length {f} cm. "
            f"Is the image of the object real or virtual?"
        )
        q_ru = (
            f"Предмет находится на расстоянии {d_o} cm от собирающей линзы с фокусным расстоянием {f} cm. "
            f"Изображение предмета действительное или мнимое?"
        )
        steps_en = [
            "1/f = 1/d_o + 1/d_i: the sign of d_i depends on whether d_o is larger than f.",
            f"Here d_o = {d_o} cm and f = {f} cm, so the image is {ans}.",
        ]
        ans_ru = "действительное" if ans == "real" else "мнимое"
        steps_ru = [
            "1/f = 1/d_o + 1/d_i: знак d_i зависит от того, больше ли d_o, чем f.",
            f"Здесь d_o = {d_o} cm, а f = {f} cm, поэтому изображение {ans_ru}.",
        ]
        d = PairDraft(SUBJECT, "", "", difficulty, atype, "", question_en=q_en, question_ru=q_ru)
        return _finish(
            _emit_exact(
                d,
                ans,
                steps_en,
                steps_ru,
                {"f": f, "do": d_o, "expected_text": ans, "kind": "real_virtual"},
                ans_ru=ans_ru,
            ),
            "lenses", difficulty,
        )
    f, d_o, d_i = _lens_triple(rng)
    if difficulty == Difficulty.OLYMPIAD or bool(rng.integers(0, 2)):
        m_mag = d_i / d_o
        mag_str = fmt(m_mag)
        q_en = (
            f"A converging lens has a focal length of {f} cm. An object is placed {d_o} cm from the lens. "
            f"What is the magnification m = d_i / d_o of the image?"
        )
        q_ru = (
            f"Собирающая линза имеет фокусное расстояние {f} cm. Предмет расположен на расстоянии {d_o} cm "
            f"от линзы. Чему равно увеличение линзы m = d_i / d_o?"
        )
        steps_en = [
            f"1/f = 1/d_o + 1/d_i gives d_i = f*d_o/(d_o - f) = {f}*{d_o}/{d_o - f} = {d_i} cm.",
            f"m = d_i / d_o = {d_i} / {d_o} = {mag_str}.",
        ]
        steps_ru = [
            f"Из 1/f = 1/d_o + 1/d_i получаем d_i = f*d_o/(d_o - f) = {f}*{d_o}/{d_o - f} = {d_i} cm.",
            f"m = d_i / d_o = {d_i} / {d_o} = {mag_str}.",
        ]
        value, units = m_mag, ""
        extras = [(d_i / f if f else 0.0, "wrong_ratio"), (d_o / d_i if d_i else 0.0, "inverted_ratio"), (float(d_i), "returned_image_distance")]
        params = {"f": f, "do": d_o, "di": d_i, "ask": "magnification", "expected": m_mag}
    else:
        q_en = (
            f"A converging lens has a focal length of {f} cm. An object is placed {d_o} cm from the lens. "
            f"At what distance from the lens is the image formed?"
        )
        q_ru = (
            f"Собирающая линза имеет фокусное расстояние {f} cm. Предмет находится на расстоянии {d_o} cm "
            f"от линзы. На каком расстоянии от линзы образуется изображение?"
        )
        steps_en = [
            "Thin-lens equation: 1/f = 1/d_o + 1/d_i.",
            f"d_i = f*d_o/(d_o - f) = {f}*{d_o}/({d_o} - {f}) = {d_i} cm.",
        ]
        steps_ru = [
            "Формула тонкой линзы: 1/f = 1/d_o + 1/d_i.",
            f"d_i = f*d_o/(d_o - f) = {f}*{d_o}/({d_o} - {f}) = {d_i} cm.",
        ]
        value, units = float(d_i), "cm"
        extras = [(float(f), "returned_focal_length"), (float(d_o), "returned_object_distance"), (float(2 * f), "two_focal_lengths")]
        params = {"f": f, "do": d_o, "di": d_i, "ask": "distance", "expected": d_i}
    d = PairDraft(SUBJECT, "", "", difficulty, atype, "", question_en=q_en, question_ru=q_ru)
    return _finish(_emit(d, rng, atype, value, units, steps_en, steps_ru, extras, params), "lenses", difficulty)


def g_lenses_uni(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    return g_lenses(rng, idx, atype, Difficulty.UNIVERSITY)


def g_lenses_olym(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    f1, d_o1, d_i1 = _lens_triple(rng)
    f2, d_o2, d_i2 = _lens_triple(rng)
    separation = d_i1 + d_o2
    m1 = d_i1 / d_o1
    m2 = d_i2 / d_o2
    total_mag = m1 * m2
    phr = idx % 3
    if phr == 0:
        q_en = (
            f"Two thin converging lenses have focal lengths {f1} cm and {f2} cm and are {separation} cm "
            f"apart. An object is {d_o1} cm before the first lens. Its first image lies between the lenses "
            "and is a real object for the second lens. What is the magnitude of the total magnification?"
        )
        q_ru = (
            f"Две тонкие собирающие линзы с фокусными расстояниями {f1} cm и {f2} cm находятся на "
            f"расстоянии {separation} cm друг от друга. Предмет расположен в {d_o1} cm перед первой линзой; "
            "её изображение лежит между линзами и служит действительным предметом для второй. "
            "Чему равен модуль общего увеличения?"
        )
    elif phr == 1:
        q_en = (
            f"An optical bench holds two ideal converging lenses, f1 = {f1} cm and f2 = {f2} cm, separated "
            f"by {separation} cm. A real object is {d_o1} cm to the left of lens 1, and the intermediate "
            "image is before lens 2. Find the absolute overall lateral magnification."
        )
        q_ru = (
            f"На оптической скамье установлены две идеальные собирающие линзы: f1 = {f1} cm и f2 = {f2} cm; "
            f"расстояние между ними {separation} cm. Действительный предмет находится в {d_o1} cm слева от "
            "первой линзы, а промежуточное изображение — перед второй. Найдите модуль полного поперечного увеличения."
        )
    else:
        q_en = (
            f"Light passes through two ideal thin converging lenses {separation} cm apart. Their focal lengths "
            f"are {f1} cm and {f2} cm; the object distance for the first lens is {d_o1} cm. The first real "
            "image becomes the second lens's real object. What total magnification magnitude results?"
        )
        q_ru = (
            f"Свет проходит через две идеальные тонкие собирающие линзы, разделённые расстоянием {separation} cm. "
            f"Их фокусные расстояния равны {f1} cm и {f2} cm, а предмет находится в {d_o1} cm от первой линзы. "
            "Первое действительное изображение становится предметом для второй линзы. Каков модуль общего увеличения?"
        )
    steps_en = [
        f"Lens 1 gives d_i1 = f1*d_o1/(d_o1-f1) = {d_i1} cm, so the second object distance is "
        f"{separation} - {d_i1} = {d_o2} cm.",
        f"Lens 2 gives d_i2 = f2*d_o2/(d_o2-f2) = {d_i2} cm.",
        f"Magnification magnitudes multiply: ({d_i1}/{d_o1})*({d_i2}/{d_o2}) = {fmt(total_mag)}.",
    ]
    steps_ru = [
        f"Первая линза даёт d_i1 = f1*d_o1/(d_o1-f1) = {d_i1} cm, поэтому расстояние до предмета "
        f"для второй линзы равно {separation} - {d_i1} = {d_o2} cm.",
        f"Вторая линза даёт d_i2 = f2*d_o2/(d_o2-f2) = {d_i2} cm.",
        f"Модули увеличений перемножаются: ({d_i1}/{d_o1})*({d_i2}/{d_o2}) = {fmt(total_mag)}.",
    ]
    extras = [
        (m1, "used_first_lens_only"),
        (m2, "used_second_lens_only"),
        (1 / total_mag, "inverted_total_magnification"),
        ((d_i1 + d_i2) / (d_o1 + d_o2), "added_distances_before_ratio"),
    ]
    params = {
        "kind": "two_lens",
        "f1": f1,
        "do1": d_o1,
        "di1": d_i1,
        "separation": separation,
        "f2": f2,
        "do2": d_o2,
        "di2": d_i2,
        "expected": total_mag,
        "challenge_concepts": ["sequential thin-lens imaging", "compound magnification"],
        "challenge_feature": "The intermediate-image position determines the second signed object distance.",
    }
    d = PairDraft(
        SUBJECT,
        "",
        "",
        Difficulty.OLYMPIAD,
        atype,
        "",
        question_en=q_en,
        question_ru=q_ru,
    )
    return _finish(
        _emit(d, rng, atype, total_mag, "", steps_en, steps_ru, extras, params),
        "lenses_olym",
        Difficulty.OLYMPIAD,
    )


# --------------------------------------------------------------------------- #
# Uniform circular motion (olympiad)
# --------------------------------------------------------------------------- #
def g_circular(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    if idx % 2 == 0:
        mass = int(rng.integers(1, 7))
        radius = int(rng.integers(3, 11))
        height_factor = int(rng.integers(3, 6))
        height = height_factor * radius
        speed_sq_top = 20 * (height - 2 * radius)
        normal = mass * (speed_sq_top / radius - 10)
        phr = idx % 3
        if phr == 0:
            q_en = (
                f"A {mass} kg bead starts from rest at height {height} m above the bottom of a vertical "
                f"frictionless loop of radius {radius} m and moves on its inside (g = 10 m/s^2). "
                "Assuming it remains in contact, what normal force does the track exert at the top?"
            )
            q_ru = (
                f"Бусинка массой {mass} kg начинает движение из состояния покоя с высоты {height} m над "
                f"нижней точкой вертикальной гладкой петли радиуса {radius} m и движется по её внутренней "
                "стороне (g = 10 m/s^2). Считая, что контакт не теряется, найдите силу реакции в верхней точке?"
            )
        elif phr == 1:
            q_en = (
                f"A small {mass} kg cart is released from rest {height} m above the bottom of an ideal "
                f"vertical loop with radius {radius} m. With no friction and g = 10 m/s^2, what is the "
                "track's normal force on the cart at the loop's top?"
            )
            q_ru = (
                f"Тележку массой {mass} kg отпускают без начальной скорости с высоты {height} m над нижней "
                f"точкой идеальной вертикальной петли радиуса {radius} m. Трения нет, g = 10 m/s^2. "
                "Какова сила реакции пути в верхней точке петли?"
            )
        else:
            q_en = (
                f"Inside a smooth vertical circular track of radius {radius} m, a {mass} kg block is released "
                f"from rest at height {height} m measured from the bottom. Take g = 10 m/s^2 and assume "
                "continuous contact. Find the normal force at the highest point."
            )
            q_ru = (
                f"Внутри гладкой вертикальной круговой дорожки радиуса {radius} m брусок массой {mass} kg "
                f"отпускают без начальной скорости с высоты {height} m от нижней точки. Примите g = 10 m/s^2 "
                "и считайте контакт непрерывным. Найдите силу реакции в верхней точке?"
            )
        steps_en = [
            f"Energy conservation from height {height} m to the top at 2R = {2 * radius} m gives "
            f"v_top^2 = 2g({height} - {2 * radius}) = {fmt(speed_sq_top)} m^2/s^2.",
            f"At the top, mg + N = m*v_top^2/R, so N = {mass}*({fmt(speed_sq_top)}/{radius} - 10) "
            f"= {fmt(normal)} N.",
        ]
        steps_ru = [
            f"Из сохранения энергии между высотой {height} m и верхней точкой 2R = {2 * radius} m: "
            f"v_top^2 = 2g({height} - {2 * radius}) = {fmt(speed_sq_top)} m^2/s^2.",
            f"В верхней точке mg + N = m*v_top^2/R, поэтому N = {mass}*({fmt(speed_sq_top)}/{radius} - 10) "
            f"= {fmt(normal)} N.",
        ]
        value, units = normal, "N"
        extras = [
            (mass * speed_sq_top / radius, "forgot_weight_at_top"),
            (mass * (speed_sq_top / radius + 10), "wrong_force_direction"),
            (mass * 10, "returned_weight"),
        ]
        params = {
            "kind": "vertical_loop_normal",
            "mass": mass,
            "radius": radius,
            "height": height,
            "g": 10.0,
            "expected": normal,
            "challenge_concepts": ["mechanical-energy conservation", "centripetal force balance"],
            "challenge_feature": "At the top, gravity and the normal force point toward the center.",
        }
    else:
        radius = 10 * int(rng.integers(2, 7))
        mu = float(rng.choice([0.10, 0.15, 0.20, 0.25]))
        sin_theta, cos_theta = 0.5, 0.866
        ratio = (sin_theta + mu * cos_theta) / (cos_theta - mu * sin_theta)
        max_speed = math.sqrt(radius * 10 * ratio)
        phr = idx % 3
        if phr == 0:
            q_en = (
                f"A car rounds a {radius} m radius road banked at 30°. The coefficient of static friction is "
                f"{fmt(mu)}; take g = 10 m/s^2, sin 30° = 0.5, cos 30° = 0.866. At the high-speed "
                "slipping threshold friction acts down the slope. What is the maximum speed?"
            )
            q_ru = (
                f"Автомобиль проходит вираж радиуса {radius} m с углом наклона 30°. Коэффициент трения покоя "
                f"равен {fmt(mu)}; g = 10 m/s^2, sin 30° = 0.5, cos 30° = 0.866. На пределе заноса при "
                "большой скорости трение направлено вниз по склону. Какова максимальная скорость?"
            )
        elif phr == 1:
            q_en = (
                f"On a 30° banked circular test track of radius {radius} m, tire-road static friction has "
                f"coefficient {fmt(mu)}. Using g = 10 m/s^2, sin 30° = 0.5 and cos 30° = 0.866, find the "
                "largest speed before the car tends to slide outward."
            )
            q_ru = (
                f"Круговая испытательная трасса радиуса {radius} m наклонена под углом 30°, коэффициент "
                f"трения покоя шин равен {fmt(mu)}. При g = 10 m/s^2, sin 30° = 0.5 и cos 30° = 0.866 "
                "найдите наибольшую скорость до начала скольжения автомобиля наружу?"
            )
        else:
            q_en = (
                f"A banked turn has radius {radius} m and angle 30°. With static-friction coefficient "
                f"{fmt(mu)}, determine the upper limiting speed; use g = 10 m/s^2, sin 30° = 0.5 and "
                "cos 30° = 0.866, and take limiting friction down the bank."
            )
            q_ru = (
                f"Вираж радиуса {radius} m наклонён под углом 30°. При коэффициенте трения покоя {fmt(mu)} "
                f"определите верхнюю предельную скорость; используйте g = 10 m/s^2, sin 30° = 0.5, "
                "cos 30° = 0.866 и считайте предельное трение направленным вниз по склону?"
            )
        steps_en = [
            "At impending outward slip, N*cos(theta) - mu*N*sin(theta) = mg, while "
            "N*sin(theta) + mu*N*cos(theta) = m*v^2/r.",
            f"Eliminating N gives v = sqrt(rg*(sin+mu*cos)/(cos-mu*sin)) = {fmt(max_speed)} m/s.",
        ]
        steps_ru = [
            "На грани скольжения наружу N*cos(theta) - mu*N*sin(theta) = mg, а "
            "N*sin(theta) + mu*N*cos(theta) = m*v^2/r.",
            f"Исключая N, получаем v = sqrt(rg*(sin+mu*cos)/(cos-mu*sin)) = {fmt(max_speed)} m/s.",
        ]
        value, units = max_speed, "m/s"
        extras = [
            (math.sqrt(radius * 10 * sin_theta / cos_theta), "ignored_friction"),
            (
                math.sqrt(radius * 10 * (sin_theta - mu * cos_theta) / (cos_theta + mu * sin_theta)),
                "used_low_speed_limit",
            ),
            (radius * 10 * ratio, "forgot_square_root"),
        ]
        params = {
            "kind": "banked_friction_max",
            "radius": radius,
            "mu": mu,
            "sin": sin_theta,
            "cos": cos_theta,
            "g": 10.0,
            "expected": max_speed,
            "challenge_concepts": ["static-friction limit", "banked centripetal force resolution"],
            "challenge_feature": "The high-speed limit fixes friction down the bank and changes both force components.",
        }
    d = PairDraft(SUBJECT, "", "", Difficulty.OLYMPIAD, atype, "", question_en=q_en, question_ru=q_ru)
    return _finish(_emit(d, rng, atype, value, units, steps_en, steps_ru, extras, params), "circular", Difficulty.OLYMPIAD)


# --------------------------------------------------------------------------- #
# Projectile motion (olympiad)
# --------------------------------------------------------------------------- #
def g_projectile(
    rng: np.random.Generator,
    idx: int,
    atype: AnswerType,
) -> PairDraft:
    """Challenge projectile tasks with explicit idealizations and root handling."""
    if atype == AnswerType.EXACT:
        theta = int(rng.choice([20, 25, 30, 35, 40]))
        other = 90 - theta
        phr = idx % 4
        if phr == 0:
            q_en = (
                "With negligible air resistance, a projectile is launched and lands at the same elevation. "
                f"At fixed speed it reaches a target using angle {theta}°. What other angle in (0°, 90°) "
                "gives the same range?"
            )
            q_ru = (
                "Сопротивлением воздуха пренебрегают; снаряд стартует и падает на одной высоте. При фиксированной "
                f"скорости он попадает в цель под углом {theta}°. Какой другой угол из (0°, 90°) даёт ту же дальность?"
            )
        elif phr == 1:
            q_en = (
                f"A no-drag projectile travels between points at equal height. One launch angle is {theta}°, "
                "and the speed is unchanged. Which second acute launch angle reaches the same point?"
            )
            q_ru = (
                f"Снаряд без сопротивления воздуха летит между точками одной высоты. Один угол броска равен "
                f"{theta}°, скорость не меняется. Какой второй острый угол приводит в ту же точку?"
            )
        elif phr == 2:
            q_en = (
                f"For ideal projectile motion with equal launch and landing elevations, angle {theta}° is one "
                "solution for a specified range at fixed speed. What is the distinct companion angle?"
            )
            q_ru = (
                f"При идеальном движении с одинаковыми высотами старта и падения угол {theta}° является одним "
                "решением для заданной дальности при фиксированной скорости. Каков второй отличный угол?"
            )
        else:
            q_en = (
                f"Ignoring drag and keeping the launch speed fixed, a ball fired at {theta}° returns to its "
                "launch height at a certain horizontal distance. At what other acute angle does it return at "
                "that same distance?"
            )
            q_ru = (
                f"Пренебрегая сопротивлением и сохраняя скорость броска, мяч, запущенный под углом {theta}°, "
                "возвращается на высоту старта на некотором расстоянии. При каком другом остром угле расстояние "
                "будет тем же?"
            )
        steps_en = [
            "For equal launch and landing heights, R = v0^2*sin(2*theta)/g.",
            f"sin(2*theta) is unchanged by 2*theta -> 180° - 2*theta, so the companion angle is "
            f"90° - {theta}° = {other}°.",
        ]
        steps_ru = [
            "При одинаковых высотах старта и падения R = v0^2*sin(2*theta)/g.",
            f"sin(2*theta) не меняется при замене 2*theta на 180° - 2*theta, поэтому второй угол равен "
            f"90° - {theta}° = {other}°.",
        ]
        params = {
            "kind": "complementary_angle",
            "theta": theta,
            "expected": other,
            "challenge_concepts": ["projectile range", "trigonometric supplementary-angle symmetry"],
            "challenge_feature": "The same-range equation has two distinct acute-angle branches.",
        }
        d = PairDraft(
            SUBJECT,
            "",
            "",
            Difficulty.OLYMPIAD,
            atype,
            "",
            question_en=q_en,
            question_ru=q_ru,
        )
        return _finish(
            _emit_exact(d, str(other), steps_en, steps_ru, params),
            "projectile",
            Difficulty.OLYMPIAD,
        )

    trig = {
        30: (0.5, 0.866),
        45: (0.7071, 0.7071),
        60: (0.866, 0.5),
    }
    theta = int(rng.choice(list(trig)))
    sin_theta, cos_theta = trig[theta]
    v0 = int(rng.choice(range(12, 33, 2)))
    height = 5 * int(rng.integers(1, 7))
    vy = v0 * sin_theta
    vx = v0 * cos_theta
    discriminant = vy * vy + 20 * height
    flight_time = (vy + math.sqrt(discriminant)) / 10
    horizontal_range = vx * flight_time
    phr = idx % 3
    if phr == 0:
        q_en = (
            f"From a platform {height} m above level ground, a ball is launched at {v0} m/s and {theta}° "
            f"above horizontal. Ignore drag; use g = 10 m/s^2, sin {theta}° = {sin_theta}, "
            f"cos {theta}° = {cos_theta}. How far horizontally from the platform does it land?"
        )
        q_ru = (
            f"С платформы высотой {height} m над горизонтальной землёй мяч бросают со скоростью {v0} m/s "
            f"под углом {theta}° к горизонту. Сопротивлением пренебречь; g = 10 m/s^2, "
            f"sin {theta}° = {sin_theta}, cos {theta}° = {cos_theta}. На каком горизонтальном расстоянии он упадёт?"
        )
    elif phr == 1:
        q_en = (
            f"A projectile leaves a cliff edge {height} m above flat ground with speed {v0} m/s at angle "
            f"{theta}°. With no air resistance, g = 10 m/s^2, sin {theta}° = {sin_theta} and "
            f"cos {theta}° = {cos_theta}, find its horizontal displacement before impact."
        )
        q_ru = (
            f"Снаряд вылетает с края обрыва высотой {height} m над ровной землёй со скоростью {v0} m/s "
            f"под углом {theta}°. Сопротивления воздуха нет; g = 10 m/s^2, sin {theta}° = {sin_theta}, "
            f"cos {theta}° = {cos_theta}. Найдите горизонтальное перемещение до удара о землю?"
        )
    else:
        q_en = (
            f"An object is projected from a tower {height} m high at {v0} m/s, {theta}° above horizontal. "
            f"Assume flat ground and negligible drag; take g = 10 m/s^2, sin {theta}° = {sin_theta}, "
            f"cos {theta}° = {cos_theta}. What horizontal range is obtained?"
        )
        q_ru = (
            f"Тело бросают с башни высотой {height} m со скоростью {v0} m/s под углом {theta}° к горизонту. "
            f"Земля горизонтальна, сопротивлением пренебречь; g = 10 m/s^2, sin {theta}° = {sin_theta}, "
            f"cos {theta}° = {cos_theta}. Какова горизонтальная дальность?"
        )
    steps_en = [
        f"Vertical motion gives 0 = {height} + {fmt(vy)}t - 5t^2; the positive root is "
        f"t = ({fmt(vy)} + sqrt({fmt(discriminant)}))/10 = {fmt(flight_time)} s.",
        f"The horizontal speed is {v0}*{cos_theta} = {fmt(vx)} m/s, so x = v_x*t = "
        f"{fmt(horizontal_range)} m.",
    ]
    steps_ru = [
        f"По вертикали 0 = {height} + {fmt(vy)}t - 5t^2; положительный корень равен "
        f"t = ({fmt(vy)} + sqrt({fmt(discriminant)}))/10 = {fmt(flight_time)} s.",
        f"Горизонтальная скорость равна {v0}*{cos_theta} = {fmt(vx)} m/s, поэтому x = v_x*t = "
        f"{fmt(horizontal_range)} m.",
    ]
    equal_height_time = 2 * vy / 10
    extras = [
        (vx * equal_height_time, "ignored_launch_height"),
        (vx * math.sqrt(2 * height / 10), "ignored_initial_vertical_speed"),
        (v0 * flight_time, "used_total_speed_horizontally"),
    ]
    params = {
        "kind": "elevated_range",
        "height": height,
        "v0": v0,
        "theta": theta,
        "sin": sin_theta,
        "cos": cos_theta,
        "g": 10.0,
        "expected": horizontal_range,
        "challenge_concepts": ["vertical quadratic flight time", "horizontal component motion"],
        "challenge_feature": "Only the positive time root is physical because launch and landing elevations differ.",
    }
    d = PairDraft(
        SUBJECT,
        "",
        "",
        Difficulty.OLYMPIAD,
        atype,
        "",
        question_en=q_en,
        question_ru=q_ru,
    )
    return _finish(
        _emit(d, rng, atype, horizontal_range, "m", steps_en, steps_ru, extras, params),
        "projectile",
        Difficulty.OLYMPIAD,
    )


GENERATORS: dict[str, Any] = {
    "kinem_const": g_kinem_const,
    "newton2": g_newton2,
    "ohm_law": g_ohm_law,
    "power": g_power,
    "heat_q": g_heat_school,
    "heat_q_uni": g_heat_uni,
    "kinem_accel": g_kinem_accel,
    "work_energy": g_work_energy,
    "momentum": g_momentum,
    "ohm_circuits": g_ohm_circuits,
    "gas_law": g_gas_law,
    "hydrostatic": g_hydrostatic,
    "lenses": g_lenses_uni,
    "lenses_olym": g_lenses_olym,
    "circular": g_circular,
    "projectile": g_projectile,
}

KEY_ALIASES: dict[str, str] = {
    "heat_q_uni": "heat_q",
    "lenses_olym": "lenses",
}

__all__ = ["SUBJECT", "PREFIX", "TOPICS", "RUBRICS", "SPEC", "GENERATORS", "KEY_ALIASES"]
