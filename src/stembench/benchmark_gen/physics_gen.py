"""Original bilingual (ru/en) physics item generators.

All prompts state the numeric conventions they rely on (g = 9.8 or 10 m/s^2,
c_water = 4200 J/(kg*K), pi = 3.14) so that every item is self-contained.
Units are untranslated SI symbols in both languages; the decimal separator is
"." in both languages (see ``_core`` docstring).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from stembench.schemas import AnswerType, Difficulty, Subject

from ._core import (
    PairDraft,
    fmt,
    pick_distractors,
    sol_en,
    sol_ru,
)
from .math_gen import _mc_numeric, _set_numeric, _std_pool, pick_distractors_str

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
        "Thin-lens setup with magnification; multi-step arithmetic.",
        "Задача о тонкой линзе с увеличением; многошаговая арифметика.",
    ),
    ("circular", Difficulty.OLYMPIAD): (
        "Centripetal acceleration a = v^2/r or orbital speed from the period.",
        "Центростремительное ускорение a = v^2/r или скорость по периоду обращения.",
    ),
    ("projectile", Difficulty.OLYMPIAD): (
        "Projectile range, flight time or maximum height at a special angle.",
        "Дальность, время полёта или максимальная высота при особом угле броска.",
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
    ("heat_q", Difficulty.UNIVERSITY, 4, AnswerType.NUMERIC),
    ("heat_q", Difficulty.UNIVERSITY, 4, AnswerType.MC),
    ("circular", Difficulty.OLYMPIAD, 2, AnswerType.NUMERIC),
    ("circular", Difficulty.OLYMPIAD, 8, AnswerType.MC),
    ("projectile", Difficulty.OLYMPIAD, 6, AnswerType.NUMERIC),
    ("projectile", Difficulty.OLYMPIAD, 4, AnswerType.MC),
    ("projectile", Difficulty.OLYMPIAD, 4, AnswerType.EXACT),
    ("lenses_olym", Difficulty.OLYMPIAD, 6, AnswerType.MC),
]

G98 = 9.8
PARALLEL_PAIRS = [(30, 60), (20, 30), (6, 3), (12, 4), (60, 20), (15, 10), (24, 12), (10, 40)]


def _finish(d: PairDraft, key: str, difficulty: Difficulty) -> PairDraft:
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
) -> PairDraft:
    d.canonical = ans
    d.solution_en = sol_en(steps_en, ans, "")
    d.solution_ru = sol_ru(steps_ru, ans, "")
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
        if bool(rng.integers(0, 2)):
            v1 = int(rng.integers(1, 4))
            v2 = v1 + int(rng.integers(1, 4))
            d0 = 5 * int(rng.integers(2, 19))
            t_big = int(rng.choice([5, 10, 15, 20, 30, 40, 60]))
            closing = v2 - v1
            catch = d0 / closing <= t_big
            ans = "yes" if catch else "no"
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
            q_en = (
                f"A high-speed train travels at {v2} m/s while a freight train travels at {v1} m/s "
                f"on a parallel track. How many times is the speed of the high-speed train greater?"
            )
            q_ru = (
                f"Скорый поезд движется со скоростью {v2} m/s, а грузовой поезд по параллельному пути — "
                f"со скоростью {v1} m/s. Во сколько раз скорость скорого поезда больше?"
            )
            steps_en = [f"Ratio of speeds: {v2} / {v1} = {k}."]
            steps_ru = [f"Отношение скоростей: {v2} / {v1} = {k}."]
            exact_params = {"kind": "ratio", "v1": v1, "v2": v2, "expected": k}
        d = PairDraft(SUBJECT, "", "", Difficulty.SCHOOL, atype, "", question_en=q_en, question_ru=q_ru)
        return _finish(_emit_exact(d, ans, steps_en, steps_ru, exact_params), "kinem_const", Difficulty.SCHOOL)
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
        q_ru = f"Сколько времени понадобится {agent_ru} при равномерном движении со скоростью {v} m/s, чтобы пройти {s} m?"
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
        k = int(rng.choice([2, 3]))
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
        k = int(rng.choice([2, 3]))
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
            f"Heating {fmt(m)} kg of an unknown metal from {20} °C to {20 + dt} °C required {fmt(q_heat)} J of heat. "
            f"What is the specific heat capacity of the metal?"
        )
        q_ru = (
            f"Для нагревания {fmt(m)} kg неизвестного металла от {20} °C до {20 + dt} °C потребовалось "
            f"{fmt(q_heat)} J теплоты. Чему равна удельная теплоёмкость этого металла?"
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
            f"Boyle's law: p1 * V1 = p2 * V2.",
            f"p2 = {p1} * {v1} / {v2} = {fmt(p2)} kPa.",
        ]
        steps_ru = [
            f"Закон Бойля — Мариотта: p1 * V1 = p2 * V2.",
            f"p2 = {p1} * {v1} / {v2} = {fmt(p2)} kPa.",
        ]
        value, units = float(p2), "kPa"
        extras = [(p1 * v2 / v1 if v1 else 0.0, "inverted_ratio"), (p1 + v2 - v1, "delta_slip"), (p1 * 2, "factor_of_2")]
        params = {"kind": "boyle", "p1": p1, "v1": v1, "v2": v2, "expected": p2}
    else:  # Charles (isobaric)
        t1 = 10 * int(rng.integers(20, 41))  # 200..400 K
        ratio_num, ratio_den = [(3, 2), (4, 3), (5, 4), (2, 1), (3, 1)][int(rng.integers(0, 5))]
        v2 = ratio_den * int(rng.integers(2, 7))
        v1 = ratio_num * (v2 // ratio_den)
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
            f"Charles's law: V1 / T1 = V2 / T2.",
            f"V1 = V2 * T1 / T2 = {v2} * {t1} / {fmt(t2)} = {fmt(v1)} L.",
        ]
        steps_ru = [
            f"Закон Гей-Люссака: V1 / T1 = V2 / T2.",
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
        q_en = (
            f"By what factor does the hydrostatic pressure of water grow if the depth increases "
            f"{k}-fold?"
        )
        q_ru = (
            f"Во сколько раз вырастет гидростатическое давление воды, если глубина погружения "
            f"увеличится в {k} раза?"
        )
        steps_en = [f"P = rho*g*h is proportional to the depth h, so the pressure grows {k}-fold."]
        steps_ru = [f"P = rho*g*h пропорциональна глубине h, поэтому давление вырастет в {k} раза."]
        d = PairDraft(SUBJECT, "", "", Difficulty.UNIVERSITY, atype, "", question_en=q_en, question_ru=q_ru)
        return _finish(_emit_exact(d, str(k), steps_en, steps_ru, {"k": k, "expected": k}), "hydrostatic", Difficulty.UNIVERSITY)
    h = float(rng.choice([2.5, 3, 4, 5, 7.5, 10, 12.5, 15, 20, 25, 30, 40, 50]))
    if bool(rng.integers(0, 2)):
        press = 1000 * G98 * h
        q_en = f"Find the gauge pressure of water at a depth of {fmt(h)} m (rho = 1000 kg/m^3, g = 9.8 m/s^2)."
        q_ru = f"Вычислите давление воды на глубине {fmt(h)} m (rho = 1000 kg/m^3, g = 9.8 m/s^2)."
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
            f"A horizontal port hole of area {fmt(area)} m^2 is located at a depth of {fmt(h)} m in fresh water "
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
            f"1/f = 1/d_o + 1/d_i: the sign of d_i depends on whether d_o is larger than f.",
            f"Here d_o = {d_o} cm and f = {f} cm, so the image is {ans}.",
        ]
        ans_ru = "действительное" if ans == "real" else "мнимое"
        steps_ru = [
            f"1/f = 1/d_o + 1/d_i: знак d_i зависит от того, больше ли d_o, чем f.",
            f"Здесь d_o = {d_o} cm, а f = {f} cm, поэтому изображение {ans_ru}.",
        ]
        d = PairDraft(SUBJECT, "", "", difficulty, atype, "", question_en=q_en, question_ru=q_ru)
        return _finish(
            _emit_exact(d, ans, steps_en, steps_ru, {"f": f, "do": d_o, "expected_text": ans, "kind": "real_virtual"}),
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
            f"Thin-lens equation: 1/f = 1/d_o + 1/d_i.",
            f"d_i = f*d_o/(d_o - f) = {f}*{d_o}/({d_o} - {f}) = {d_i} cm.",
        ]
        steps_ru = [
            f"Формула тонкой линзы: 1/f = 1/d_o + 1/d_i.",
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
    return g_lenses(rng, idx, atype, Difficulty.OLYMPIAD)


# --------------------------------------------------------------------------- #
# Uniform circular motion (olympiad)
# --------------------------------------------------------------------------- #
def g_circular(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    if bool(rng.integers(0, 2)):
        for _ in range(300):
            v = int(rng.integers(2, 21))
            a_num = int(rng.choice([10, 20, 25, 40, 50, 100]))  # a = v^2/r numerator
            if v * v % a_num == 0:
                r = v * v // a_num
                if 2 <= r <= 400:
                    break
        else:
            v, r, a_val = 10, 20, 5.0
        a_val = v * v / r
        q_en = f"A car moves along a circular track of radius {r} m at a constant speed of {v} m/s. What is its centripetal acceleration?"
        q_ru = f"Автомобиль движется по круговой трассе радиусом {r} m с постоянной скоростью {v} m/s. Чему равно его центростремительное ускорение?"
        steps_en = [f"a = v^2 / r = {v}^2 / {r} = {fmt(a_val)} m/s^2."]
        steps_ru = [f"a = v^2 / r = {v}^2 / {r} = {fmt(a_val)} m/s^2."]
        value, units = a_val, "m/s^2"
        extras = [(v / r if r else 0.0, "forgot_square"), (v * v * r, "multiplied_extra"), (2 * v * v / r, "factor_of_2")]
        params = {"v": v, "r": r, "expected": a_val, "kind": "a"}
    else:
        t_per = int(rng.integers(2, 11))
        r = 25 * int(rng.integers(1, 9))
        v = 2 * 3.14 * r / t_per
        q_en = (
            f"A body moves uniformly along a circle of radius {r} m with a period of {t_per} s "
            f"(pi = 3.14). What is its speed?"
        )
        q_ru = (
            f"Тело равномерно движется по окружности радиусом {r} m с периодом обращения {t_per} s "
            f"(pi = 3.14). Чему равна его скорость?"
        )
        steps_en = [
            f"v = 2*pi*r / T = 2 * 3.14 * {r} / {t_per} = {fmt(v)} m/s.",
        ]
        steps_ru = [
            f"v = 2*pi*r / T = 2 * 3.14 * {r} / {t_per} = {fmt(v)} m/s.",
        ]
        value, units = v, "m/s"
        extras = [(3.14 * r / t_per if t_per else 0.0, "forgot_factor_2"), (2 * 3.14 * r * t_per, "multiplied_extra"), (2 * 3.14 * r, "forgot_period")]
        params = {"T": t_per, "r": r, "expected": v, "kind": "v"}
    d = PairDraft(SUBJECT, "", "", Difficulty.OLYMPIAD, atype, "", question_en=q_en, question_ru=q_ru)
    return _finish(_emit(d, rng, atype, value, units, steps_en, steps_ru, extras, params), "circular", Difficulty.OLYMPIAD)


# --------------------------------------------------------------------------- #
# Projectile motion (olympiad)
# --------------------------------------------------------------------------- #
def g_projectile(rng: np.random.Generator, idx: int, atype: AnswerType) -> PairDraft:
    if atype == AnswerType.EXACT:
        q_en = (
            "A ball is thrown at some fixed angle to the horizontal. By what factor does its range "
            "increase if the initial speed is doubled (the angle is unchanged)?"
        )
        q_ru = (
            "Мяч бросают под некоторым фиксированным углом к горизонту. Во сколько раз увеличится "
            "дальность полёта, если начальную скорость удвоить (угол не меняется)?"
        )
        steps_en = [f"R = v0^2 * sin(2*theta) / g is proportional to v0^2, and 2^2 = 4."]
        steps_ru = [f"R = v0^2 * sin(2*theta) / g пропорциональна v0^2, а 2^2 = 4."]
        d = PairDraft(SUBJECT, "", "", Difficulty.OLYMPIAD, atype, "", question_en=q_en, question_ru=q_ru)
        return _finish(_emit_exact(d, "4", steps_en, steps_ru, {"expected": 4}), "projectile", Difficulty.OLYMPIAD)
    theta = int(rng.choice([30, 45, 60]))
    v0 = int(rng.choice([n for n in range(10, 41, 2)]))
    ask = int(rng.integers(0, 2))
    if theta == 45:
        if ask == 0:
            h_max = v0 * v0 / 40
            q_en = (
                f"A ball is thrown from the ground at {theta}° to the horizontal with an initial speed of "
                f"{v0} m/s (g = 10 m/s^2). What is the maximum height it reaches?"
            )
            q_ru = (
                f"Мяч бросают с поверхности земли под углом {theta}° к горизонту с начальной скоростью "
                f"{v0} m/s (g = 10 m/s^2). На какую максимальную высоту он поднимется?"
            )
            steps_en = [
                f"Vertical component: v0y = v0 * sin(45°) = {fmt(v0 * 0.7071)} m/s; h = v0y^2 / (2*g) = {fmt(v0 * v0 / 40)} m.",
            ]
            steps_ru = [
                f"Вертикальная составляющая: v0y = v0 * sin(45°) = {fmt(v0 * 0.7071)} m/s; h = v0y^2 / (2*g) = {fmt(v0 * v0 / 40)} m.",
            ]
            value, units = h_max, "m"
            extras = [(v0 * v0 / 20, "factor_of_2"), (v0 * v0 / 10, "range_confusion"), (v0 * v0 / 80, "factor_of_4_small")]
            params = {"theta": theta, "v0": v0, "ask": "height", "expected": h_max}
        else:
            rng_val = v0 * v0 / 10
            q_en = (
                f"A ball is thrown from the ground at {theta}° to the horizontal with an initial speed of "
                f"{v0} m/s (g = 10 m/s^2). At what distance from the launch point does it land?"
            )
            q_ru = (
                f"Мяч бросают с поверхности земли под углом {theta}° к горизонту с начальной скоростью "
                f"{v0} m/s (g = 10 m/s^2). На каком расстоянии от точки броска он упадёт?"
            )
            steps_en = [
                f"Range: R = v0^2 * sin(2*theta) / g = {v0}^2 * sin(90°) / 10 = {fmt(rng_val)} m.",
            ]
            steps_ru = [
                f"Дальность: R = v0^2 * sin(2*theta) / g = {v0}^2 * sin(90°) / 10 = {fmt(rng_val)} m.",
            ]
            value, units = rng_val, "m"
            extras = [(v0 * v0 / 20, "factor_of_2"), (v0 * v0 / 40, "height_confusion"), (v0 * v0 / 5, "factor_of_2_other")]
            params = {"theta": theta, "v0": v0, "ask": "range", "expected": rng_val}
    elif theta == 30:
        if ask == 0:
            h_max = v0 * v0 / 160
            q_en = (
                f"A stone is thrown from the ground at {theta}° to the horizontal with an initial speed of "
                f"{v0} m/s (g = 10 m/s^2, sin(30°) = 0.5). What is the maximum height it reaches?"
            )
            q_ru = (
                f"Камень бросают с поверхности земли под углом {theta}° к горизонту с начальной скоростью "
                f"{v0} m/s (g = 10 m/s^2, sin(30°) = 0.5). На какую максимальную высоту он поднимется?"
            )
            steps_en = [
                f"v0y = v0 * sin(30°) = {fmt(v0 * 0.5)} m/s; h = v0y^2 / (2*g) = {fmt(h_max)} m.",
            ]
            steps_ru = [
                f"v0y = v0 * sin(30°) = {fmt(v0 * 0.5)} m/s; h = v0y^2 / (2*g) = {fmt(h_max)} m.",
            ]
            value, units = h_max, "m"
            extras = [(v0 * v0 / 40, "angle_confusion_45"), (v0 * v0 / 80, "factor_of_2"), (v0 * v0 / 20, "factor_of_4")]
            params = {"theta": theta, "v0": v0, "ask": "height", "expected": h_max}
        else:
            t_flight = v0 / 10
            q_en = (
                f"A stone is thrown from the ground at {theta}° to the horizontal with an initial speed of "
                f"{v0} m/s (g = 10 m/s^2, sin(30°) = 0.5). How long is it in the air?"
            )
            q_ru = (
                f"Камень бросают с поверхности земли под углом {theta}° к горизонту с начальной скоростью "
                f"{v0} m/s (g = 10 m/s^2, sin(30°) = 0.5). Сколько времени он будет находиться в полёте?"
            )
            steps_en = [
                f"t = 2 * v0 * sin(30°) / g = 2 * {v0} * 0.5 / 10 = {fmt(t_flight)} s.",
            ]
            steps_ru = [
                f"t = 2 * v0 * sin(30°) / g = 2 * {v0} * 0.5 / 10 = {fmt(t_flight)} s.",
            ]
            value, units = t_flight, "s"
            extras = [(v0 / 5, "factor_of_2"), (v0 / 20, "forgot_factor_2"), (v0 / 10 + 1, "off_by_one")]
            params = {"theta": theta, "v0": v0, "ask": "time", "expected": t_flight}
    else:
        h_max = 3 * v0 * v0 / 160
        q_en = (
            f"A ball is thrown from the ground at {theta}° to the horizontal with an initial speed of "
            f"{v0} m/s (g = 10 m/s^2, sin(60°) = 0.866). What is the maximum height it reaches?"
        )
        q_ru = (
            f"Мяч бросают с поверхности земли под углом {theta}° к горизонту с начальной скоростью "
            f"{v0} m/s (g = 10 m/s^2, sin(60°) = 0.866). На какую максимальную высоту он поднимется?"
        )
        steps_en = [
            f"v0y = v0 * sin(60°) = {fmt(v0 * 0.866)} m/s; h = v0y^2 / (2*g) = {fmt(h_max)} m.",
        ]
        steps_ru = [
            f"v0y = v0 * sin(60°) = {fmt(v0 * 0.866)} m/s; h = v0y^2 / (2*g) = {fmt(h_max)} m.",
        ]
        value, units = h_max, "m"
        extras = [(v0 * v0 / 40, "angle_confusion_45"), (3 * v0 * v0 / 80, "factor_of_2"), (v0 * v0 / 160, "angle_confusion_30")]
        params = {"theta": theta, "v0": v0, "ask": "height", "expected": h_max}
    d = PairDraft(SUBJECT, "", "", Difficulty.OLYMPIAD, atype, "", question_en=q_en, question_ru=q_ru)
    return _finish(_emit(d, rng, atype, value, units, steps_en, steps_ru, extras, params), "projectile", Difficulty.OLYMPIAD)


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
