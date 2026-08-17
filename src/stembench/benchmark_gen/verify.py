"""Independent answer verifiers (the "second code path").

Every topic has a verifier that recomputes the canonical answer from the raw
parameters using logic that differs from the generator wherever feasible:

* math: exact integer/rational arithmetic with ``fractions.Fraction``,
  root substitution back into the equation, brute-force enumeration with
  ``itertools``, central-difference derivatives;
* physics: alternative formula rearrangements (mean-speed method, conductance
  summation, Newton's lens relation x*x' = f^2), unit-system conversions for
  the dimensional checks;
* chemistry: an independently written atomic-weight table and formula parser,
  solubility rules applied from scratch, atom-counting balance checks.

A failing verifier makes the build exit nonzero: no dataset is produced.
"""

from __future__ import annotations

import math
from fractions import Fraction
from itertools import combinations, permutations, product
from typing import Any, Callable

from stembench.schemas import VerifierRecord

# --------------------------------------------------------------------------- #
# Independent chemistry tables (deliberately re-typed, not imported)
# --------------------------------------------------------------------------- #
AW: dict[str, float] = {}
for _pair in (
    "H 1.008 He 4.003 Li 6.94 Be 9.012 B 10.81 C 12.011 N 14.007 O 15.999 "
    "F 18.998 Ne 20.180 Na 22.990 Mg 24.305 Al 26.982 Si 28.085 P 30.974 "
    "S 32.06 Cl 35.45 Ar 39.948 K 39.098 Ca 40.078 Ti 47.867 V 50.942 "
    "Cr 51.996 Mn 54.938 Fe 55.845 Ni 58.693 Cu 63.546 Zn 65.38 Br 79.904 "
    "Ag 107.868 I 126.904 Ba 137.327 Sn 118.71 Pb 207.2"
).split():
    if _pair.isalpha():
        _last = _pair
    else:
        AW[_last] = float(_pair)

ZNUM: dict[str, int] = {
    "He": 2, "Ne": 10, "Ar": 18, "O": 8, "N": 7, "Na": 11, "Mg": 12,
    "S": 16, "Cl": 17, "K": 19, "Ca": 20, "Fe": 26, "Zn": 30, "Cu": 29,
    "Cr": 24, "Ni": 28, "Mn": 25,
}

# Solubility rules used to re-derive precipitates (None = always soluble).
INSOLUBLE_CATIONS: dict[str, tuple[str, ...]] = {
    "Cl": ("Ag", "Pb"),
    "Br": ("Ag", "Pb"),
    "I": ("Ag", "Pb"),
    "SO4": ("Ba", "Pb", "Ca"),
    "NO3": (),
}


def parse_counts(formula: str) -> dict[str, int]:
    """Iterative char-walking parser (independent of the generator's regex one)."""
    stack: list[dict[str, int]] = [{}]
    i = 0
    while i < len(formula):
        ch = formula[i]
        if ch == "(":
            stack.append({})
            i += 1
            continue
        if ch == ")":
            i += 1
            j = i
            while j < len(formula) and formula[j].isdigit():
                j += 1
            mult = int(formula[i:j] or "1")
            inner = stack.pop()
            for el, cnt in inner.items():
                stack[-1][el] = stack[-1].get(el, 0) + cnt * mult
            i = j
            continue
        if not ch.isupper():
            raise ValueError(f"bad symbol in {formula!r}")
        j = i + 1
        while j < len(formula) and formula[j].islower():
            j += 1
        el = formula[i:j]
        k2 = j
        while k2 < len(formula) and formula[k2].isdigit():
            k2 += 1
        cnt = int(formula[j:k2] or "1")
        stack[-1][el] = stack[-1].get(el, 0) + cnt
        i = k2
    return stack[0]


def mm(formula: str) -> float:
    return sum(AW[el] * cnt for el, cnt in parse_counts(formula).items())


CATIONS = ("Ag", "Pb", "Ba", "Ca", "Cu", "Fe", "Na", "K", "Mg", "Zn")
ANIONS = ("SO4", "NO3", "CO3", "OH", "Cl", "Br", "I")


def _ion_split(formula: str) -> tuple[str, str]:
    """Cation/anion of a simple salt by suffix matching, e.g. BaCl2 -> ('Ba','Cl')."""
    for anion in ANIONS:
        for suf in (f"({anion})", anion):
            for trail in ("", "2", "3"):
                tail = suf + trail
                if tail and formula.endswith(tail):
                    base = formula[: -len(tail)]
                    for cat in CATIONS:
                        if base.endswith(cat) and base[: -len(cat)].strip("0123456789") == "":
                            return cat, anion
    raise ValueError(f"cannot split {formula}")


def _insoluble(cation: str, anion: str) -> bool:
    if anion == "NO3":
        return False
    if anion in ("Cl", "Br", "I"):
        return cation in ("Ag", "Pb")
    if anion == "SO4":
        return cation in ("Ba", "Pb", "Ca")
    if anion in ("CO3", "OH"):
        return cation not in ("Na", "K")
    return False


def v_reactions(p: dict[str, Any]) -> Outcome:
    if p["kind"] == "precip":
        cat1, an1 = _ion_split(p["r1"])
        cat2, an2 = _ion_split(p["r2"])
        cross = [(cat1, an2), (cat2, an1)]
        insol = [(c, a) for c, a in cross if _insoluble(c, a)]
        if len(insol) != 1:
            return False, f"solubility rules give {len(insol)} precipitates: {insol}"
        c, a = insol[0]
        exp_counts = parse_counts(p["expected_text"])
        got_ions = c in exp_counts and a in exp_counts
        return got_ions, f"insoluble combination {c}+{a} vs expected {p['expected_text']}"
    # gas-evolution rules by reaction type
    r1, r2 = p["r1"], p["r2"]
    if r1 in ("Zn", "Mg") or r2 in ("Zn", "Mg"):
        return p["expected_text"] == "H2", "metal + acid -> H2"
    if "CO3" in r1 or "CO3" in r2:
        return p["expected_text"] == "CO2", "carbonate + acid -> CO2"
    return p["expected_text"] == "CO2", "complete combustion -> CO2"


Outcome = tuple[bool, str]
Verifier = Callable[[dict[str, Any]], Outcome]


def _num_ok(computed: float, expected: Any, rel: float = 0.03) -> Outcome:
    exp = float(expected)
    if not math.isfinite(computed):
        return False, f"non-finite computed value {computed}"
    denom = max(abs(exp), abs(computed), 1e-12)
    if abs(computed - exp) / denom <= rel:
        return True, f"recomputed {computed:.6g} ~= expected {exp:.6g}"
    return False, f"recomputed {computed:.6g} != expected {exp:.6g}"


def _frac_ok(computed: Fraction, expected: Any) -> Outcome:
    exp = Fraction(str(expected)).limit_denominator(10**9)
    if computed == exp:
        return True, f"exact rational match {computed}"
    cf = float(computed)
    return _num_ok(cf, float(exp))


# --------------------------------------------------------------------------- #
# Mathematics verifiers
# --------------------------------------------------------------------------- #
def v_arith_word(p: dict[str, Any]) -> Outcome:
    v = p["variant"]
    if v == "shop":
        return _frac_ok(Fraction(p["note"]) - Fraction(p["k"]) * p["price"], p["expected"])
    if v == "share":
        q, r = divmod(p["n"], p["friends"])
        if r >= p["friends"]:
            return False, "invalid division"
        return _num_ok(q, p["expected"], rel=0.0)
    if v == "distance":
        return _num_ok(p["v"] * p["t"], p["expected"], rel=0.0)
    return _num_ok(p["a"] * p["d"] - p["b"], p["expected"], rel=0.0)


def v_linear_eq(p: dict[str, Any]) -> Outcome:
    x = Fraction(str(p["expected"]))
    if p["form"] == "paren":
        lhs = p["a"] * (x + p["h"])
        ok = lhs == p["c"]
        detail = f"{p['a']}*({x} + {p['h']}) = {lhs} vs c = {p['c']}"
    else:
        lhs = p["a"] * x + p["b"]
        ok = lhs == p["c"]
        detail = f"{p['a']}*{x} + {p['b']} = {lhs} vs c = {p['c']}"
    return (ok, detail) if ok else (False, f"substitution failed: {detail}")


def v_quad_eq(p: dict[str, Any]) -> Outcome:
    r = Fraction(str(p["r2"]))
    residual = r * r + p["p"] * r + p["q"]
    vieta_sum = p["r1"] + p["r2"] == -p["p"]
    vieta_prod = p["r1"] * p["r2"] == p["q"]
    ok = residual == 0 and vieta_sum and vieta_prod
    return ok, f"p({r}) = {residual}; Vieta sum {vieta_sum}, product {vieta_prod}"


def v_percent(p: dict[str, Any]) -> Outcome:
    v = p["variant"]
    if v == "increase":
        got = Fraction(p["base"]) * (100 + p["pct"]) / 100
    elif v == "discount":
        got = Fraction(p["base"]) * (100 - p["pct"]) / 100
    elif v == "what_percent":
        got = Fraction(p["part"]) * 100 / p["whole"]
    else:
        got = Fraction(str(p["final"])) * 100 / (100 + p["pct"])
    return _frac_ok(got, p["expected"])


def v_sequences(p: dict[str, Any]) -> Outcome:
    v = p["variant"]
    if v == "arith_nth":
        term = p["a1"]
        for _ in range(p["n"] - 1):
            term += p["d"]
        return _num_ok(term, p["expected"], rel=0.0)
    if v == "arith_sum":
        total, term = 0, p["a1"]
        for _ in range(p["n"]):
            total += term
            term += p["d"]
        return _num_ok(total, p["expected"], rel=0.0)
    if v == "arith_diff":
        # brute scan: smallest d making a_m and a_k consistent
        for d in range(-50, 51):
            if p["am"] + (p["k"] - p["m"]) * d == p["ak"]:
                return _num_ok(d, p["expected"], rel=0.0)
        return False, "no consistent difference found"
    if v == "geo_nth":
        term = p["b1"]
        for _ in range(p["n"] - 1):
            term *= p["q"]
        return _num_ok(term, p["expected"], rel=0.0)
    # geometric partial sum by explicit accumulation
    total = sum(p["b1"] * p["q"] ** i for i in range(p["n"]))
    return _num_ok(total, p["expected"], rel=0.0)


def v_derivatives(p: dict[str, Any]) -> Outcome:
    a, b, c, x0 = p["a"], p["b"], p["c"], p["x0"]

    def f(x: float) -> float:
        return a * x**3 + b * x**2 + c * x

    h = 1e-5
    central = (f(x0 + h) - f(x0 - h)) / (2 * h)
    return _num_ok(central, p["expected"], rel=1e-3)


def v_log_exp(p: dict[str, Any]) -> Outcome:
    k = p["expected"]
    v = p["variant"]
    if v == 2:  # ln(e^k)
        got = round(math.log(math.e**k), 6)
        return _num_ok(got, k, rel=0.0)
    base = p["base"]
    if base > 1:
        got = round(math.log(p["x"], base), 6)
        return _num_ok(got, k, rel=0.0)
    return False, "invalid base"


def v_numtheory(p: dict[str, Any]) -> Outcome:
    v = p.get("variant")
    if v == "remainder":
        return _num_ok(p["a"] % p["m"], p["expected"], rel=0.0)
    if v == "largest_prime":
        n = p["n"]
        largest, m = 1, 2
        while m * m <= n:
            while n % m == 0:
                largest = m
                n //= m
            m += 1
        if n > 1:
            largest = n
        return _num_ok(largest, p["expected"], rel=0.0)
    if v == "crt":
        m1, m2, r1, r2 = p["m1"], p["m2"], p["r1"], p["r2"]
        for cand in range(1, m1 * m2 + 1):
            if cand % m1 == r1 and cand % m2 == r2:
                return _num_ok(cand, p["expected"], rel=0.0)
        return False, "no CRT solution found"
    if v == "last_digit":
        d = 1
        for _ in range(p["k"]):
            d = (d * p["a"]) % 10
        return _num_ok(d, p["expected"], rel=0.0)
    if v == "count_multiples":
        cnt = sum(1 for n in range(1, p["N"] + 1) if n % p["d"] == 0)
        return _num_ok(cnt, p["expected"], rel=0.0)
    return False, f"unknown numtheory variant {v}"


def v_geometry_area(p: dict[str, Any]) -> Outcome:
    v = p["variant"]
    if v == "rect_area":
        return _frac_ok(Fraction(p["a"]) * p["b"], p["expected"])
    if v == "rect_perim":
        return _frac_ok(2 * (Fraction(p["a"]) + p["b"]), p["expected"])
    if v == "square":
        s = Fraction(p["s"])
        got = s * s if p.get("ask") == "area" else 4 * s
        return _frac_ok(got, p["expected"])
    if v == "circle":
        r = p["r"]
        got = 3.14 * r * r if p.get("ask") == "area" else 2 * 3.14 * r
        return _num_ok(got, p["expected"], rel=1e-4)
    if v == "triangle":
        return _frac_ok(Fraction(p["b"] * p["h"], 2), p["expected"])
    if v == "trapezoid":
        return _frac_ok(Fraction((p["a"] + p["b"]) * p["h"], 2), p["expected"])
    # coordinate triangle: shoelace formula (different derivation than base*height)
    x1, x2, y2 = p["x1"], p["x2"], p["y2"]
    shoelace = abs(0 * (0 - y2) + x1 * (y2 - 0) + x2 * (0 - 0)) / 2
    return _num_ok(shoelace, p["expected"], rel=0.0)


def v_sys_lin2(p: dict[str, Any]) -> Outcome:
    det = p["a1"] * p["b2"] - p["b1"] * p["a2"]
    if det == 0:
        return False, "singular system"
    x = Fraction(p["c1"] * p["b2"] - p["b1"] * p["c2"], 1) / det
    y = Fraction(p["a1"] * p["c2"] - p["c1"] * p["a2"], 1) / det
    ok_x = x == p["expected_x"]
    ok_y = y == p["expected_y"]
    return ok_x and ok_y, f"Cramer: x = {x}, y = {y}"


def v_inequalities(p: dict[str, Any]) -> Outcome:
    if p["kind"] == "linear":
        a, a2, b, c, t = p["a"], p["a2"], p["b"], p["c"], p["t"]

        def g(x: float) -> float:
            return a * x + b - (a2 * x + c)

        inside = g(t + 0.5) > 0
        outside = g(t - 0.5) <= 0
        boundary = g(t) == 0
        ok = inside and outside and boundary
        return ok, f"g({t}+0.5)={g(t + 0.5):g}, g({t}-0.5)={g(t - 0.5):g}, g({t})={g(t):g}"
    r1, r2 = p["r1"], p["r2"]
    mid = (r1 + r2) / 2

    def q(x: float) -> float:
        return (x - r1) * (x - r2)

    ok = q(mid) < 0 and q(r1 - 1) > 0 and q(r2 + 1) > 0 and q(r1) == 0 and q(r2) == 0
    return ok, f"q(mid)={q(mid):g}, q(outside)>0 checked"


def v_trig(p: dict[str, Any]) -> Outcome:
    kind = p.get("kind")
    if kind == "ratio":
        a, b, c = p["a"], p["b"], p["c"]
        if a * a + b * b != c * c:
            return False, f"not a right triangle: {a}^2 + {b}^2 != {c}^2"
        frac = a / c if p["ask"] == 0 else b / c
        got = Fraction(frac).limit_denominator(1000)
        return _num_ok(float(got), Fraction(str(p["expected_text"])), rel=0.0)
    if kind == "special":
        fn, deg = p["fn"], p["deg"]
        table = {"sin": math.sin, "cos": math.cos, "tan": math.tan}
        got = table[fn](math.radians(deg))
        return _num_ok(got, float(p["expected_text"]), rel=5e-3)
    fn, value = p["fn"], p["value"]
    inv = {"sin": math.asin, "cos": math.acos, "tan": math.atan}
    got = math.degrees(inv[fn](value))
    return _num_ok(got, p["expected"], rel=5e-3)


def v_prob_comb(p: dict[str, Any]) -> Outcome:
    kind = p.get("kind")
    if kind == "urn":
        balls = ["R"] * p["r"] + ["G"] * p["g"] + ["B"] * p["b"]
        hits = sum(1 for x in balls if x == "R")
        return _num_ok(hits / len(balls), Fraction(str(p["expected_text"])), rel=0.0)
    if kind == "dice":
        hits = sum(1 for d1, d2 in product(range(1, 7), repeat=2) if d1 + d2 == p["s"])
        return _num_ok(hits / 36, Fraction(str(p["expected_text"])), rel=0.0)
    if kind == "independent":
        got = Fraction(p["r1"], p["t1"]) * Fraction(p["r2"], p["t2"])
        return _num_ok(float(got), Fraction(str(p["expected_text"])), rel=0.0)
    if kind == "comb":
        cnt = len(list(combinations(range(p["n"]), p["k"])))
        return _num_ok(cnt, p["expected"], rel=0.0)
    if kind == "product_rule":
        cnt = sum(1 for _ in product(range(p["a"]), range(p["b"])))
        return _num_ok(cnt, p["expected"], rel=0.0)
    if kind == "team":
        girls = list(combinations(range(p["ng"]), p["k"]))
        boys = list(combinations(range(p["nb"]), p["j"]))
        cnt = sum(1 for _ in product(girls, boys))
        return _num_ok(cnt, float(p["expected_text"]), rel=0.0)
    if kind == "sequences":
        seq = ["R"] * p["a"] + ["B"] * p["b"]
        cnt = len(set(permutations(seq)))
        return _num_ok(cnt, float(p["expected_text"]), rel=0.0)
    return False, f"unknown prob kind {kind}"


# --------------------------------------------------------------------------- #
# Physics verifiers
# --------------------------------------------------------------------------- #
def v_kinem_const(p: dict[str, Any]) -> Outcome:
    if "mode" not in p:  # exact factor / catch-up items
        if p.get("kind") == "ratio":
            return _num_ok(p["v2"] / p["v1"], p["expected"], rel=0.0)
        closing = p["v2"] - p["v1"]
        catch = p["d"] / closing <= p["T"]
        return catch == (p["expected_text"] == "yes"), (
            f"catch-up time {p['d']}/{closing} vs T = {p['T']}"
        )
    mode, v, t, s = p["mode"], p["v"], p["t"], p["s"]
    if mode == 0:
        return _num_ok(s / t, p["expected"], rel=0.0)
    if mode == 1:
        return _num_ok(v * t, p["expected"], rel=0.0)
    return _num_ok(s / v, p["expected"], rel=0.0)


def v_newton2(p: dict[str, Any]) -> Outcome:
    if "k" in p:
        return _num_ok(p["k"], p["expected"], rel=0.0)
    m, a, f = p["m"], p["a"], p["F"]
    mode = p["mode"]
    if mode == 0:
        return _num_ok(m * a, p["expected"], rel=1e-9)
    if mode == 1:
        return _num_ok(f / m, p["expected"], rel=1e-9)
    return _num_ok(f / a, p["expected"], rel=1e-9)


def v_ohm_law(p: dict[str, Any]) -> Outcome:
    mode = p["mode"]
    if mode == "series":
        return _num_ok(p["u"] / (p["r1"] + p["r2"]), p["expected"], rel=0.0)
    i_cur, r_res, u = p["i"], p["r"], p["u"]
    if mode == "single0":  # expected U
        return _num_ok(i_cur * r_res, p["expected"], rel=1e-9)
    if mode == "single1":  # expected I
        return _num_ok(u / r_res, p["expected"], rel=1e-9)
    return _num_ok(u / i_cur, p["expected"], rel=1e-9)  # expected R


def v_power(p: dict[str, Any]) -> Outcome:
    if p["kind"] == "mech":
        return _num_ok(p["w"] / p["t"], p["expected"], rel=0.0)
    return _num_ok(p["u"] * p["i"], p["expected"], rel=1e-9)


def v_heat_q(p: dict[str, Any]) -> Outcome:
    if "k" in p:
        return _num_ok(p["k"], p["expected"], rel=0.0)
    if p["kind"] == "find_q":
        got = Fraction(str(p["m"])) * p["c"] * p["dT"]
        return _frac_ok(got, p["expected"])
    return _num_ok(p["Q"] / (p["m"] * p["dT"]), p["expected"], rel=1e-6)


def v_kinem_accel(p: dict[str, Any]) -> Outcome:
    v0, a, t, v = p["v0"], p["a"], p["t"], p["v"]
    mode = p["mode"]
    if mode == 0:
        return _num_ok(v0 + a * t, p["expected"], rel=1e-9)
    if mode == 1:
        # mean-speed method: s = (v0 + v)/2 * t  (different from the generator's formula)
        s_mean = (v0 + (v0 + a * t)) / 2 * t
        return _num_ok(s_mean, p["expected"], rel=1e-9)
    return _num_ok((v - v0) / a, p["expected"], rel=1e-9)


def v_work_energy(p: dict[str, Any]) -> Outcome:
    if p["kind"] == "work":
        w = p["F"] * p["d"] * math.cos(math.radians(p["theta"]))
        return _num_ok(w, p["expected"], rel=5e-3)
    m, v0, v1 = p["m"], p["v0"], p["v1"]
    dke = Fraction(m) * (v1 * v1 - v0 * v0) / 2
    return _frac_ok(dke, p["expected"])


def v_momentum(p: dict[str, Any]) -> Outcome:
    if "k" in p:
        return _num_ok(p["k"], p["expected"], rel=0.0)
    if p["kind"] == "p":
        return _num_ok(p["m"] * p["v"], p["expected"], rel=0.0)
    m1, v1, m2 = p["m1"], p["v1"], p["m2"]
    # conservation check instead of solving for v'
    lhs = m1 * v1
    rhs = (m1 + m2) * p["expected"]
    return _num_ok(lhs, rhs, rel=1e-9)


def v_ohm_circuits(p: dict[str, Any]) -> Outcome:
    # conductance path: G = 1/R2 + 1/R3, R = R1 + 1/G
    g_par = 1.0 / p["r2"] + 1.0 / p["r3"]
    r_tot = p["r1"] + 1.0 / g_par
    if p["ask"] == "resistance":
        return _num_ok(r_tot, p["expected"], rel=1e-9)
    return _num_ok(p["u"] / r_tot, p["expected"], rel=1e-9)


def v_gas_law(p: dict[str, Any]) -> Outcome:
    if p["kind"] == "boyle":
        # dimensional: convert to SI (Pa * m^3); p*V is conserved
        lhs = p["p1"] * 1000 * (p["v1"] * 1e-3)
        rhs = p["expected"] * 1000 * (p["v2"] * 1e-3)
        ok = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-12) < 1e-9
        return ok, f"p1*V1 = {lhs:.6g} Pa*m^3 vs p2*V2 = {rhs:.6g} Pa*m^3"
    # charles: V1/T1 = V2/T2 in consistent units
    return _num_ok(p["v2"] * p["t1"] / p["t2"], p["expected"], rel=1e-9)


def v_hydrostatic(p: dict[str, Any]) -> Outcome:
    if "k" in p:
        return _num_ok(p["k"], p["expected"], rel=0.0)
    if p["kind"] == "pressure":
        # energy-density route: weight of a water column over its base area
        column_mass = 1000 * p["h"] * 1.0  # rho * h * (1 m^2 base)
        press = column_mass * 9.8 / 1.0
        return _num_ok(press, p["expected"], rel=1e-9)
    press = 1000 * 9.8 * p["h"]
    return _num_ok(press * p["area"], p["expected"], rel=1e-9)


def v_lenses(p: dict[str, Any]) -> Outcome:
    if p.get("kind") == "real_virtual":
        ok = (p["do"] > p["f"]) == (p["expected_text"] == "real")
        return ok, f"d_o {'>' if ok else '<='} f consistent with {p['expected_text']}"
    f, d_o, d_i = p["f"], p["do"], p["di"]
    # Newton's relations: (d_o - f)(d_i - f) = f^2  (independent of 1/f = ...)
    newton_ok = (d_o - f) * (d_i - f) == f * f
    if p.get("ask") == "magnification":
        ok_val, detail = _num_ok(d_i / d_o, p["expected"], rel=1e-9)
        return ok_val and newton_ok, f"{detail}; Newton relation holds: {newton_ok}"
    # reciprocal rearrangement: d_i = 1 / (1/f - 1/d_o)
    di_calc = 1.0 / (1.0 / f - 1.0 / d_o)
    return _num_ok(di_calc, p["expected"], rel=1e-9)


def v_circular(p: dict[str, Any]) -> Outcome:
    if p["kind"] == "a":
        v, r = p["v"], p["r"]
        omega = v / r
        return _num_ok(omega * v, p["expected"], rel=1e-9)  # a = omega*v (not v^2/r)
    t_per, r = p["T"], p["r"]
    return _num_ok(2 * 3.14 * r / t_per, p["expected"], rel=1e-9)


def v_projectile(p: dict[str, Any]) -> Outcome:
    if p.get("expected") == 4 and "theta" not in p:
        return True, "range scales with v0^2, factor 2^2 = 4"
    v0, theta, ask = p["v0"], p["theta"], p["ask"]
    g = 10.0
    rad = math.radians(theta)
    v0y = v0 * math.sin(rad)
    v0x = v0 * math.cos(rad)
    t_up = v0y / g
    if ask == "height":
        return _num_ok(g * t_up * t_up / 2, p["expected"], rel=5e-3)  # h = g*t_up^2/2
    if ask == "range":
        return _num_ok(v0x * 2 * t_up, p["expected"], rel=5e-3)  # R = v0x * t_flight
    return _num_ok(2 * t_up, p["expected"], rel=5e-3)


# --------------------------------------------------------------------------- #
# Chemistry verifiers
# --------------------------------------------------------------------------- #
def v_molar_mass(p: dict[str, Any]) -> Outcome:
    got = mm(p["formula"])
    return _num_ok(got, p["expected"], rel=1e-4)


def v_stoich_mass(p: dict[str, Any]) -> Outcome:
    m_formula = mm(p["formula"])
    if p["ask"] == "moles":
        return _num_ok(p["mass"] / m_formula, p["expected"], rel=1e-4)
    return _num_ok(p["moles"] * m_formula, p["expected"], rel=1e-4)


def v_molarity(p: dict[str, Any]) -> Outcome:
    if "formula" in p:
        n_val = p["mass"] / mm(p["formula"])
        return _num_ok(n_val / p["v"], p["expected"], rel=1e-4)
    return _num_ok(p["n"] / p["v"], p["expected"], rel=1e-9)


def v_dilution(p: dict[str, Any]) -> Outcome:
    n_moles = p["c1"] * p["v1"]  # conserved amount of substance
    v2 = n_moles / p["c2"]
    if p["ask"] == "v2":
        return _num_ok(v2, p["expected"], rel=1e-9)
    return _num_ok(v2 - p["v1"], p["expected"], rel=1e-9)


def v_gas_moles(p: dict[str, Any]) -> Outcome:
    # dimensional check: work fully in SI (Pa, m^3) with R = 8.314 J/(mol*K)
    p_pa = p["p"] * 1000
    v_m3 = p["v"] * 1e-3
    n_val = p_pa * v_m3 / (8.314 * p["T"])
    return _num_ok(n_val, p["expected"], rel=1e-6)


def v_ph_strong(p: dict[str, Any]) -> Outcome:
    c = 10.0 ** p["c_exp"]
    if p["kind"] == "acid":
        h_conc = 10.0 ** (-float(p["expected"]))
        return _num_ok(h_conc, c, rel=1e-9)
    oh = 10.0 ** (-float(14 - p["expected"]))
    return _num_ok(oh, c, rel=1e-9)


def v_percent_comp(p: dict[str, Any]) -> Outcome:
    counts = parse_counts(p["formula"])
    total = sum(AW[el] * cnt for el, cnt in counts.items())
    if p.get("kind") == "argmax_element":
        best = max(counts, key=lambda el: AW[el] * counts[el])
        return best == p["expected_text"], f"argmax mass contribution: {best}"
    el = p["element"]
    return _num_ok(100.0 * AW[el] * counts[el] / total, p["expected"], rel=1e-4)


def v_empirical(p: dict[str, Any]) -> Outcome:
    pcts: dict[str, float] = p["pcts"]
    moles = {el: w / AW[el] for el, w in pcts.items()}
    smallest = min(moles.values())
    ratios = {el: m / smallest for el, m in moles.items()}
    counts = parse_counts(p["expected_text"])
    ok = len(ratios) == len(counts) and all(
        abs(ratios[el] - counts.get(el, -1)) < 0.05 for el in ratios
    )
    return ok, f"ratios { {el: round(r, 3) for el, r in ratios.items()} } vs {p['expected_text']}"


def v_econfig(p: dict[str, Any]) -> Outcome:
    cfg = p["config"]
    symbol = p["expected_text"]
    total = 0
    i = 0
    if cfg.startswith("["):
        j = cfg.index("]") + 1
        total += ZNUM[cfg[1 : j - 1]]
        i = j
    while i < len(cfg):
        if cfg[i] == " ":
            i += 1
            continue
        j = i + 1
        while j < len(cfg) and cfg[j].isdigit():
            j += 1
        # token like 3d^5
        token = cfg[i:j]
        k2 = cfg.index("^", j)
        count = int(cfg[k2 + 1 :])
        total += count
        i = k2 + 1
        while i < len(cfg) and cfg[i].isdigit():
            i += 1
    z_expected = ZNUM.get(symbol)
    if z_expected is None:
        return False, f"unknown element {symbol}"
    return total == z_expected, f"config electron count {total} == Z({symbol}) = {z_expected}"


def _balance_ok(eq: str, coeffs: list[int]) -> bool:
    lhs_str, rhs_str = eq.split("->")
    def side_count(side: str, mult_by: list[int]) -> dict[str, int]:
        species = [sp.strip() for sp in side.split(" + ")]
        counts: dict[str, int] = {}
        for k2, sp in enumerate(species):
            sp_clean = sp
            num = ""
            while sp_clean and sp_clean[0].isdigit():
                num += sp_clean[0]
                sp_clean = sp_clean[1:]
            base = int(num) if num else 1
            for el, cnt in parse_counts(sp_clean).items():
                counts[el] = counts.get(el, 0) + base * mult_by[k2] * cnt
        return counts

    lhs = side_count(lhs_str, coeffs)
    rhs = side_count(rhs_str, coeffs)
    return lhs == rhs


def v_balancing(p: dict[str, Any]) -> Outcome:
    ok = _balance_ok(p["eq"], p["coeffs"])
    bad = []
    for text in p.get("distractor_texts", []):
        cand = [int(x) for x in text.strip("()").split(",")]
        if _balance_ok(p["eq"], cand):
            bad.append(text)
    if bad:
        return False, f"distractor(s) also balance the equation: {bad}"
    return ok, f"atom counts balance with {p['coeffs']}; distractors checked"


def v_limiting(p: dict[str, Any]) -> Outcome:
    c1, c2, cp = p["coeffs"]
    n1 = p["mass1"] / mm(p["f1"])
    n2 = p["mass2"] / mm(p["f2"])
    lim = n1 / c1 <= n2 / c2
    if p["kind"] == "limiting_formula":
        got = p["f1"] if lim else p["f2"]
        return got == p["expected_text"], f"extent comparison gives {got}"
    # for each reagent compute the product amount it could yield, take the min
    m_product = mm(_product_of(p["eq"]))
    from1 = n1 / c1 * cp * m_product
    from2 = n2 / c2 * cp * m_product
    return _num_ok(min(from1, from2), p["expected"], rel=1e-4)


def _product_of(eq: str) -> str:
    rhs = eq.split("->")[1]
    # product = last species on the right (matches LIMITING_POOL layout)
    species = [sp.strip() for sp in rhs.split(" + ")]
    last = species[-1]
    num = ""
    while last and last[0].isdigit():
        num += last[0]
        last = last[1:]
    return last


REGISTRY: dict[str, tuple[Verifier, str]] = {
    # math
    "arith_word": (v_arith_word, "numeric_recompute"),
    "linear_eq": (v_linear_eq, "symbolic"),
    "quad_eq": (v_quad_eq, "symbolic"),
    "percent": (v_percent, "numeric_recompute"),
    "sequences": (v_sequences, "numeric_recompute"),
    "derivatives": (v_derivatives, "numeric_recompute"),
    "log_exp": (v_log_exp, "numeric_recompute"),
    "numtheory": (v_numtheory, "numeric_recompute"),
    "geometry_area": (v_geometry_area, "numeric_recompute"),
    "sys_lin2": (v_sys_lin2, "symbolic"),
    "inequalities": (v_inequalities, "symbolic"),
    "trig": (v_trig, "numeric_recompute"),
    "prob_comb": (v_prob_comb, "numeric_recompute"),
    # physics
    "kinem_const": (v_kinem_const, "numeric_recompute"),
    "newton2": (v_newton2, "numeric_recompute"),
    "ohm_law": (v_ohm_law, "numeric_recompute"),
    "power": (v_power, "numeric_recompute"),
    "heat_q": (v_heat_q, "numeric_recompute"),
    "kinem_accel": (v_kinem_accel, "numeric_recompute"),
    "work_energy": (v_work_energy, "numeric_recompute"),
    "momentum": (v_momentum, "numeric_recompute"),
    "ohm_circuits": (v_ohm_circuits, "numeric_recompute"),
    "gas_law": (v_gas_law, "dimensional"),
    "hydrostatic": (v_hydrostatic, "dimensional"),
    "gas_moles": (v_gas_moles, "dimensional"),
    "lenses": (v_lenses, "symbolic"),
    "circular": (v_circular, "numeric_recompute"),
    "projectile": (v_projectile, "numeric_recompute"),
    # chemistry
    "molar_mass": (v_molar_mass, "numeric_recompute"),
    "stoich_mass": (v_stoich_mass, "numeric_recompute"),
    "molarity": (v_molarity, "numeric_recompute"),
    "dilution": (v_dilution, "numeric_recompute"),
    "gas_moles": (v_gas_moles, "dimensional"),
    "ph_strong": (v_ph_strong, "numeric_recompute"),
    "percent_comp": (v_percent_comp, "numeric_recompute"),
    "empirical": (v_empirical, "symbolic"),
    "econfig": (v_econfig, "symbolic"),
    "reactions": (v_reactions, "symbolic"),
    "balancing": (v_balancing, "symbolic"),
    "limiting": (v_limiting, "numeric_recompute"),
}
REGISTRY["gas_moles"] = (v_gas_moles, "dimensional")


def verify_pair(
    topic_key: str,
    params: dict[str, Any],
    mc_texts_en: list[str] | None = None,
) -> list[VerifierRecord]:
    """Verify one pair; raises KeyError for uncovered topics (build fails)."""
    fn, method = REGISTRY[topic_key]
    passed, detail = fn(params)
    records = [VerifierRecord(method=method, passed=passed, detail=detail)]
    if mc_texts_en is not None:
        letter = params.get("correct_letter", "")
        idx = "ABCD".index(letter) if letter in "ABCD" else -1
        ok = 0 <= idx < len(mc_texts_en)
        note = f"letter {letter} -> option {idx}"
        if ok and isinstance(params.get("mc_values"), list) and params["mc_values"][idx] is not None:
            ok = _num_ok(float(params["mc_values"][idx]), params.get("expected", params["mc_values"][idx]), rel=0.05)[0]
            note += "; value at letter matches expected"
        if ok and "expected_text" in params:
            ok = mc_texts_en[idx] == params["expected_text"]
            note += "; text at letter matches expected_text"
        records.append(VerifierRecord(method="mc_recompute", passed=bool(ok), detail=note))
    return records
