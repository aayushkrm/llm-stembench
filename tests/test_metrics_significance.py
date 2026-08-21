"""Tests for stembench.metrics.significance with hand-computed fixtures."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from stembench.metrics.significance import (
    benjamini_hochberg,
    chi_square_independence,
    cochran_q,
    holm_bonferroni,
    mcnemar_test,
)


def _pair_counts(a, b):
    a, b = np.asarray(a), np.asarray(b)
    b01 = int(((a == 1) & (b == 0)).sum())  # A correct, B wrong
    b10 = int(((a == 0) & (b == 1)).sum())  # A wrong, B correct
    return b01, b10


def test_mcnemar_discordant_counts():
    # a = 12 ones then 3 zeros; b = 8 ones then 7 zeros (n = 15, hand-counted):
    #   idx 0..7 : (1,1) both right
    #   idx 8..11: (1,0) -> A right, B wrong  => b01 = 4
    #   idx 12..14: (0,0) both wrong          => b10 = 0
    a = np.array([1] * 12 + [0] * 3)
    b = np.array([1] * 8 + [0] * 7)
    assert _pair_counts(a, b) == (4, 0)
    r = mcnemar_test(a, b)
    assert r.b == 4
    assert r.c == 0
    # continuity-corrected chi2 = (|b01 - b10| - 1)^2 / (b01 + b10) = (4-1)^2/4 = 2.25
    assert r.statistic == pytest.approx(2.25)
    # p_chi2 = P(chi2_1 >= 2.25)
    assert r.p_chi2 == pytest.approx(stats.chi2.sf(2.25, df=1))


def test_mcnemar_second_fixture():
    # a = 12 ones then 3 zeros; b = 8 ones then [0,0,0,0,0,1,0] (hand-counted):
    #   idx 8..11: (1,0) -> b01 = 4
    #   idx 12,14: (0,0) ; idx 13: (0,1) -> b10 = 1
    a = np.array([1] * 12 + [0] * 3)
    b = np.array([1] * 8 + [0, 0, 0, 0, 0, 1, 0])
    assert _pair_counts(a, b) == (4, 1)
    r = mcnemar_test(a, b)
    assert (r.b, r.c) == (4, 1)
    # chi2 = (|4-1| - 1)^2 / (4+1) = 4/5 = 0.8
    assert r.statistic == pytest.approx(0.8)
    # p_exact per the implemented formula = 2 * binomtest(min(b,c), b+c, 0.5).pvalue.
    # min = 1, n = 5: P(X<=1) = (1+5)/32 = 0.1875, scipy two-sided p = 0.375,
    # doubled = 0.75
    assert r.p_exact == pytest.approx(0.75)


@pytest.mark.parametrize(
    "a,b",
    [
        (np.array([1] * 12 + [0] * 3), np.array([1] * 8 + [0] * 7)),
        (np.array([1] * 8 + [0] * 7), np.array([1] * 12 + [0] * 3)),
        (np.array([1, 0, 1, 0, 1, 1]), np.array([0, 1, 1, 0, 0, 1])),
    ],
)
def test_mcnemar_p_exact_matches_binomtest_formula(a, b):
    # documented formula: p_exact = min(1, 2 * binomtest(min(b01, b10), b01+b10, .5).pvalue)
    r = mcnemar_test(a, b)
    b01, b10 = r.b, r.c
    expected = min(1.0, 2.0 * stats.binomtest(min(b01, b10), b01 + b10, 0.5).pvalue)
    assert r.p_exact == pytest.approx(expected)


def test_mcnemar_identical_vectors():
    # identical outcomes -> no discordant pairs -> b = c = 0 -> p = 1 by definition
    v = np.array([1, 0, 1, 1, 0])
    r = mcnemar_test(v, v.copy())
    assert (r.b, r.c) == (0, 0)
    assert r.statistic == 0.0
    assert r.p_exact == 1.0
    assert r.p_chi2 == 1.0


# Cochran's Q fixture: 8 items x 3 models (columns m1, m2, m3), hand-computed:
#
#   item | m1 m2 m3        col sums: m1 = 1+1+1+1+1+0+0+1 = 6
#   -----+---------                    m2 = 1+1+0+0+1+1+0+1 = 5
#     1  |  1  1  1                    m3 = 1+1+1+0+0+0+0+0 = 3
#     2  |  1  1  1        row sums: [3, 3, 2, 1, 2, 1, 0, 2]
#     3  |  1  0  1        sum(row^2) = 9+9+4+1+4+1+0+4 = 32
#     4  |  1  0  0        T = total successes = 6+5+3 = 14
#     5  |  1  1  0        sum(col^2) = 36+25+9 = 70
#     6  |  0  1  0        denom = k*T - sum(row^2) = 3*14 - 32 = 10
#     7  |  0  0  0        Q = (k-1)(k*sum(col^2) - T^2)/denom
#     8  |  1  1  0             = 2*(3*70 - 196)/10 = 2*14/10 = 2.8, df = 2
COCHRAN_M = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 0, 1],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 0],
    [1, 1, 0],
])


def test_cochran_q_hand_computed():
    out = cochran_q(COCHRAN_M)
    assert out["Q"] == pytest.approx(2.8)
    assert out["df"] == 2
    # For df = 2 the chi2 survival function has the closed form exp(-x/2):
    # p = exp(-2.8/2) = exp(-1.4) = 0.2465970
    assert out["p"] == pytest.approx(math.exp(-1.4))


def test_cochran_q_all_identical_models():
    # identical columns -> denom = 0 -> degenerate case returns Q = 0, p = 1
    col = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]])
    m = np.column_stack([col[:, 0], col[:, 0], col[:, 0]])
    out = cochran_q(m)
    assert out["Q"] == 0.0
    assert out["p"] == 1.0


def test_chi_square_independence_hand_computed():
    # 2x2 table [[10, 10], [10, 30]], n = 60 (hand computation, WITH the Yates
    # continuity correction that scipy applies by default for 2x2 tables):
    #   row totals 20, 40 ; col totals 20, 40
    #   expected = [[20*20/60, 20*40/60], [40*20/60, 40*40/60]]
    #            = [[6.6667, 13.3333], [13.3333, 26.6667]]
    #   |O - E| = 3.3333 in every cell ; Yates term = 3.3333 - 0.5 = 2.8333
    #   chi2 = 4 * 2.8333^2 / E summed:
    #       = 8.02778/6.6667 + 8.02778/13.3333 + 8.02778/13.3333 + 8.02778/26.6667
    #       = 1.204167 + 0.602083 + 0.602083 + 0.301042 = 2.709375
    out = chi_square_independence(np.array([[10, 10], [10, 30]]))
    assert out["chi2"] == pytest.approx(2.709375)
    assert out["df"] == 1
    assert out["p"] == pytest.approx(stats.chi2.sf(2.709375, df=1))
    # min expected cell = 6.6667 >= 5 -> chi-square assumptions hold
    assert out["min_expected"] == pytest.approx(20 * 20 / 60)
    assert out["cells_below_5"] == 0
    assert out["assumptions_ok"] is True


def test_benjamini_hochberg_known_adjustment():
    # pvals [0.01, 0.04, 0.03] (hand):
    #   sorted: [0.01, 0.03, 0.04], n = 3, ranks 1, 2, 3
    #   raw adjusted = p * n / rank = [0.03, 0.045, 0.04]
    #   enforce monotonicity (cumulative min from the largest p down):
    #       [0.03, 0.04, 0.04]
    #   mapped back to input order [0.01, 0.04, 0.03] -> [0.03, 0.04, 0.04]
    adj, rejected = benjamini_hochberg([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.04, 0.04])
    # all adjusted values <= 0.05 -> all rejected at q = 0.05
    assert rejected == [True, True, True]


def test_benjamini_hochberg_monotonicity():
    # adjusted p-values must be non-decreasing when the raw p-values are sorted
    pvals = [0.03, 0.001, 0.2, 0.04, 0.35, 0.6]
    adj, _ = benjamini_hochberg(pvals)
    # consecutive pairs: zip lengths differ by design
    for raw_a, adj_a, raw_b, adj_b in zip(pvals, adj, pvals[1:], adj[1:], strict=False):
        if raw_b > raw_a:
            assert adj_b >= adj_a - 1e-12
    # adjusted values are valid p-values (clipped to [0, 1])
    assert all(0.0 <= a <= 1.0 for a in adj)


def test_holm_bonferroni_hand_case():
    # pvals [0.01, 0.04, 0.2] (hand):
    #   sorted [0.01, 0.04, 0.2], multipliers (n - rank + 1) = [3, 2, 1]
    #   raw = [0.03, 0.08, 0.2] ; step-down monotonicity (cumulative max) leaves them
    #   unchanged -> original order [0.03, 0.08, 0.2]
    #   rejected at 0.05: [True, False, False]
    adj, rejected = holm_bonferroni([0.01, 0.04, 0.2])
    assert adj == pytest.approx([0.03, 0.08, 0.2])
    assert rejected == [True, False, False]


def test_holm_dominates_bh():
    # Holm adjusted p-values are always >= BH adjusted p-values elementwise.
    # For [0.01, 0.04, 0.03]: Holm = sorted [0.01,0.03,0.04] x [3,2,1] = [0.03,0.06,0.04]
    #   -> cumulative max [0.03, 0.06, 0.06] -> input order [0.03, 0.06, 0.06]
    # BH = [0.03, 0.04, 0.04]  -> 0.03>=0.03, 0.06>=0.04, 0.06>=0.04 (all true)
    bh_adj, _ = benjamini_hochberg([0.01, 0.04, 0.03])
    holm_adj, _ = holm_bonferroni([0.01, 0.04, 0.03])
    assert holm_adj == pytest.approx([0.03, 0.06, 0.06])
    assert all(h >= b - 1e-12 for h, b in zip(holm_adj, bh_adj, strict=True))
