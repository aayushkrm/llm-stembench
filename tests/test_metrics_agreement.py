"""Tests for stembench.metrics.agreement: Cohen's kappa and Fleiss' kappa.

All expected values hand-computed (arithmetic in comments).
"""

from __future__ import annotations

import numpy as np
import pytest

from stembench.metrics.agreement import cohens_kappa, fleiss_kappa


def _ratings_from_cm(cm: list[list[int]]) -> tuple[list[int], list[int]]:
    """Build paired rating lists that realize a given confusion matrix cm[r1][r2]."""
    r1: list[int] = []
    r2: list[int] = []
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            r1.extend([i] * cm[i][j])
            r2.extend([j] * cm[i][j])
    return r1, r2


def test_cohens_kappa_perfect_agreement():
    # identical raters -> po = 1, pe < 1 -> kappa = (1 - pe)/(1 - pe) = 1
    r1 = [0, 1, 2, 0, 1, 2, 2, 1]
    out = cohens_kappa(r1, list(r1), n_classes=3)
    assert out["po"] == pytest.approx(1.0)
    assert out["kappa"] == pytest.approx(1.0)
    assert out["n"] == 8


def test_cohens_kappa_known_fixture():
    # Standard literature fixture: 2 raters, 50 items, confusion matrix
    #   cm = [[20, 5],       r1=0 row total 25 ; r2=0 col total 30
    #         [10, 15]]      r1=1 row total 25 ; r2=1 col total 20
    # Hand computation:
    #   po = trace/n = (20 + 15)/50 = 35/50 = 0.7
    #   pe = (25*30 + 25*20)/(50*50) = (750 + 500)/2500 = 1250/2500 = 0.5
    #   kappa = (po - pe)/(1 - pe) = (0.7 - 0.5)/(1 - 0.5) = 0.2/0.5 = 0.4
    r1, r2 = _ratings_from_cm([[20, 5], [10, 15]])
    assert len(r1) == 50
    out = cohens_kappa(r1, r2, n_classes=2)
    assert out["po"] == pytest.approx(0.7)
    assert out["pe"] == pytest.approx(0.5)
    assert out["kappa"] == pytest.approx(0.4)
    assert out["n"] == 50


def test_cohens_kappa_disjoint_is_negative():
    # complete disagreement: cm = [[0, 10], [10, 0]]
    #   po = 0 ; row = [0.5, 0.5], col = [0.5, 0.5]
    #   pe = 0.5*0.5 + 0.5*0.5 = 0.5 ; kappa = (0 - 0.5)/(1 - 0.5) = -1
    r1 = [0, 0, 1, 1] * 5
    r2 = [1, 1, 0, 0] * 5
    out = cohens_kappa(r1, r2, n_classes=2)
    assert out["po"] == pytest.approx(0.0)
    assert out["pe"] == pytest.approx(0.5)
    assert out["kappa"] == pytest.approx(-1.0)
    assert out["kappa"] < 0


# Fleiss' kappa fixture (published-style worked example): 6 items, 10 raters,
# 3 categories. Matrix rows = per-item rater counts per category (each row sums
# to m = 10). Hand computation:
#
#   item  counts [c1, c2, c3]   P_i = (sum r^2 - m) / (m*(m-1)) = (sum r^2 - 10)/90
#   1:   [10,  0,  0]           (100 - 10)/90 = 1
#   2:   [ 0, 10,  0]           (100 - 10)/90 = 1
#   3:   [ 6,  4,  0]           (36 + 16 - 10)/90 = 42/90 = 7/15
#   4:   [ 0,  6,  4]           (36 + 16 - 10)/90 = 7/15
#   5:   [ 5,  5,  0]           (25 + 25 - 10)/90 = 40/90 = 4/9
#   6:   [ 3,  3,  4]           (9 + 9 + 16 - 10)/90 = 24/90 = 4/15
#
#   Pbar = (1 + 1 + 7/15 + 7/15 + 4/9 + 4/15)/6
#        = (45 + 45 + 21 + 21 + 20 + 12)/45 / 6 = (164/45)/6 = 82/135 = 0.607407
#   column sums = [24, 28, 8], total ratings = 60
#   p_j = [24/60, 28/60, 8/60] = [0.4, 7/15, 2/15]
#   Pe = 0.4^2 + (7/15)^2 + (2/15)^2 = 36/225 + 49/225 + 4/225 = 89/225 = 0.395556
#   kappa = (Pbar - Pe)/(1 - Pe) = (82/135 - 89/225)/(1 - 89/225)
#         = (143/675)/(408/675) = 143/408 = 0.350490
FLEISS_R = np.array([
    [10, 0, 0],
    [0, 10, 0],
    [6, 4, 0],
    [0, 6, 4],
    [5, 5, 0],
    [3, 3, 4],
], dtype=float)


def test_fleiss_kappa_worked_example():
    out = fleiss_kappa(FLEISS_R)
    assert out["Pbar"] == pytest.approx(82 / 135)
    assert out["Pe"] == pytest.approx(89 / 225)
    assert out["kappa"] == pytest.approx(143 / 408)
    assert out["n_items"] == 6
    assert out["n_raters"] == 10
    assert out["n_categories"] == 3


def test_fleiss_kappa_perfect_agreement():
    # every item rated unanimously by all 10 raters in a distinct category mix:
    # each P_i = (100 - 10)/90 = 1 -> Pbar = 1; p_j = [2/6, 3/6, 1/6] ->
    # Pe = 4/36 + 9/36 + 1/36 = 14/36 = 7/18 -> kappa = (1 - 7/18)/(1 - 7/18) = 1
    R = np.array([
        [10, 0, 0],
        [10, 0, 0],
        [0, 10, 0],
        [0, 10, 0],
        [0, 10, 0],
        [0, 0, 10],
    ], dtype=float)
    out = fleiss_kappa(R)
    assert out["Pbar"] == pytest.approx(1.0)
    assert out["Pe"] == pytest.approx(7 / 18)
    assert out["kappa"] == pytest.approx(1.0)


def test_fleiss_kappa_unequal_rater_totals_raises():
    # rows summing to different rater totals (10 vs 11) violate the method's
    # equal-m assumption and must raise an AssertionError
    bad = np.array([[10, 0, 0], [8, 2, 1]], dtype=float)  # row sums 10 and 11
    with pytest.raises(AssertionError):
        fleiss_kappa(bad)


def test_cohens_kappa_length_mismatch_raises():
    with pytest.raises(AssertionError):
        cohens_kappa([0, 1], [0], n_classes=2)
