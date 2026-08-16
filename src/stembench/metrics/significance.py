"""Significance tests: McNemar, Cochran's Q, Pearson chi-square, BH correction.

Design rule (decisions.md D5): the same items are answered by every model, so paired
tests are the default for model comparison. Chi-square is provided for genuinely
independent designs and for the source-specification requirement, with its assumption
check documented in reports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class McNemarResult:
    b: int  # model A correct, B wrong
    c: int  # A wrong, B correct
    statistic: float  # continuity-corrected chi2 (0 when b+c==0)
    p_chi2: float
    p_exact: float


def mcnemar_test(a: np.ndarray, b: np.ndarray) -> McNemarResult:
    """Exact (binomial) + continuity-corrected chi-square McNemar on paired binary outcomes."""
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    assert len(a) == len(b)
    b01 = int(((a == 1) & (b == 0)).sum())
    b10 = int(((a == 0) & (b == 1)).sum())
    n = b01 + b10
    if n == 0:
        return McNemarResult(b01, b10, 0.0, 1.0, 1.0)
    chi2 = (abs(b01 - b10) - 1) ** 2 / n
    p_chi2 = float(stats.chi2.sf(chi2, df=1))
    p_exact = float(stats.binomtest(min(b01, b10), n, 0.5).pvalue * 2)
    p_exact = min(1.0, p_exact)
    return McNemarResult(b01, b10, chi2, p_chi2, p_exact)


def cochran_q(matrix: np.ndarray) -> dict[str, float]:
    """Cochran's Q over a (n_items x k_models) binary correctness matrix."""
    M = np.asarray(matrix, dtype=float)
    assert M.ndim == 2
    n, k = M.shape
    col_sums = M.sum(axis=0)
    row_sums = M.sum(axis=1)
    T = col_sums.sum()
    denom = k * T - (row_sums**2).sum()
    if denom == 0:
        return {"Q": 0.0, "df": float(k - 1), "p": 1.0}
    Q = (k - 1) * (k * (col_sums**2).sum() - T**2) / denom
    p = float(stats.chi2.sf(Q, df=k - 1))
    return {"Q": float(Q), "df": float(k - 1), "p": p}


def chi_square_independence(table: np.ndarray) -> dict[str, float]:
    """Pearson chi-square test of independence (2-D contingency table)."""
    chi2, p, dof, expected = stats.chi2_contingency(np.asarray(table))
    # assumption flags: expected counts
    min_expected = float(expected.min())
    low_cells = int((expected < 5).sum())
    return {
        "chi2": float(chi2),
        "p": float(p),
        "df": int(dof),
        "min_expected": min_expected,
        "cells_below_5": low_cells,
        "assumptions_ok": bool(min_expected >= 5),
    }


def benjamini_hochberg(pvals: list[float]) -> tuple[list[float], list[bool]]:
    """Return BH-adjusted p-values and rejection flags at q=0.05."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    rejected = out <= 0.05
    return out.tolist(), rejected.tolist()


def holm_bonferroni(pvals: list[float]) -> tuple[list[float], list[bool]]:
    """Holm step-down adjusted p-values and rejection flags at alpha=0.05."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = np.maximum.accumulate(ranked * (n - np.arange(n)))
    adj = np.minimum(adj, 1.0)
    out = np.empty(n)
    out[order] = adj
    rejected = out <= 0.05
    return out.tolist(), rejected.tolist()
