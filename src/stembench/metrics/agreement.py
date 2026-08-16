"""Inter-annotator agreement: Cohen's kappa (2 raters), Fleiss' kappa (m raters).

Only genuine independent human annotations may feed these functions in reported
analyses (contract §4). Unit tests use fixtures; no human-kappa value is reported
anywhere in this project because no human annotators were available.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def cohens_kappa(r1: list[int], r2: list[int], n_classes: int) -> dict[str, float]:
    r1 = np.asarray(r1)
    r2 = np.asarray(r2)
    assert len(r1) == len(r2)
    n = len(r1)
    cm = np.zeros((n_classes, n_classes), dtype=float)
    for a, b in zip(r1, r2, strict=True):
        cm[a, b] += 1
    po = np.trace(cm) / n
    pe = float((cm.sum(axis=1) @ cm.sum(axis=0)) / (n * n))
    if pe == 1.0:
        kappa = 1.0
    else:
        kappa = (po - pe) / (1 - pe)
    # large-sample SE and 95% CI (Fleiss 1971 style approximation)
    se = _kappa_se(cm, po, pe, n)
    z = 1.959963984540054
    return {
        "kappa": float(kappa),
        "po": float(po),
        "pe": pe,
        "se": float(se),
        "ci_lo": float(kappa - z * se),
        "ci_hi": float(kappa + z * se),
        "n": int(n),
    }


def _kappa_se(cm: np.ndarray, po: float, pe: float, n: int) -> float:
    k = cm.shape[0]
    row = cm.sum(axis=1) / n
    col = cm.sum(axis=0) / n
    theta = 0.0
    for i in range(k):
        for j in range(k):
            w = 1.0 if i == j else 0.0
            theta += cm[i, j] / n * (w - po) ** 2
    # standard large-sample variance of kappa
    sum_terms = 0.0
    for i in range(k):
        inner = row[i] * col[i]
        for j in range(k):
            inner_j = row[j] * col[j]
            sum_terms += row[i] * col[i] * (row[j] + col[j]) - (inner if i == j else 0) - 2 * inner * inner_j / max(pe, 1e-12)
    var = (theta + pe**2 - sum_terms * pe / max(1 - pe, 1e-12) * 0) / (n * (1 - pe) ** 2)
    return float(np.sqrt(max(var, 0.0)))


def fleiss_kappa(ratings: np.ndarray) -> dict[str, float]:
    """ratings: (n_items x n_categories) counts of raters choosing each category.

    All rows must have the same total number of raters m.
    """
    R = np.asarray(ratings, dtype=float)
    n, k = R.shape
    m = R[0].sum()
    assert np.allclose(R.sum(axis=1), m), "each item must have the same number of raters"
    p_j = R.sum(axis=0) / (n * m)
    P_i = ((R**2).sum(axis=1) - m) / (m * (m - 1))
    Pbar = P_i.mean()
    Pe = float((p_j**2).sum())
    if Pe == 1.0:
        kappa = 1.0
    else:
        kappa = (Pbar - Pe) / (1 - Pe)
    return {"kappa": float(kappa), "Pbar": float(Pbar), "Pe": Pe, "n_items": int(n),
            "n_raters": int(m), "n_categories": int(k)}


def kappa_significance_boot(r1: list[int], r2: list[int], n_classes: int,
                            n_boot: int = 2000, seed: int = 0) -> float:
    """Bootstrap p-value for H0: kappa <= 0 (resample paired labels)."""
    rng = np.random.default_rng(seed)
    r1a, r2a = np.asarray(r1), np.asarray(r2)
    n = len(r1a)
    obs = cohens_kappa(r1, r2, n_classes)["kappa"]
    count = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        k = cohens_kappa(r1a[idx].tolist(), r2a[idx].tolist(), n_classes)["kappa"]
        if k >= obs:
            count += 1
    return (count + 1) / (n_boot + 1)


def binomial_ci_for_kappa_po(n_correct: int, n: int) -> tuple[float, float]:
    from stembench.metrics.intervals import wilson_interval

    return wilson_interval(n_correct, n)
