"""Confidence intervals: Wilson score interval, normal interval, paired bootstrap.

The paired bootstrap supports clustering (e.g., RU/EN variants of one benchmark pair:
resample pair clusters, not rows) for honest language-gap intervals.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def wilson_interval(k: int | float, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def normal_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    se = math.sqrt(p * (1 - p) / n)
    return max(0.0, p - z * se), min(1.0, p + z * se)


def paired_bootstrap_ci(
    values: np.ndarray,
    clusters: np.ndarray | None = None,
    n_boot: int = 10000,
    seed: int = 42,
    statistic=np.mean,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """CI for statistic(values). If clusters given, resample whole clusters.

    values: 1-D array of per-item (or per-cluster-aggregated) outcomes.
    clusters: integer cluster id per value; all values of a cluster move together.
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    if clusters is None:
        idx = rng.integers(0, len(values), size=(n_boot, len(values)))
        stats_ = statistic(values[idx], axis=1)
    else:
        clusters = np.asarray(clusters)
        uniq = np.unique(clusters)
        by_cluster = [values[clusters == c] for c in uniq]
        n_c = len(uniq)
        stats_ = np.empty(n_boot)
        for b in range(n_boot):
            pick = rng.integers(0, n_c, size=n_c)
            merged = np.concatenate([by_cluster[i] for i in pick])
            stats_[b] = statistic(merged)
    lo, hi = np.percentile(stats_, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_difference_ci(
    a: np.ndarray,
    b: np.ndarray,
    clusters: np.ndarray | None = None,
    n_boot: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """CI for mean(a) - mean(b), paired design (same items/clusters in a and b).

    If clusters given, resample clusters and use both a and b values of each cluster.
    Returns point estimate and CI, plus a bootstrap two-sided p-value (H0: diff = 0)
    computed as the fraction of bootstrap diffs crossing zero, doubled (percentile).
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert len(a) == len(b)
    point = float(np.mean(a) - np.mean(b))
    n = len(a)
    if clusters is None:
        idx = rng.integers(0, n, size=(n_boot, n))
        diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    else:
        clusters = np.asarray(clusters)
        uniq = np.unique(clusters)
        ac = {c: a[clusters == c] for c in uniq}
        bc = {c: b[clusters == c] for c in uniq}
        n_c = len(uniq)
        diffs = np.empty(n_boot)
        for i in range(n_boot):
            pick = rng.integers(0, n_c, size=n_c)
            aa = np.concatenate([ac[c] for c in uniq[pick]])
            bb = np.concatenate([bc[c] for c in uniq[pick]])
            diffs[i] = aa.mean() - bb.mean()
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_boot = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return {"diff": point, "ci_lo": float(lo), "ci_hi": float(hi), "p_bootstrap": float(p_boot)}
