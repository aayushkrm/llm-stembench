"""Tests for stembench.metrics.intervals: Wilson interval and paired bootstraps."""

from __future__ import annotations

import numpy as np
import pytest

from stembench.metrics.intervals import (
    bootstrap_difference_ci,
    normal_interval,
    paired_bootstrap_ci,
    wilson_interval,
)


def test_wilson_zero_successes():
    # k=0, n=10: p=0 -> center - half collapses to exactly 0 (clamped at 0)
    lo, hi = wilson_interval(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-12)
    # hi > 0: the upper bound of a Wilson interval for 0/10 is ~0.2775
    assert 0.2 < hi < 0.35


def test_wilson_all_successes():
    # k=n=10: p=1 -> center + half = (1 + z^2/n)/(1 + z^2/n) = 1 (clamped at 1)
    lo, hi = wilson_interval(10, 10)
    assert hi == pytest.approx(1.0, abs=1e-12)
    assert 0.65 < lo < 0.8


def test_wilson_known_value():
    # Hand computation of the Wilson score interval for k=8, n=10, z=1.96:
    #   p = 0.8, z^2 = 3.8416
    #   denom  = 1 + z^2/n          = 1 + 0.38416        = 1.38416
    #   center = (p + z^2/(2n))/denom = (0.8 + 0.19208)/1.38416 = 0.99208/1.38416
    #          = 0.71672528...
    #   half   = (z/denom) * sqrt(p(1-p)/n + z^2/(4n^2))
    #          = (1.96/1.38416) * sqrt(0.016 + 0.009604)
    #          = 1.416055... * sqrt(0.025604) = 1.416055... * 0.160012...
    #          = 0.226568...
    #   lo = 0.716725 - 0.226568 = 0.490157 ; hi = 0.716725 + 0.226568 = 0.943319
    lo, hi = wilson_interval(8, 10, z=1.96)
    z, n, p = 1.96, 10, 0.8
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    assert lo == pytest.approx(center - half, abs=1e-6)
    assert hi == pytest.approx(center + half, abs=1e-6)
    assert lo == pytest.approx(0.490157, abs=1e-6)
    assert hi == pytest.approx(0.943319, abs=1e-6)


def test_wilson_contains_point_estimate():
    # k=5, n=20 -> p = 0.25 must lie inside the interval
    lo, hi = wilson_interval(5, 20)
    assert lo < 0.25 < hi


def test_wilson_zero_n_is_nan():
    lo, hi = wilson_interval(0, 0)
    assert np.isnan(lo) and np.isnan(hi)
    nlo, nhi = normal_interval(0, 0)
    assert np.isnan(nlo) and np.isnan(nhi)


def test_normal_interval_known_value():
    # k=8, n=10, z=1.96: p = 0.8, se = sqrt(0.8*0.2/10) = sqrt(0.016) = 0.126491
    # lo = 0.8 - 1.96*0.126491 = 0.8 - 0.247923 = 0.552077
    # hi = 0.8 + 0.247923 = 1.047923 -> clamped to 1.0
    lo, hi = normal_interval(8, 10, z=1.96)
    assert lo == pytest.approx(0.8 - 1.96 * np.sqrt(0.016), abs=1e-9)
    assert hi == 1.0


def _cluster_fixture():
    # 10 clusters x 2 values each: cluster c contributes [c, c]
    values = np.array([c for c in range(10) for _ in range(2)], dtype=float)
    clusters = np.array([c for c in range(10) for _ in range(2)])
    # point estimate: mean = (0+0+1+1+...+9+9)/20 = (2*(0+1+...+9))/20 = 90/20 = 4.5
    return values, clusters


def test_paired_bootstrap_ci_clusters_contains_point_estimate():
    values, clusters = _cluster_fixture()
    lo, hi = paired_bootstrap_ci(values, clusters=clusters, n_boot=1000, seed=42)
    # resampling whole clusters keeps every bootstrap mean in [0, 9];
    # the 2.5%/97.5% percentiles must bracket the point estimate 4.5
    assert lo <= 4.5 <= hi
    assert 0.0 <= lo < hi <= 9.0


def test_paired_bootstrap_ci_deterministic_with_same_seed():
    values, clusters = _cluster_fixture()
    r1 = paired_bootstrap_ci(values, clusters=clusters, n_boot=500, seed=7)
    r2 = paired_bootstrap_ci(values, clusters=clusters, n_boot=500, seed=7)
    assert r1 == r2


def test_paired_bootstrap_ci_unclustered():
    # plain iid bootstrap over 20 zeros/ones: 12 ones -> mean 0.6 must be inside the CI
    values = np.array([1.0] * 12 + [0.0] * 8)
    lo, hi = paired_bootstrap_ci(values, clusters=None, n_boot=1000, seed=1)
    assert lo <= 0.6 <= hi


def test_bootstrap_difference_ci_sign_and_content():
    # a is right on 15/20 items (mean 0.75), b on 5/20 (mean 0.25)
    a = np.array([1.0] * 15 + [0.0] * 5)
    b = np.array([1.0] * 5 + [0.0] * 15)
    out = bootstrap_difference_ci(a, b, n_boot=2000, seed=3)
    # point estimate mean(a) - mean(b) = 0.75 - 0.25 = +0.5 (sign correct: a > b)
    assert out["diff"] == pytest.approx(0.5)
    assert out["ci_lo"] <= 0.5 <= out["ci_hi"]
    assert out["ci_lo"] > 0.0  # the CI excludes 0 for this strongly paired difference
    # bootstrap p-value (2 * fraction of resampled diffs crossing 0) is tiny
    assert 0.0 <= out["p_bootstrap"] < 0.05
    assert set(out) >= {"diff", "ci_lo", "ci_hi", "p_bootstrap"}


def test_bootstrap_difference_ci_zero_when_identical():
    a = np.array([1.0, 0.0, 1.0, 0.0])
    out = bootstrap_difference_ci(a, a.copy(), n_boot=200, seed=5)
    # identical vectors: diff = 0, all bootstrap diffs = 0 -> p = 2*1 = 2 -> clamped 1
    assert out["diff"] == 0.0
    assert out["p_bootstrap"] == 1.0
