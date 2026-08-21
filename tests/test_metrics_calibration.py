"""Tests for stembench.metrics.calibration with hand-computed fixtures."""

from __future__ import annotations

import math

import numpy as np
import pytest

from stembench.metrics.calibration import (
    brier_multiclass,
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    negative_log_likelihood,
    reliability_bins,
)


def test_ece_perfectly_calibrated_two_bins():
    # confidences [0.5, 0.5, 1.0, 1.0], corrects [0, 1, 1, 1], n_bins=2.
    # Bin 0 = [0, 0.5]: the two 0.5s -> avg_conf = 0.5, acc = 1/2 = 0.5 -> |diff| = 0
    # Bin 1 = (0.5, 1.0]: the two 1.0s -> avg_conf = 1.0, acc = 1.0 -> |diff| = 0
    # ECE = (2/4)*0 + (2/4)*0 = 0
    conf = np.array([0.5, 0.5, 1.0, 1.0])
    corr = np.array([0.0, 1.0, 1.0, 1.0])
    assert expected_calibration_error(conf, corr, n_bins=2) == pytest.approx(0.0)


def test_ece_mce_overconfident_fixture():
    # confidences [0.9, 0.9], corrects [1, 0], n_bins=10: both fall in bin (0.8, 0.9]
    # avg_conf = 0.9, acc = 1/2 = 0.5
    # ECE = (2/2) * |0.9 - 0.5| = 0.4 ; MCE = max over non-empty bins = 0.4
    conf = np.array([0.9, 0.9])
    corr = np.array([1.0, 0.0])
    assert expected_calibration_error(conf, corr, n_bins=10) == pytest.approx(0.4)
    assert maximum_calibration_error(conf, corr, n_bins=10) == pytest.approx(0.4)


def test_ece_weighted_bins():
    # 12 items at conf 0.9 (all correct), 4 items at conf 0.6 (all wrong), n=16.
    # bin (0.8,0.9]: avg_conf 0.9, acc 1.0, |diff| = 0.1, weight 12/16
    # bin (0.5,0.6]: avg_conf 0.6, acc 0.0, |diff| = 0.6, weight 4/16
    # ECE = (12/16)*0.1 + (4/16)*0.6 = 0.075 + 0.15 = 0.225
    # MCE = max(0.1, 0.6) = 0.6
    conf = np.array([0.9] * 12 + [0.6] * 4)
    corr = np.array([1.0] * 12 + [0.0] * 4)
    assert expected_calibration_error(conf, corr, n_bins=10) == pytest.approx(0.225)
    assert maximum_calibration_error(conf, corr, n_bins=10) == pytest.approx(0.6)


def test_brier_score_hand_case():
    # Brier = mean((conf - y)^2):
    # (0.8 - 1)^2 = 0.04 ; (0.5 - 0)^2 = 0.25 ; mean = (0.04 + 0.25)/2 = 0.145
    assert brier_score(np.array([0.8, 0.5]), np.array([1.0, 0.0])) == pytest.approx(0.145)
    # perfect confident prediction -> 0
    assert brier_score(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(0.0)


def test_nll_clip():
    # probabilities of the true class: p = [1e-20, 1.0], clipped to [1e-12, 1]
    # NLL = -mean(log(1e-12), log(1.0)) = (27.631021... + 0)/2 = 13.815511...
    assert negative_log_likelihood(np.array([1e-20, 1.0])) == pytest.approx(
        -math.log(1e-12) / 2
    )
    # p = 1 -> NLL = 0
    assert negative_log_likelihood(np.array([1.0])) == pytest.approx(0.0)


def test_brier_multiclass_hand_case():
    # 2 items, 3 classes; y = [0, 1]:
    #   item 0 probs [0.7, 0.2, 0.1] vs one-hot [1,0,0]:
    #     (0.7-1)^2 + 0.2^2 + 0.1^2 = 0.09 + 0.04 + 0.01 = 0.14
    #   item 1 probs [0.1, 0.8, 0.1] vs one-hot [0,1,0]:
    #     0.1^2 + (0.8-1)^2 + 0.1^2 = 0.01 + 0.04 + 0.01 = 0.06
    #   mean over items = (0.14 + 0.06)/2 = 0.10
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    assert brier_multiclass(probs, [0, 1]) == pytest.approx(0.10)


@pytest.mark.parametrize(
    "fn",
    [expected_calibration_error, brier_score],
)
def test_empty_input_nan_not_crash(fn):
    # empty arrays -> NaN (guarded), no exception
    assert math.isnan(fn(np.array([]), np.array([])))


def test_reliability_bins_counts_sum_to_n():
    # every confidence lands in exactly one bin, so the per-bin counts sum to n
    conf = np.array([0.05, 0.15, 0.55, 0.95, 1.0])
    corr = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    bins = reliability_bins(conf, corr, n_bins=10)
    assert sum(b["n"] for b in bins) == 5
    # edges are 0.0, 0.1, ..., 1.0 with bin i = (lo, hi], except bin 0 = [0, 0.1]
    # 0.05 -> bin 0 ; 0.15 -> bin 1 ; 0.55 -> bin 5 ; 0.95 -> bin 9 ; 1.0 -> bin 9
    assert bins[0]["n"] == 1
    assert bins[1]["n"] == 1
    assert bins[5]["n"] == 1
    assert bins[9]["n"] == 2
    # empty bins report NaN stats but keep their counts at 0
    assert bins[2]["n"] == 0
    assert math.isnan(bins[2]["avg_conf"])
