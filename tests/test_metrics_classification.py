"""Tests for stembench.metrics.classification with fully hand-computed fixtures."""

from __future__ import annotations

import math

import numpy as np
import pytest

from stembench.metrics.classification import accuracy, confusion_matrix, prf


def test_accuracy():
    # y_true = [0,1,1], y_pred = [0,1,0]: matches at positions 0 and 1 -> 2/3
    assert accuracy([0, 1, 1], [0, 1, 0]) == pytest.approx(2 / 3)
    assert accuracy([1, 1], [1, 1]) == 1.0
    assert accuracy([0], [1]) == 0.0


def test_accuracy_empty_is_nan():
    assert math.isnan(accuracy([], []))


def test_confusion_matrix_counts():
    # rows = true, cols = pred. Hand-counted for y_true=[0,0,1,2,1], y_pred=[0,1,1,2,1]:
    # (0,0)->cm[0,0]=1 ; (0,1)->cm[0,1]=1 ; (1,1)->cm[1,1]=2 ; (2,2)->cm[2,2]=1
    cm = confusion_matrix([0, 0, 1, 2, 1], [0, 1, 1, 2, 1], k=3)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [0, 0, 1]]


# Fixture: 3x3 confusion matrix (rows = true, cols = pred), hand-computed below.
#
#        pred: 0    1    2
# true 0:     5    1    0     row sum 6
# true 1:     0    3    2     row sum 5
# true 2:     1    0    4     row sum 5
# col sums:   6    4    6     total 16
#
# tp           = [5, 3, 4]
# fp (col-t p) = colsum - tp = [6-5, 4-3, 6-4] = [1, 1, 2]
# fn (row-t p) = rowsum - tp = [6-5, 5-3, 5-4] = [1, 2, 1]
# precision    = tp/(tp+fp)  = [5/6, 3/4, 4/6] = [0.833333, 0.75, 0.666667]
# recall       = tp/(tp+fn)  = [5/6, 3/5, 4/5] = [0.833333, 0.6, 0.8]
# f1           = 2PR/(P+R)   = [5/6, 2/3, 8/11]  (hand:
#                 class0: 2*(5/6*5/6)/(5/6+5/6) = 5/6
#                 class1: 2*(3/4*3/5)/(3/4+3/5) = (9/10)/(27/20) = 2/3
#                 class2: 2*(2/3*4/5)/(2/3+4/5) = (16/15)/(22/15) = 8/11)
CM = np.array([[5, 1, 0], [0, 3, 2], [1, 0, 4]])


def test_prf_per_class():
    out = prf(CM, average="none")
    assert out["precision_per_class"] == pytest.approx([5 / 6, 3 / 4, 2 / 3])
    assert out["recall_per_class"] == pytest.approx([5 / 6, 3 / 5, 4 / 5])
    assert out["f1_per_class"] == pytest.approx([5 / 6, 2 / 3, 8 / 11])
    assert out["support"] == [6, 5, 5]


def test_prf_macro():
    out = prf(CM, average="macro")
    # macro = unweighted class mean:
    # precision = (5/6 + 3/4 + 2/3)/3 = (0.833333+0.75+0.666667)/3 = 2.25/3 = 0.75
    # recall    = (5/6 + 3/5 + 4/5)/3 = (0.833333+0.6+0.8)/3 = 2.233333/3 = 0.744444
    # f1        = (5/6 + 2/3 + 8/11)/3 = (0.833333+0.666667+0.727273)/3 = 0.742424
    assert out["precision"] == pytest.approx(0.75)
    assert out["recall"] == pytest.approx((5 / 6 + 3 / 5 + 4 / 5) / 3)
    assert out["f1"] == pytest.approx((5 / 6 + 2 / 3 + 8 / 11) / 3)


def test_prf_micro():
    out = prf(CM, average="micro")
    # micro over the whole table: tp.sum() = 12, n = 16
    # precision = recall = 12/16 = 0.75
    # f1 = 2*12 / (2*12 + fp.sum() + fn.sum()) = 24 / (24 + 4 + 4) = 24/32 = 0.75
    assert out["precision"] == pytest.approx(12 / 16)
    assert out["recall"] == pytest.approx(12 / 16)
    assert out["f1"] == pytest.approx(24 / 32)


def test_prf_weighted():
    out = prf(CM, average="weighted")
    # support = [6, 5, 5], total 16 (support = row sums = true-class counts)
    # precision = (5/6*6 + 3/4*5 + 2/3*5)/16 = (5 + 3.75 + 10/3)/16 = 12.083333/16
    # recall    = (5/6*6 + 3/5*5 + 4/5*5)/16 = (5 + 3 + 4)/16 = 12/16 = 0.75
    # f1        = (5/6*6 + 2/3*5 + 8/11*5)/16 = (5 + 10/3 + 40/11)/16
    assert out["precision"] == pytest.approx((5 + 3.75 + 10 / 3) / 16)
    assert out["recall"] == pytest.approx(12 / 16)
    assert out["f1"] == pytest.approx((5 + 10 / 3 + 40 / 11) / 16)


def test_prf_empty_class_no_nan():
    # class 1 never occurs and is never predicted: tp=fp=fn=0.
    # Division-by-zero must be handled as 0 (not NaN, not an exception).
    cm = np.array([[2, 0, 0], [0, 0, 0], [0, 0, 2]])
    out = prf(cm, average="macro")
    # per class 0 and 2 everything is perfect (prec=rec=f1=1), class 1 contributes 0
    assert out["precision_per_class"] == pytest.approx([1.0, 0.0, 1.0])
    assert out["recall_per_class"] == pytest.approx([1.0, 0.0, 1.0])
    assert out["f1_per_class"] == pytest.approx([1.0, 0.0, 1.0])
    # macro over 3 classes = 2/3
    assert out["precision"] == pytest.approx(2 / 3)
    for key in ("precision", "recall", "f1"):
        assert not math.isnan(out[key])
