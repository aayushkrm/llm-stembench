"""Classification metrics: accuracy, P/R/F1 (macro/micro/weighted), confusion matrix."""

from __future__ import annotations

from collections import Counter

import numpy as np


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    assert len(y_true) == len(y_pred)
    if not y_true:
        return float("nan")
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion_matrix(y_true: list[int], y_pred: list[int], k: int) -> np.ndarray:
    """k x k matrix, rows=true, cols=pred."""
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred, strict=True):
        cm[t, p] += 1
    return cm


def prf(cm: np.ndarray, average: str = "macro") -> dict[str, float]:
    """Precision/recall/F1 per class and averaged, from a confusion matrix.

    average: 'macro' (unweighted class mean), 'micro' (global counts),
    'weighted' (support-weighted), 'none' (per class only).
    """
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    support = cm.sum(axis=1)
    out = {
        "precision_per_class": prec.tolist(),
        "recall_per_class": rec.tolist(),
        "f1_per_class": f1.tolist(),
        "support": support.tolist(),
    }
    if average == "micro":
        out.update(precision=float(tp.sum() / cm.sum()), recall=float(tp.sum() / cm.sum()),
                   f1=float(2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum())))
    elif average == "macro":
        out.update(precision=float(prec.mean()), recall=float(rec.mean()), f1=float(f1.mean()))
    elif average == "weighted":
        tot = support.sum()
        if tot:
            out.update(
                precision=float((prec * support).sum() / tot),
                recall=float((rec * support).sum() / tot),
                f1=float((f1 * support).sum() / tot),
            )
    return out


def class_distribution(y: list[int], k: int) -> dict[int, int]:
    return {c: n for c, n in Counter(y).items() if 0 <= c < k}
