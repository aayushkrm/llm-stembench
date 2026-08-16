"""Calibration: ECE, MCE, Brier, NLL, reliability bins.

Two confidence channels are supported and must be reported separately (decisions.md D4):
- `token_prob`: provider-exposed top choice probability;
- `self_report`: model-stated confidence from the output contract.
Binning: equal-width bins over [0,1] with sample counts per bin; empty bins are excluded
from ECE/MCE with their counts still reported.
"""

from __future__ import annotations

import numpy as np


def reliability_bins(
    confidences: np.ndarray, corrects: np.ndarray, n_bins: int = 10
) -> list[dict[str, float]]:
    out = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (
            (confidences >= lo) & (confidences <= hi)
        )
        n = int(mask.sum())
        row = {
            "bin_lo": float(lo),
            "bin_hi": float(hi),
            "n": n,
            "avg_conf": float(confidences[mask].mean()) if n else float("nan"),
            "acc": float(corrects[mask].mean()) if n else float("nan"),
        }
        out.append(row)
    return out


def expected_calibration_error(
    confidences: np.ndarray, corrects: np.ndarray, n_bins: int = 10
) -> float:
    if len(confidences) == 0:
        return float("nan")
    total = len(confidences)
    ece = 0.0
    for row in reliability_bins(confidences, corrects, n_bins):
        if row["n"]:
            ece += (row["n"] / total) * abs(row["avg_conf"] - row["acc"])
    return float(ece)


def maximum_calibration_error(
    confidences: np.ndarray, corrects: np.ndarray, n_bins: int = 10
) -> float:
    worst = 0.0
    for row in reliability_bins(confidences, corrects, n_bins):
        if row["n"]:
            worst = max(worst, abs(row["avg_conf"] - row["acc"]))
    return float(worst)


def brier_score(confidences: np.ndarray, corrects: np.ndarray) -> float:
    """Binary-outcome Brier: mean (confidence_in_answer - correct)^2.

    confidences here is the probability assigned to the given answer being correct
    (self-reported confidence in the given answer, or top-choice prob for MC).
    """
    if len(confidences) == 0:
        return float("nan")
    c = np.asarray(confidences, dtype=float)
    y = np.asarray(corrects, dtype=float)
    return float(np.mean((c - y) ** 2))


def brier_multiclass(probs: np.ndarray, y_true: list[int]) -> float:
    """Multiclass Brier: mean over items of sum_k (p_k - 1[y=k])^2."""
    P = np.asarray(probs, dtype=float)
    Y = np.zeros_like(P)
    Y[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((P - Y) ** 2, axis=1)))


def negative_log_likelihood(probs_true: np.ndarray) -> float:
    """NLL from the probability assigned to the true class / correct answer."""
    p = np.clip(np.asarray(probs_true, dtype=float), 1e-12, 1.0)
    return float(-np.mean(np.log(p)))
