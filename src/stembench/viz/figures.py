"""Publication figures. All generated from run artifacts; accessible color palette."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Okabe-Ito colorblind-safe palette
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9",
           "#F0E442", "#000000"]


def _short(name: str) -> str:
    return (name.split("/")[-1].replace(":free", "").replace("-free", "")
            .replace("openrouter__", "").replace("zen__", ""))[:28]


def reliability_grid(metrics: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for key, m in sorted(metrics.items()):
        for channel in ("calibration_self_report", "calibration_token_prob"):
            cal = m.get(channel)
            if cal and cal.get("n"):
                entries.append((key, channel.replace("calibration_", ""), cal))
    if not entries:
        return
    cols = 3
    rows = (len(entries) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)
    for ax, (key, channel, cal) in zip(
        (a for row in axes for a in row), entries, strict=False
    ):
        bins = [b for b in cal["reliability_bins"] if b["n"]]
        x = [b["avg_conf"] for b in bins]
        y = [b["acc"] for b in bins]
        n = [b["n"] for b in bins]
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=0.8, label="perfect")
        ax.plot(x, y, "o-", color=PALETTE[0], label="observed")
        for xi, yi, ni in zip(x, y, n, strict=True):
            ax.annotate(f"{ni}", (xi, yi), fontsize=6, color="grey",
                        xytext=(2, -8), textcoords="offset points")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("confidence")
        ax.set_ylabel("accuracy")
        ax.set_title(f"{_short(key)} [{channel}] ECE={cal['ECE']:.3f} n={cal['n']}",
                     fontsize=8)
        ax.legend(fontsize=7)
    for ax in list(a for row in axes for a in row)[len(entries):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "reliability.png")
    plt.close(fig)


def accuracy_ci_plot(metrics: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, m in sorted(metrics.items()):
        l = m["accuracy_lenient"]
        rows.append((_short(key), l["acc"], l["ci_lo"], l["ci_hi"], l["n"]))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(rows) + 1.5))
    for i, (name, acc, lo, hi, n) in enumerate(rows):
        ax.errorbar(acc, i, xerr=[[acc - lo], [hi - acc]], fmt="o",
                    color=PALETTE[0], capsize=3)
        ax.annotate(f"n={n}", (hi, i), fontsize=7, color="grey", xytext=(4, -3),
                    textcoords="offset points")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("accuracy (lenient) with Wilson 95% CI")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_ci.png")
    plt.close(fig)


def subject_heatmap(by_subject: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    models = sorted(by_subject)
    subjects = sorted({s for m in models for s in by_subject[m]})
    M = np.full((len(models), len(subjects)), np.nan)
    for i, m in enumerate(models):
        for j, s in enumerate(subjects):
            cell = by_subject[m].get(s)
            if cell and not cell["small_cell"]:
                M[i, j] = cell["acc"]
    if np.all(np.isnan(M)):
        return
    fig, ax = plt.subplots(figsize=(0.42 * len(subjects) + 3, 0.5 * len(models) + 2))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels(subjects, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([_short(m) for m in models], fontsize=8)
    for i in range(len(models)):
        for j in range(len(subjects)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8, label="accuracy")
    ax.set_title("Accuracy by subject (cells with n<5 blank)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "subject_heatmap.png")
    plt.close(fig)


def confusion_grid(metrics: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = [(k, m["mc_confusion"]) for k, m in sorted(metrics.items())
               if "mc_confusion" in m]
    if not entries:
        return
    cols = 3
    rows = (len(entries) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.0 * rows), squeeze=False)
    flat = [a for row in axes for a in row]
    for ax, (key, cm) in zip(flat, entries, strict=False):
        arr = np.array(cm, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            norm = np.where(arr.sum(axis=1, keepdims=True) > 0,
                            arr / arr.sum(axis=1, keepdims=True), 0.0)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                ax.text(j, i, int(arr[i, j]), ha="center", va="center",
                        fontsize=7, color="black" if norm[i, j] < 0.6 else "white")
        ax.set_xticks(range(4))
        ax.set_xticklabels(list("ABCD"))
        ax.set_yticks(range(4))
        ax.set_yticklabels(list("ABCD"))
        ax.set_xlabel("predicted")
        ax.set_ylabel("gold")
        ax.set_title(_short(key), fontsize=8)
    for ax in flat[len(entries):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion.png")
    plt.close(fig)


def language_gap_forest(results: dict, out_dir: Path) -> None:
    """results: {model_display: {diff, ci_lo, ci_hi, p_bootstrap}} for RU-EN gaps."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(results)
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(names) + 1.5))
    for i, name in enumerate(names):
        r = results[name]
        ax.errorbar(r["diff"], i, xerr=[[r["diff"] - r["ci_lo"]],
                                        [r["ci_hi"] - r["diff"]]],
                    fmt="o", color=PALETTE[1], capsize=3)
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("accuracy(EN) − accuracy(RU) on paired items, cluster bootstrap 95% CI")
    fig.tight_layout()
    fig.savefig(out_dir / "language_gap_forest.png")
    plt.close(fig)


def error_taxonomy_bar(dist: dict[str, float], out_dir: Path, n_total: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    items = sorted(dist.items(), key=lambda kv: -kv[1])
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(items) + 1.5))
    ax.barh([k for k, _ in items][::-1], [v for _, v in items][::-1],
            color=PALETTE[2])
    ax.set_xlabel("share of incorrect responses (multi-label)")
    ax.set_title(f"Error taxonomy distribution (n={n_total} incorrect responses)")
    fig.tight_layout()
    fig.savefig(out_dir / "error_taxonomy.png")
    plt.close(fig)
