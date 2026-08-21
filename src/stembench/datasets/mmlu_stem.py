"""MMLU-STEM loader (Stage 1 pilot dataset).

Uses `cais/mmlu` "all" config, test split, filtered to the official STEM subject group.
Dataset revision (HF commit sha) is captured for the run manifest; loading is cached by
HF datasets. License of cais/mmlu data: MIT (per dataset card) — used here for evaluation
only, not redistributed.
"""

from __future__ import annotations

import random
from typing import Any

from stembench.schemas import Language, MCItem

STEM_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "college_biology",
    "college_chemistry", "college_computer_science", "college_mathematics",
    "college_physics", "computer_security", "conceptual_physics",
    "electrical_engineering", "elementary_mathematics", "formal_logic",
    "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_mathematics", "high_school_physics", "high_school_statistics",
    "machine_learning", "virology",
]


def dataset_revision() -> str:
    try:
        from huggingface_hub import HfApi

        info = HfApi().dataset_info("cais/mmlu")
        return str(info.sha or "")
    except Exception:  # noqa: BLE001 - offline fallback
        return ""


def load_mmlu_stem(split: str = "test") -> tuple[list[MCItem], dict[str, Any]]:
    """Load STEM-subject MMLU items from the test split as MCItems."""
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split=split)
    items: list[MCItem] = []
    n_by_subject: dict[str, int] = {}
    for row in ds:
        subj = row["subject"]
        if subj not in STEM_SUBJECTS:
            continue
        n_by_subject[subj] = n_by_subject.get(subj, 0) + 1
        items.append(
            MCItem(
                item_id=f"mmlu-{subj}-{n_by_subject[subj]}",
                dataset="mmlu_stem",
                subject=subj,
                language=Language.EN,
                difficulty="unknown",
                question=row["question"],
                choices=list(row["choices"]),
                gold=int(row["answer"]),
            )
        )
    snapshot = {
        "dataset": "cais/mmlu",
        "config": "all",
        "split": split,
        "revision": dataset_revision(),
        "stem_subjects": STEM_SUBJECTS,
        "n_total": len(items),
        "n_by_subject": n_by_subject,
    }
    return items, snapshot


def stratified_sample(
    items: list[MCItem], n: int, seed: int, by: str = "subject"
) -> list[MCItem]:
    """Deterministic stratified sample: proportional allocation by stratum with
    largest-remainder rounding; shuffling inside strata by seeded RNG."""
    rng = random.Random(seed)
    strata: dict[str, list[MCItem]] = {}
    for it in items:
        strata.setdefault(getattr(it, by), []).append(it)
    if not strata:
        return []
    total = sum(len(v) for v in strata.values())
    n = min(n, total)  # cannot sample more than exists (prevents alloc overflow)
    alloc: dict[str, int] = {}
    raw: dict[str, float] = {}
    for s, v in strata.items():
        raw[s] = len(v) / total * n
        alloc[s] = int(raw[s])
    # largest remainder; each stratum is capped at its own size
    remaining = n - sum(alloc.values())
    order = sorted(strata, key=lambda s: (raw[s] - alloc[s], rng.random()), reverse=True)
    guard = 0
    while remaining > 0 and guard < 10 * max(n, 1):
        s = order[guard % len(order)]
        if alloc[s] < len(strata[s]):
            alloc[s] += 1
            remaining -= 1
        guard += 1
    out: list[MCItem] = []
    for s, v in strata.items():
        picked = rng.sample(v, min(alloc[s], len(v)))
        out.extend(picked)
    out.sort(key=lambda it: it.item_id)
    return out
