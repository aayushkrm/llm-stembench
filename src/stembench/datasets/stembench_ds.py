"""Loader for the original STEMBench bilingual benchmark (data/stembench_v1/items.jsonl)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stembench.schemas import (
    AnswerType,
    BenchmarkItem,
    FreeResponseItem,
    MCItem,
    Tolerance,
)


def load_stembench(path: str | Path = "data/stembench_v1/items.jsonl",
                   split: str | None = None,
                   language: str | None = None) -> tuple[list[BenchmarkItem], dict[str, Any]]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(BenchmarkItem.model_validate(json.loads(line)))
    total = len(items)
    if split:
        items = [i for i in items if i.split == split]
    if language:
        items = [i for i in items if i.language == language]
    stats = {
        "path": str(path),
        "n_total": total,
        "n_selected": len(items),
        "split": split,
        "language": language,
    }
    return items, stats


def to_eval_items(bench_items: list[BenchmarkItem]) -> list[MCItem | FreeResponseItem]:
    """Convert BenchmarkItems into model-facing eval items (MC or free response)."""
    out: list[MCItem | FreeResponseItem] = []
    for b in bench_items:
        if b.answer_type == AnswerType.MC:
            gold = next(
                i for i, ch in enumerate(b.choices) if ch.label == b.canonical_answer
            )
            out.append(
                MCItem(
                    item_id=b.item_id,
                    dataset="stembench_v1",
                    subject=b.subject,
                    template_id=b.template_id,
                    language=b.language,
                    difficulty=b.difficulty,
                    question=b.question,
                    choices=[c.text for c in b.choices],
                    gold=gold,
                )
            )
        else:
            out.append(
                FreeResponseItem(
                    item_id=b.item_id,
                    dataset="stembench_v1",
                    subject=b.subject,
                    template_id=b.template_id,
                    language=b.language,
                    difficulty=b.difficulty,
                    question=b.question,
                    answer_type=b.answer_type,
                    gold=b.canonical_answer,
                    alternatives=b.acceptable_alternatives,
                    tolerance=Tolerance(
                        rel=b.tolerance.rel if b.tolerance else None,
                        abs=b.tolerance.abs if b.tolerance else None,
                    )
                    if b.tolerance
                    else None,
                    units=b.units,
                )
            )
    return out
