#!/usr/bin/env python3
"""Re-parse stored responses without making provider calls.

The command is deliberately limited to records with a non-empty raw response and a
stored reference answer.  Changed scoring fields retain their original values under
``extra.pre_rescore`` so every correction is auditable.  Files are replaced
atomically and a machine-readable summary is written alongside the run artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stembench.scoring import score_exact, score_mc, score_numeric

SCORING_FIELDS = ("parsed_answer", "parse_method", "correctness", "error_status")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _score(record: dict[str, Any], meta: dict[str, Any] | None = None) -> tuple[bool | None, str, str]:
    answer_type = record.get("answer_type")
    reference = record.get("reference_answer", "")
    raw = record.get("raw_response", "")
    extra = dict(meta or {})
    extra.update({k: v for k, v in (record.get("extra") or {}).items() if v not in (None, "", [])})
    if answer_type == "mc":
        if len(reference) != 1 or not reference.upper().isalpha():
            raise ValueError(f"invalid MC reference {reference!r}")
        n_choices = int(extra.get("n_choices", 4))
        return score_mc(raw, ord(reference.upper()) - ord("A"), n_choices=n_choices)
    if answer_type == "exact":
        alternatives = extra.get("acceptable_answers")
        return score_exact(raw, reference, alternatives=alternatives)
    if answer_type == "numeric":
        return score_numeric(
            raw,
            float(reference),
            rel_tol=extra.get("rel_tol"),
            abs_tol=extra.get("abs_tol"),
            require_unit=extra.get("unit", ""),
        )
    raise ValueError(f"unsupported answer_type {answer_type!r}")


def _load_items_meta(path: Path) -> dict[str, dict[str, Any]]:
    """Canonical scoring metadata per item_id from the built dataset."""
    meta: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        it = json.loads(line)
        entry: dict[str, Any] = {}
        if it.get("answer_type") == "mc":
            entry["n_choices"] = len(it.get("choices") or [])
        elif it.get("answer_type") == "numeric":
            entry["unit"] = it.get("units") or ""
            tol = it.get("tolerance") or {}
            entry["rel_tol"] = tol.get("rel")
            entry["abs_tol"] = tol.get("abs")
        else:
            entry["acceptable_answers"] = it.get("acceptable_alternatives") or []
        meta[it["item_id"]] = entry
    return meta


def _atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def rescore_run(run_dir: Path, *, apply: bool, items_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    totals = {"records": 0, "eligible": 0, "changed": 0, "correctness_flips": 0}

    for path in sorted(run_dir.glob("*.jsonl")):
        before_hash = _sha256(path)
        records: list[dict[str, Any]] = []
        changed = 0
        correctness_flips = 0
        eligible = 0

        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            records.append(record)
            totals["records"] += 1

            if not (record.get("raw_response") or "").strip():
                continue
            if not record.get("reference_answer"):
                raise ValueError(f"{path}:{line_number}: response has no reference_answer")

            eligible += 1
            totals["eligible"] += 1
            ok, parsed, method = _score(record, (items_meta or {}).get(record.get("item_id")))
            new_status = "parse_failure" if ok is None else ""
            new_values = {
                "parsed_answer": parsed,
                "parse_method": method,
                "correctness": ok,
                "error_status": new_status,
            }
            old_values = {field: record.get(field) for field in SCORING_FIELDS}
            if old_values == new_values:
                continue

            extra = record.setdefault("extra", {})
            if "pre_rescore" in extra:
                raise ValueError(
                    f"{path}:{line_number}: already contains extra.pre_rescore; "
                    "refusing to overwrite provenance"
                )
            extra["pre_rescore"] = old_values
            record.update(new_values)
            changed += 1
            totals["changed"] += 1
            if old_values["correctness"] != ok:
                correctness_flips += 1
                totals["correctness_flips"] += 1

        if apply and changed:
            _atomic_write_jsonl(path, records)
        after_hash = _sha256(path) if apply else before_hash
        files.append(
            {
                "path": str(path),
                "records": len(records),
                "eligible": eligible,
                "changed": changed,
                "correctness_flips": correctness_flips,
                "sha256_before": before_hash,
                "sha256_after": after_hash,
            }
        )

    return {"run_id": run_dir.name, "files": files, "totals": totals}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="run directories containing JSONL")
    parser.add_argument("--apply", action="store_true", help="atomically rewrite changed files")
    parser.add_argument(
        "--items", type=Path, default=None,
        help="items.jsonl supplying canonical unit/tolerance/alternatives per item_id",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/stage1/rescore_manifest.json"),
        help="summary path (written only with --apply)",
    )
    args = parser.parse_args()

    items_meta = _load_items_meta(args.items) if args.items else None
    summary = {
        "schema_version": 2,
        "operation": "offline_rescore_from_stored_raw_responses",
        "scoring_change": (
            "numeric unit check is answer-scoped (Answer: segment) with membership "
            "matching and RU aliases; whole-response unit extraction was picking "
            "units out of reasoning text"
        ),
        "items_source": str(args.items) if args.items else None,
        "provider_calls": 0,
        "applied": bool(args.apply),
        "runs": [rescore_run(path, apply=args.apply, items_meta=items_meta) for path in args.runs],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.apply:
        summary["completed_at"] = datetime.now(timezone.utc).isoformat()
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
