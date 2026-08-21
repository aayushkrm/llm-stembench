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


def _score(record: dict[str, Any]) -> tuple[bool | None, str, str]:
    answer_type = record.get("answer_type")
    reference = record.get("reference_answer", "")
    raw = record.get("raw_response", "")
    if answer_type == "mc":
        if len(reference) != 1 or not reference.upper().isalpha():
            raise ValueError(f"invalid MC reference {reference!r}")
        n_choices = int(record.get("extra", {}).get("n_choices", 4))
        return score_mc(raw, ord(reference.upper()) - ord("A"), n_choices=n_choices)
    if answer_type == "exact":
        alternatives = record.get("extra", {}).get("acceptable_answers")
        return score_exact(raw, reference, alternatives=alternatives)
    if answer_type == "numeric":
        extra = record.get("extra", {})
        return score_numeric(
            raw,
            float(reference),
            rel_tol=extra.get("rel_tol"),
            abs_tol=extra.get("abs_tol"),
            require_unit=extra.get("unit", ""),
        )
    raise ValueError(f"unsupported answer_type {answer_type!r}")


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


def rescore_run(run_dir: Path, *, apply: bool) -> dict[str, Any]:
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
            ok, parsed, method = _score(record)
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
        "--manifest",
        type=Path,
        default=Path("results/stage1/rescore_manifest.json"),
        help="summary path (written only with --apply)",
    )
    args = parser.parse_args()

    summary = {
        "schema_version": 1,
        "operation": "offline_rescore_from_stored_raw_responses",
        "parser_change": "MC answer letter must not be part of a longer word",
        "provider_calls": 0,
        "applied": bool(args.apply),
        "runs": [rescore_run(path, apply=args.apply) for path in args.runs],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.apply:
        summary["completed_at"] = datetime.now(timezone.utc).isoformat()
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
