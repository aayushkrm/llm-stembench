#!/usr/bin/env python3
"""Validate and optionally publish the candidate dataset to Hugging Face.

All local hashes, metadata, version, and card checks run before authentication or
any remote operation. Each dataset version is published on its own branch; an
existing version branch is never changed unless ``--allow-overwrite`` is explicit.

  python scripts/publish_hf.py --repo-id <username>/stembench \
      --version v0.1.0-candidate --validate-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any


def _normalized_version(version: str) -> str:
    return version[1:] if version.startswith("v") else version


def validate_local_bundle(data: Path, card: Path, version: str) -> dict[str, Any]:
    """Validate all publication inputs without importing or calling Hugging Face."""
    if not re.fullmatch(r"v?[0-9][0-9A-Za-z._-]*", version):
        raise SystemExit(f"invalid version branch name: {version!r}")
    items = data / "items.jsonl"
    version_file = data / "DATASET_VERSION"
    report_file = data / "verification_report.json"
    missing = [path for path in (items, version_file, report_file, card) if not path.is_file()]
    if missing:
        raise SystemExit(f"publication input missing: {', '.join(str(path) for path in missing)}")

    meta = json.loads(version_file.read_text(encoding="utf-8"))
    report = json.loads(report_file.read_text(encoding="utf-8"))
    if _normalized_version(version) != _normalized_version(str(meta.get("version", ""))):
        raise SystemExit(
            f"requested version {version!r} != DATASET_VERSION {meta.get('version')!r}"
        )
    digest = hashlib.sha256(items.read_bytes()).hexdigest()
    if digest != meta.get("items_jsonl_sha256"):
        raise SystemExit("items.jsonl hash mismatch vs DATASET_VERSION — rebuild")
    if digest != report.get("checksums", {}).get("items_jsonl_sha256"):
        raise SystemExit("items.jsonl hash mismatch vs verification_report.json — rebuild")
    if "candidate" in _normalized_version(version).lower():
        card_text = card.read_text(encoding="utf-8").lower()
        if "candidate" not in card_text or "expert" not in card_text:
            raise SystemExit(
                "candidate dataset card must state candidate status and pending expert validation"
            )
    return {
        "items": items,
        "version_file": version_file,
        "report_file": report_file,
        "card": card,
        "meta": meta,
        "digest": digest,
    }


def ensure_version_branch(
    api: Any,
    repo_id: str,
    version: str,
    *,
    allow_overwrite: bool,
) -> bool:
    """Create a version branch or reject an existing one; return whether it existed."""
    refs = api.list_repo_refs(repo_id=repo_id, repo_type="dataset")
    branch_names = {ref.name for ref in refs.branches}
    existed = version in branch_names
    if existed and not allow_overwrite:
        raise SystemExit(
            f"dataset version branch {version!r} already exists; "
            "use --allow-overwrite only for an intentional replacement"
        )
    if not existed:
        api.create_branch(
            repo_id=repo_id,
            repo_type="dataset",
            branch=version,
            exist_ok=False,
        )
    return existed


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True, help="e.g. aayushkrm/stembench")
    ap.add_argument("--data", default="data/stembench_v1")
    ap.add_argument("--card", default="docs/dataset_card.md")
    ap.add_argument("--version", default="v0.1.0-candidate")
    ap.add_argument("--allow-overwrite", action="store_true")
    ap.add_argument(
        "--validate-only",
        action="store_true",
        help="perform every local release check without authentication or network calls",
    )
    args = ap.parse_args(argv)

    bundle = validate_local_bundle(Path(args.data), Path(args.card), args.version)
    digest = bundle["digest"]
    print(f"local bundle valid: version={args.version} sha256={digest}")
    if args.validate_only:
        return

    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import RepositoryNotFoundError
    except ImportError as exc:
        raise SystemExit("huggingface_hub not installed") from exc

    api = HfApi()
    who = api.whoami()
    print(f"authenticated as {who.get('name')}")
    try:
        api.repo_info(args.repo_id, repo_type="dataset")
        repo_exists = True
    except RepositoryNotFoundError:
        repo_exists = False

    if repo_exists:
        ensure_version_branch(
            api,
            args.repo_id,
            args.version,
            allow_overwrite=args.allow_overwrite,
        )
    else:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            exist_ok=False,
            private=False,
        )
        ensure_version_branch(api, args.repo_id, args.version, allow_overwrite=False)

    with tempfile.TemporaryDirectory(prefix="stembench_hf_") as temp_dir:
        staging = Path(temp_dir)
        shutil.copy(bundle["items"], staging / "items.jsonl")
        shutil.copy(bundle["version_file"], staging / "DATASET_VERSION")
        shutil.copy(bundle["report_file"], staging / "verification_report.json")
        shutil.copy(bundle["card"], staging / "README.md")
        commit = api.upload_folder(
            folder_path=str(staging),
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.version,
            commit_message=(
                f"STEMBench {args.version} (sha256 {digest[:16]}…) — "
                "automated verification complete; human expert validation pending"
            ),
        )

    commit_url = str(getattr(commit, "commit_url", ""))
    commit_oid = str(getattr(commit, "oid", ""))
    print(f"published {args.repo_id}@{args.version}: {commit_url or commit_oid}")
    print(f"dataset sha256: {digest}")
    notes = Path("releases")
    notes.mkdir(exist_ok=True)
    with open(notes / "RELEASE_NOTES.md", "a", encoding="utf-8") as file:
        file.write(
            f"- {args.version}: uploaded to {commit_url} "
            f"(commit {commit_oid}, sha256 {digest}, source seed {bundle['meta'].get('seed')})\n"
        )


if __name__ == "__main__":
    main()
