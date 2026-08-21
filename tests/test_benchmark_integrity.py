"""Negative regressions for benchmark verification and release integrity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.publish_hf import ensure_version_branch, validate_local_bundle
from stembench.benchmark_gen.qc import _check_solution_final
from stembench.benchmark_gen.verify import (
    v_econfig,
    v_empirical,
    v_lenses,
    v_reactions,
    verify_pair,
)
from stembench.schemas import AnswerType, BenchmarkItem, Choice, Language


def _item(**changes) -> BenchmarkItem:  # noqa: ANN003
    values = {
        "item_id": "TEST-0001-en",
        "pair_id": "TEST-0001",
        "language": Language.EN,
        "subject": "math",
        "topic": "audit fixture",
        "difficulty": "school",
        "difficulty_rubric": "fixture",
        "question": "Which answer is the intended answer?",
        "answer_type": AnswerType.MC,
        "choices": [
            Choice(label="A", text="10"),
            Choice(label="B", text="20"),
            Choice(label="C", text="30"),
            Choice(label="D", text="40"),
        ],
        "canonical_answer": "A",
        "solution": "1) Compute the value.\n2) Select it.\nAnswer: 10 (option A).",
    }
    values.update(changes)
    return BenchmarkItem(**values)


def test_candidate_binding_rejects_canonical_drift() -> None:
    records = verify_pair(
        "molar_mass",
        {"formula": "H2O", "expected": 18.015},
        candidate_canonical="0",
        answer_type=AnswerType.NUMERIC,
        candidate_numeric_value=0.0,
    )
    assert records[0].passed is True
    binding = next(record for record in records if record.method == "candidate_binding")
    assert binding.passed is False


def test_solution_final_requires_canonical_letter_and_choice() -> None:
    assert _check_solution_final(_item()) is None
    wrong_letter = _item(solution="1) Compute.\n2) Select.\nAnswer: 20 (option B).")
    assert "option B" in (_check_solution_final(wrong_letter) or "")
    wrong_display = _item(solution="1) Compute.\n2) Select.\nAnswer: 99 (option A).")
    assert "displayed answer" in (_check_solution_final(wrong_display) or "")


def test_empirical_verifier_requires_primitive_formula() -> None:
    params = {"pcts": {"C": 85.7, "H": 14.3}, "expected_text": "CH2"}
    assert v_empirical(params)[0] is True
    assert v_empirical({**params, "expected_text": "C2H4"})[0] is False


def test_econfig_verifier_checks_ground_state_not_only_count() -> None:
    assert v_econfig({"config": "[Ne] 3s^2", "expected_text": "Mg"})[0] is True
    assert v_econfig({"config": "[Ne] 3s^1 3p^1", "expected_text": "Mg"})[0] is False


def test_precipitate_verifier_checks_stoichiometry() -> None:
    base = {"kind": "precip", "r1": "BaCl2", "r2": "Na2SO4"}
    assert v_reactions({**base, "expected_text": "BaSO4"})[0] is True
    assert v_reactions({**base, "expected_text": "Ba2SO4"})[0] is False
    hydroxide = {"kind": "precip", "r1": "CuSO4", "r2": "NaOH"}
    assert v_reactions({**hydroxide, "expected_text": "Cu(OH)2"})[0] is True


def test_virtual_lens_detail_reports_actual_relation() -> None:
    passed, detail = v_lenses(
        {"kind": "real_virtual", "do": 10, "f": 12, "expected_text": "virtual"}
    )
    assert passed is True
    assert "d_o < f" in detail


def _publication_fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    data = tmp_path / "data"
    data.mkdir()
    payload = b'{"item_id":"fixture"}\n'
    (data / "items.jsonl").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (data / "DATASET_VERSION").write_text(
        json.dumps({"version": "0.1.0-candidate", "items_jsonl_sha256": digest}),
        encoding="utf-8",
    )
    (data / "verification_report.json").write_text(
        json.dumps({"checksums": {"items_jsonl_sha256": digest}}), encoding="utf-8"
    )
    card = tmp_path / "README.md"
    card.write_text("Candidate dataset; independent expert validation is pending.", encoding="utf-8")
    return data, card, digest


def test_publication_validation_uses_built_checksum_key(tmp_path: Path) -> None:
    data, card, digest = _publication_fixture(tmp_path)
    result = validate_local_bundle(data, card, "v0.1.0-candidate")
    assert result["digest"] == digest

    (data / "items.jsonl").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="DATASET_VERSION"):
        validate_local_bundle(data, card, "v0.1.0-candidate")


class _FakeApi:
    def __init__(self, branches: list[str]) -> None:
        self.refs = SimpleNamespace(branches=[SimpleNamespace(name=name) for name in branches])
        self.created: list[str] = []

    def list_repo_refs(self, **_kwargs):  # noqa: ANN003, ANN201
        return self.refs

    def create_branch(self, *, branch: str, **_kwargs) -> None:  # noqa: ANN003
        self.created.append(branch)


def test_publication_version_guard_refuses_existing_branch() -> None:
    api = _FakeApi(["main", "v0.1.0-candidate"])
    with pytest.raises(SystemExit, match="already exists"):
        ensure_version_branch(
            api, "owner/dataset", "v0.1.0-candidate", allow_overwrite=False
        )
    assert ensure_version_branch(
        api, "owner/dataset", "v0.1.0-candidate", allow_overwrite=True
    ) is True
    assert api.created == []


def test_publication_version_guard_creates_new_branch() -> None:
    api = _FakeApi(["main"])
    assert ensure_version_branch(
        api, "owner/dataset", "v0.1.0-candidate", allow_overwrite=False
    ) is False
    assert api.created == ["v0.1.0-candidate"]
