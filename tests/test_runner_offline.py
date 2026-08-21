"""Offline end-to-end runner tests (FakeProvider, dry_run) plus direct prompt/score
helpers and the offline-safe parts of the mmlu_stem dataset module
(stratified_sample, STEM_SUBJECTS -- load_mmlu_stem downloads from HF and is never
called here).

KNOWN GENUINE SRC BUG documented by an intentionally failing test:
- BUG-D1: stratified_sample (src/stembench/datasets/mmlu_stem.py:92-99) hangs in an
  infinite loop when n > len(items): every stratum is already full so `remaining`
  can never decrease while the `while remaining > 0` loop keeps cycling.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from stembench import runner
from stembench.datasets.mmlu_stem import STEM_SUBJECTS, stratified_sample
from stembench.prompts import template_hash
from stembench.schemas import (
    AnswerType,
    FreeResponseItem,
    Language,
    MCItem,
    ModelSpec,
    ResponseRecord,
    RunConfig,
    SampleSpec,
    Tolerance,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

SUBJECTS = ("physics", "chemistry", "math")


def _synthetic_items(n: int = 10) -> list[MCItem]:
    return [
        MCItem(
            item_id=f"it{i:02d}",
            dataset="mmlu_stem",
            subject=SUBJECTS[i % 3],
            question=f"Synthetic question {i}?",
            choices=["choice a", "choice b", "choice c", "choice d"],
            gold=i % 4,
        )
        for i in range(n)
    ]


def _make_config(tmp_path: Path) -> RunConfig:
    return RunConfig(
        run_id="TEST-E2E",
        stage=1,
        dataset="mmlu_stem",
        sample=SampleSpec(n=10, seed=42),
        # provider "openrouter" in the spec proves dry_run swaps in the fake provider
        models=[ModelSpec(provider="openrouter", model="openai/gpt-oss-20b:free")],
        languages=[Language.EN],
        output_dir=str(tmp_path / "results"),
        dry_run=True,
        notes="offline e2e test",
    )


# --------------------------------------------------------------------------
# git_state
# --------------------------------------------------------------------------
def test_git_state_returns_string_tuple():
    commit, dirty = runner.git_state(REPO_ROOT)
    assert isinstance(commit, str)
    assert isinstance(dirty, bool)
    # this repository is a git checkout -> a full 40-hex commit hash
    assert re.fullmatch(r"[0-9a-f]{40}", commit)


def test_git_state_non_repo_is_degenerate(tmp_path):
    # outside a repo the helper degrades to ("", False) instead of raising
    assert runner.git_state(tmp_path) == ("", False)


# --------------------------------------------------------------------------
# _build_prompt / _score_item on schema objects
# --------------------------------------------------------------------------
def test_build_prompt_mc():
    item = _synthetic_items(1)[0]
    prompt = runner._build_prompt(item)
    assert "Synthetic question 0?" in prompt
    for label in ("A.", "B.", "C.", "D."):
        assert label in prompt
    assert "choice a" in prompt and "choice d" in prompt
    assert "Answer:" in prompt and "Confidence:" in prompt


def test_build_prompt_mc_russian():
    item = MCItem(
        item_id="ru-1", dataset="mmlu_stem", subject="physics",
        language=Language.RU, question="Сколько будет 2+2?",
        choices=["3", "4", "5", "6"], gold=1,
    )
    prompt = runner._build_prompt(item)
    assert "Сколько будет 2+2?" in prompt
    assert "Решите" in prompt and "Задача:" in prompt  # RU template selected


def test_build_prompt_free_response():
    item = FreeResponseItem(
        item_id="fr-1", dataset="stembench_v1", subject="chemistry",
        question="What is the molar mass of water?", gold="18 g/mol",
    )
    prompt = runner._build_prompt(item)
    assert "What is the molar mass of water?" in prompt
    assert "Answer:" in prompt and "Confidence:" in prompt


def test_score_item_mc():
    item = MCItem(item_id="m1", dataset="d", subject="physics",
                  question="q", choices=["a", "b", "c", "d"], gold=1)
    # gold index 1 = letter B
    assert runner._score_item(item, "Answer: B") == (True, "B", "pattern1")
    assert runner._score_item(item, "Answer: C")[0] is False
    assert runner._score_item(item, "I refuse to answer.") == (None, "", "none")


def test_score_item_exact():
    item = FreeResponseItem(
        item_id="f1", dataset="d", subject="astronomy",
        question="Which planet is the red one?", gold="Mars",
        alternatives=["the red planet"],
    )
    ok, parsed, method = runner._score_item(item, "The answer is Mars.\nConfidence: 90")
    assert ok is True
    assert parsed == "Mars."
    assert method == "exact"
    # alternative accepted
    assert runner._score_item(item, "Answer: The Red Planet")[0] is True
    # wrong
    assert runner._score_item(item, "Answer: Venus")[0] is False


def test_score_item_numeric():
    item = FreeResponseItem(
        item_id="n1", dataset="d", subject="physics",
        question="g on Earth?", answer_type=AnswerType.NUMERIC,
        gold="9.81", tolerance=Tolerance(rel=0.02), units="",
    )
    # |10 - 9.81| = 0.19 <= 0.02 * 9.81 = 0.1962 -> accept
    ok, raw, method = runner._score_item(item, "Answer: 10")
    assert ok is True
    assert raw == "10"
    assert method.startswith("numeric(")
    # |12 - 9.81| = 2.19 > 0.1962 -> reject
    assert runner._score_item(item, "Answer: 12")[0] is False


# --------------------------------------------------------------------------
# Full offline run loop (dry_run + FakeProvider)
# --------------------------------------------------------------------------
def test_run_end_to_end_offline(monkeypatch, tmp_path):
    items = _synthetic_items(10)
    monkeypatch.setattr(
        runner, "_load_items_for_run", lambda cfg: (items, {"revision": "test-rev"})
    )
    # budget files must land in an isolated tmp dir, never data/cache
    monkeypatch.setenv("STEMBENCH_BUDGET_DIR", str(tmp_path / "budget"))

    result = runner.run(_make_config(tmp_path), repo_root=REPO_ROOT)
    run_dir = tmp_path / "results" / "stage1" / "TEST-E2E"
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "manifest_start.json").exists()

    # records file named {provider}__{safe model}.jsonl for the *spec* provider
    records_path = run_dir / "openrouter__openai_gpt-oss-20b_free.jsonl"
    assert records_path.exists()
    lines = [ln for ln in records_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 10  # one record per item

    for line in lines:
        rec = ResponseRecord.model_validate(json.loads(line))  # must validate
        assert rec.run_id == "TEST-E2E"
        assert rec.provider == "fake"  # dry_run stamps the fake provider
        assert rec.model == "openai/gpt-oss-20b:free"
        assert rec.dataset == "mmlu_stem"
        assert rec.answer_type == "mc"
        assert rec.reference_answer in ("A", "B", "C", "D")
        assert rec.correctness is not None  # fake output always carries a letter
        assert rec.self_reported_confidence is not None
        assert rec.prompt_hash == template_hash("mc_answer_confidence_v1", "en")

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == "TEST-E2E"
    assert manifest["status"] == "complete"
    assert manifest["counts"] == {"n_items": 10, "n_models": 1, "total_evaluated": 10}
    m0 = manifest["models"][0]
    assert m0["n_new_attempted"] == 10
    assert m0["n_new_completed"] == 10
    assert m0["n_total_evaluated"] == 10
    # correct + incorrect + parse_failures == evaluated (internal consistency)
    assert (
        m0["n_total_correct"]
        + (m0["n_total_evaluated"] - m0["n_total_correct"] - m0["n_total_parse_failures"])
        == 10
    )
    # manifest totals match the record file
    n_correct_in_file = sum(1 for ln in lines if json.loads(ln)["correctness"] is True)
    assert n_correct_in_file == m0["n_total_correct"]
    # dry run: no real provider requests were budgeted
    assert manifest["budget_usage"]["openrouter_used_today"] in (0, None)
    assert result["status"] == "complete"


def test_run_resume_makes_no_new_attempts(monkeypatch, tmp_path):
    items = _synthetic_items(10)
    monkeypatch.setattr(
        runner, "_load_items_for_run", lambda cfg: (items, {"revision": "test-rev"})
    )
    monkeypatch.setenv("STEMBENCH_BUDGET_DIR", str(tmp_path / "budget"))
    config = _make_config(tmp_path)

    runner.run(config, repo_root=REPO_ROOT)
    records_path = tmp_path / "results" / "stage1" / "TEST-E2E" / (
        "openrouter__openai_gpt-oss-20b_free.jsonl"
    )
    first = records_path.read_text(encoding="utf-8")

    result2 = runner.run(config, repo_root=REPO_ROOT)
    m0 = result2["models"][0]
    # every item was already on disk with a final outcome -> nothing re-attempted
    assert m0["n_already_done"] == 10
    assert m0["n_new_attempted"] == 0
    assert m0["n_new_completed"] == 0
    assert m0["status"] == "complete"
    # the records file is byte-identical: no duplicate requests, no duplicate lines
    assert records_path.read_text(encoding="utf-8") == first


def test_run_resume_retries_rate_limited_item_only(monkeypatch, tmp_path):
    items = _synthetic_items(10)
    monkeypatch.setattr(
        runner, "_load_items_for_run", lambda cfg: (items, {"revision": "test-rev"})
    )
    monkeypatch.setenv("STEMBENCH_BUDGET_DIR", str(tmp_path / "budget"))
    config = _make_config(tmp_path)
    runner.run(config, repo_root=REPO_ROOT)

    records_path = tmp_path / "results" / "stage1" / "TEST-E2E" / (
        "openrouter__openai_gpt-oss-20b_free.jsonl"
    )
    recs = [json.loads(ln) for ln in records_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()]
    # simulate a transient failure for item 4: transient statuses are retried on resume
    recs[4]["error_status"] = "rate_limited"
    recs[4]["correctness"] = None
    records_path.write_text(
        "\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8"
    )

    result3 = runner.run(config, repo_root=REPO_ROOT)
    m0 = result3["models"][0]
    assert m0["n_already_done"] == 9
    assert m0["n_new_attempted"] == 1
    assert m0["n_new_completed"] == 1
    # file now holds 10 originals + 1 retry
    new_recs = [json.loads(ln) for ln in records_path.read_text(encoding="utf-8").splitlines()
                if ln.strip()]
    assert len(new_recs) == 11
    retried = [r for r in new_recs if r["item_id"] == recs[4]["item_id"]]
    assert len(retried) == 2
    # the appended retry has a real outcome (fake provider output parses)
    assert retried[-1]["error_status"] == ""
    assert retried[-1]["correctness"] is not None


# --------------------------------------------------------------------------
# Dataset helpers (offline-safe): stratified_sample + STEM_SUBJECTS
# --------------------------------------------------------------------------
def _sample_items() -> list[MCItem]:
    # strata: alpha=6 items, beta=3, gamma=2 (total 11)
    items = []
    for subj, cnt in (("alpha", 6), ("beta", 3), ("gamma", 2)):
        for j in range(cnt):
            items.append(MCItem(item_id=f"{subj}-{j}", dataset="t", subject=subj,
                                question="q", choices=["a", "b", "c", "d"], gold=0))
    return items


def test_stratified_sample_allocation_hand_computed():
    # n=5 of 11 items (6/3/2 per stratum), hand-computed largest-remainder allocation:
    #   raw = len*n/total: alpha 30/11 = 2.727 -> floor 2
    #                        beta  15/11 = 1.364 -> floor 1
    #                        gamma 10/11 = 0.909 -> floor 0
    #   floors sum to 3, remaining = 2; remainders (0.727, 0.364, 0.909) have a
    #   strict order gamma > alpha > beta, so gamma and alpha each gain one:
    #   final allocation alpha=3, beta=1, gamma=1 (total 5)
    out = stratified_sample(_sample_items(), n=5, seed=7)
    assert len(out) == 5
    counts = {"alpha": 0, "beta": 0, "gamma": 0}
    for it in out:
        counts[it.subject] += 1
    assert counts == {"alpha": 3, "beta": 1, "gamma": 1}
    # deterministic: same seed -> same item set in the same (id-sorted) order
    out2 = stratified_sample(_sample_items(), n=5, seed=7)
    assert [i.item_id for i in out] == [i.item_id for i in out2]
    # output is sorted by item_id (run-stable ordering)
    assert [i.item_id for i in out] == sorted(i.item_id for i in out)


def test_stratified_sample_edges():
    items = _sample_items()
    # n = 0 -> empty sample
    assert stratified_sample(items, n=0, seed=1) == []
    # n = total -> every item exactly once (allocation 6/3/2 needs no remainder)
    all_out = stratified_sample(items, n=11, seed=1)
    assert len(all_out) == 11
    assert sorted(i.item_id for i in all_out) == sorted(i.item_id for i in items)
    # empty input
    assert stratified_sample([], n=5, seed=1) == []


def test_stratified_sample_n_greater_than_total_returns_all():
    # BUG-D1 (kept failing): sampling more items than exist must return every item.
    # Fixture: strata alpha=6, beta=3, gamma=2 (11 items), n=100 -> raw allocations
    # 54.545->54, 27.27->27, 18.18->18 (sum 99, remaining 1), but every stratum is
    # already fuller than its size, so the largest-remainder loop
    # (mmlu_stem.py:94-99) can never decrement `remaining` and spins forever.
    # Run in a subprocess with a timeout so the suite cannot hang on the bug.
    code = textwrap.dedent(
        """
        from stembench.datasets.mmlu_stem import stratified_sample
        from stembench.schemas import MCItem
        items = []
        for subj, cnt in (("alpha", 6), ("beta", 3), ("gamma", 2)):
            for j in range(cnt):
                items.append(MCItem(item_id=f"{subj}-{j}", dataset="t", subject=subj,
                                    question="q", choices=["a", "b"], gold=0))
        out = stratified_sample(items, 100, seed=1)
        assert len(out) == 11
        """
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=8
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            "stratified_sample hangs forever when n > len(items) "
            "(src/stembench/datasets/mmlu_stem.py:92-99)"
        )
    assert proc.returncode == 0, proc.stderr


def test_stem_subjects():
    # 21 official MMLU STEM subjects, snake_case identifiers
    assert len(STEM_SUBJECTS) == 21
    for expected in ("college_physics", "high_school_mathematics", "machine_learning"):
        assert expected in STEM_SUBJECTS
    assert all(re.fullmatch(r"[a-z_]+", s) for s in STEM_SUBJECTS)
