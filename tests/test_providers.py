"""No-network provider tests: FakeProvider, free-model enforcement, budget tracker,
shared state, registry, and OpenAICompatProvider.complete against a stub HTTP transport.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pytest

from stembench.providers.base import Completion, DailyBudgetExceeded, ProviderError
from stembench.providers.fake import FakeProvider
from stembench.providers.openai_compat import (
    BudgetTracker,
    OpenAICompatProvider,
    _shared_state,
)
from stembench.providers.registry import (
    OPENROUTER_FREE,
    ZEN_FREE,
    build_provider,
)


# --------------------------------------------------------------------------
# FakeProvider: deterministic, records calls
# --------------------------------------------------------------------------
def test_fake_provider_deterministic():
    p = FakeProvider()
    messages = [{"role": "user", "content": "Answer: A"}]
    c1: Completion = p.complete("fake/tiny-test", messages, max_tokens=64)
    c2: Completion = p.complete("fake/tiny-test", messages, max_tokens=64)
    # content is a pure function of (model, prompt) -> identical Completions
    assert c1 == c2
    # both calls were recorded as (model, last prompt)
    assert p.calls == [("fake/tiny-test", "Answer: A")] * 2
    # synthetic-marker fields
    assert c1.raw == {"synthetic": True}
    assert c1.cost == 0.0
    assert c1.model_reported == "fake/tiny-test"
    assert c1.finish_reason == "stop"
    assert c1.usage["total_tokens"] == 120
    assert p.supports_logprobs("fake/tiny-test") is True


def test_fake_provider_answers_parseable():
    # whatever the hash decides, the emitted content must carry the contract
    p = FakeProvider()
    c = p.complete("fake/tiny-test", [{"role": "user", "content": "prompt"}], max_tokens=16)
    assert "\nAnswer: " in c.content
    assert "\nConfidence: " in c.content


# --------------------------------------------------------------------------
# Free-model enforcement
# --------------------------------------------------------------------------
def test_assert_free_refuses_paid_models():
    prov = OpenAICompatProvider(
        name="assertfree-t", base_url="https://stub.invalid/v1",
        api_key_env="NOPE", free_models=["m-free"], requests_per_minute=600,
    )
    with pytest.raises(ProviderError, match="not on the free allowlist"):
        prov._assert_free("paid-model")
    # allowlisted model passes silently
    prov._assert_free("m-free")


def test_registry_provider_refuses_paid_models():
    p = build_provider("openrouter")
    with pytest.raises(ProviderError, match="zero-spend"):
        p._assert_free("paid-model")
    p._assert_free("openai/gpt-oss-20b:free")  # allowlisted: no raise


# --------------------------------------------------------------------------
# BudgetTracker
# --------------------------------------------------------------------------
def test_budget_tracker_fresh_dir_used_zero(tmp_path):
    t = BudgetTracker("fresh", 5, path=tmp_path / "budget_fresh.json")
    # no file on disk -> count 0
    assert t.used_today() == 0


def test_budget_tracker_reserve_until_cap(tmp_path):
    t = BudgetTracker("cap", 3, path=tmp_path / "budget_cap.json")
    t.reserve()
    t.reserve()
    t.reserve()
    assert t.used_today() == 3
    # the 4th reserve exceeds the daily cap of 3
    with pytest.raises(DailyBudgetExceeded):
        t.reserve()
    assert t.used_today() == 3  # failed reserve did not bump the counter


def test_budget_tracker_date_rollover_resets(tmp_path):
    path = tmp_path / "budget_roll.json"
    # a leftover counter from yesterday must not count against today
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    path.write_text(json.dumps({"date": yesterday, "count": 99, "requests": []}))
    t = BudgetTracker("roll", 10, path=path)
    assert t.used_today() == 0
    t.reserve()
    # today's counter starts from 0 and now holds 1 (not 100)
    assert t.used_today() == 1
    data = json.loads(path.read_text())
    assert data["date"] == date.today().isoformat()
    assert data["count"] == 1


def test_budget_tracker_record_success(tmp_path):
    t = BudgetTracker("succ", 5, path=tmp_path / "budget_succ.json")
    t.reserve()
    t.record_success("m-free", 0.5)
    data = json.loads((tmp_path / "budget_succ.json").read_text())
    assert data["count"] == 1
    assert data["requests"] == [{"model": "m-free", "cost": 0.5, "ts": data["requests"][0]["ts"]}]


def test_budget_tracker_concurrent_reserve_exact_cap(tmp_path):
    # 8 threads reserve against a cap of 5 -> exactly 5 succeed, 3 are refused
    t = BudgetTracker("conc", 5, path=tmp_path / "budget_conc.json")

    def try_reserve(_):
        try:
            t.reserve()
            return "ok"
        except DailyBudgetExceeded:
            return "exceeded"

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(try_reserve, range(8)))
    assert results.count("ok") == 5
    assert results.count("exceeded") == 3
    assert t.used_today() == 5


# --------------------------------------------------------------------------
# Shared per-provider state
# --------------------------------------------------------------------------
def test_shared_state_same_name_returns_same_tracker_and_lock():
    t1, lock1, _ = _shared_state("shared-uniq-a", None, 0.0)
    t2, lock2, _ = _shared_state("shared-uniq-a", 7, 0.0)
    assert t1 is t2
    assert lock1 is lock2
    # re-requesting with a cap updates the shared tracker's cap
    assert t2.daily_cap == 7
    # a different provider name gets a different tracker
    t3, _, _ = _shared_state("shared-uniq-b", None, 0.0)
    assert t3 is not t1


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
def test_registry_builds_known_providers(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-not-used")
    monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "dummy-not-used")

    assert isinstance(build_provider("fake"), FakeProvider)

    orp = build_provider("openrouter")
    assert isinstance(orp, OpenAICompatProvider)
    assert orp.name == "openrouter"
    assert orp.base_url == "https://openrouter.ai/api/v1"
    assert orp.free_models == set(OPENROUTER_FREE)
    orp.close()

    zp = build_provider("zen")
    assert isinstance(zp, OpenAICompatProvider)
    assert zp.name == "zen"
    assert zp.free_models == set(ZEN_FREE)
    zp.close()


def test_registry_unknown_provider_raises_keyerror():
    with pytest.raises(KeyError, match="unknown provider"):
        build_provider("no-such-provider")


def test_registry_free_model_allowlists():
    assert "openai/gpt-oss-20b:free" in OPENROUTER_FREE
    assert "nemotron-3.5-lightning-free" in ZEN_FREE
    # every OpenRouter free model is suffixed ":free" (zero-spend tier)
    assert all(m.endswith(":free") for m in OPENROUTER_FREE)


# --------------------------------------------------------------------------
# OpenAICompatProvider.complete against a stub transport
# --------------------------------------------------------------------------
class _StubResponse:
    def __init__(self, status_code: int, data: dict, headers: dict | None = None):
        self.status_code = status_code
        self._data = data
        self.text = json.dumps(data)
        self.headers = headers or {}

    def json(self) -> dict:
        return self._data


class _StubClient:
    """Queued-response httpx.Client stand-in; records every post() call."""

    def __init__(self, responses: list[_StubResponse]):
        self._responses = list(responses)
        self.posts: list[dict] = []

    def post(self, url, headers=None, json=None):  # noqa: A002 (mirrors httpx API)
        self.posts.append({"url": url, "headers": headers, "json": json})
        return self._responses.pop(0)


CANNED_200 = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "model": "stub-echo-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Answer: B"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    "cost": {"total": 0.0},
}


def _make_provider(name: str, **kw) -> OpenAICompatProvider:
    defaults = dict(
        name=name,
        base_url="https://stub.invalid/v1",
        api_key_env="STUBPROV_API_KEY",
        free_models=["m-free"],
        requests_per_minute=600,  # min spacing 0.1 s; first request never waits
        daily_cap=None,
    )
    defaults.update(kw)
    return OpenAICompatProvider(**defaults)


def _complete_with_stub(monkeypatch, provider, responses):
    stub = _StubClient(responses)
    monkeypatch.setattr(provider, "_client_http", lambda: stub)
    monkeypatch.setattr(time, "sleep", lambda _s: None)  # no real backoff waits
    monkeypatch.setenv("STUBPROV_API_KEY", "stub-dummy-key")
    comp = provider.complete(
        model="m-free",
        messages=[{"role": "user", "content": "What is 2+2?"}],
        max_tokens=100,
    )
    return comp, stub


def test_complete_maps_response_fields(monkeypatch):
    prov = _make_provider("stubmap-t1")
    comp, stub = _complete_with_stub(monkeypatch, prov, [_StubResponse(200, CANNED_200)])
    # field mapping from the canned chat-completion JSON
    assert comp.content == "Answer: B"
    assert comp.finish_reason == "stop"
    assert comp.usage == {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
    assert comp.model_reported == "stub-echo-model"
    assert comp.cost == 0.0  # usage has no "cost" -> data["cost"]["total"] is used
    assert comp.logprobs is None
    assert comp.raw == CANNED_200
    # exactly one POST to /chat/completions with the auth header and payload
    assert len(stub.posts) == 1
    assert stub.posts[0]["url"] == "https://stub.invalid/v1/chat/completions"
    assert stub.posts[0]["headers"]["Authorization"] == "Bearer stub-dummy-key"
    assert stub.posts[0]["json"]["model"] == "m-free"
    assert stub.posts[0]["json"]["max_tokens"] == 100
    assert stub.posts[0]["json"]["temperature"] == 0.0
    # the request was counted against the (isolated) budget and recorded
    data = json.loads(prov.budget.path.read_text())
    assert data["count"] == 1
    assert data["requests"][0]["model"] == "m-free"
    assert data["requests"][0]["cost"] == 0.0
    prov.close()


def test_complete_rejects_nonzero_cost_zero_spend(monkeypatch):
    # Zero-spend guard: a provider-reported nonzero cost aborts the call even for
    # an allowlisted "free" model (protects against mislabeled/misrouted models).
    prov = _make_provider("stubpaid-t7")
    canned = json.loads(json.dumps(CANNED_200))
    canned["cost"] = {"total": 0.0123}
    with pytest.raises(ProviderError, match="zero-spend"):
        _complete_with_stub(monkeypatch, prov, [_StubResponse(200, canned)])
    prov.close()


def test_complete_retries_once_after_429(monkeypatch):
    prov = _make_provider("stubretry-t2")
    responses = [
        _StubResponse(429, {"error": "rate limited"}),
        _StubResponse(200, CANNED_200),
    ]
    comp, stub = _complete_with_stub(monkeypatch, prov, responses)
    # 429 then 200: succeeds on the second attempt with two POSTs total
    assert comp.content == "Answer: B"
    assert len(stub.posts) == 2


def test_complete_three_429s_exhausts_retries(monkeypatch):
    prov = _make_provider("stubfail-t3")
    responses = [_StubResponse(429, {"error": "rate limited"}) for _ in range(3)]
    stub = _StubClient(responses)
    monkeypatch.setattr(prov, "_client_http", lambda: stub)
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    monkeypatch.setenv("STUBPROV_API_KEY", "stub-dummy-key")
    with pytest.raises(ProviderError, match="retries exhausted"):
        prov.complete(model="m-free", messages=[{"role": "user", "content": "q"}],
                      max_tokens=10)
    assert len(stub.posts) == 3  # exactly 3 attempts
    prov.close()


def test_complete_400_fails_fast_without_retry(monkeypatch):
    prov = _make_provider("stub400-t4")
    stub = _StubClient([_StubResponse(400, {"error": {"message": "bad request"}})])
    monkeypatch.setattr(prov, "_client_http", lambda: stub)
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    monkeypatch.setenv("STUBPROV_API_KEY", "stub-dummy-key")
    with pytest.raises(ProviderError, match="400"):
        prov.complete(model="m-free", messages=[{"role": "user", "content": "q"}],
                      max_tokens=10)
    # client errors are fatal: exactly one POST, no retries
    assert len(stub.posts) == 1
    prov.close()


def test_complete_logprobs_requested_only_for_capable_models(monkeypatch):
    monkeypatch.setenv("STUBPROV_API_KEY", "stub-dummy-key")
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    capable = _make_provider("stublp-t5", logprob_models={"m-free"})
    stub1 = _StubClient([_StubResponse(200, CANNED_200)])
    monkeypatch.setattr(capable, "_client_http", lambda: stub1)
    assert capable.supports_logprobs("m-free") is True
    capable.complete(model="m-free", messages=[{"role": "user", "content": "q"}],
                     max_tokens=10)
    assert stub1.posts[0]["json"]["logprobs"] is True
    assert stub1.posts[0]["json"]["top_logprobs"] == 5
    capable.close()

    incapable = _make_provider("stublp-t6", logprob_models=set())
    stub2 = _StubClient([_StubResponse(200, CANNED_200)])
    monkeypatch.setattr(incapable, "_client_http", lambda: stub2)
    assert incapable.supports_logprobs("m-free") is False
    incapable.complete(model="m-free", messages=[{"role": "user", "content": "q"}],
                       max_tokens=10)
    assert "logprobs" not in stub2.posts[0]["json"]
    incapable.close()


def test_complete_missing_api_key_raises(monkeypatch):
    prov = _make_provider("stubkey-t7")
    stub = _StubClient([_StubResponse(200, CANNED_200)])
    monkeypatch.setattr(prov, "_client_http", lambda: stub)
    monkeypatch.delenv("STUBPROV_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="missing env var STUBPROV_API_KEY"):
        prov.complete(model="m-free", messages=[{"role": "user", "content": "q"}],
                      max_tokens=10)
    prov.close()


def test_complete_refuses_non_free_model_before_any_request(monkeypatch):
    prov = _make_provider("stubpaid-t8")
    stub = _StubClient([])
    monkeypatch.setattr(prov, "_client_http", lambda: stub)
    monkeypatch.setenv("STUBPROV_API_KEY", "stub-dummy-key")
    with pytest.raises(ProviderError, match="not on the free allowlist"):
        prov.complete(model="paid-model", messages=[{"role": "user", "content": "q"}],
                      max_tokens=10)
    assert stub.posts == []  # refused before touching the network
    prov.close()
