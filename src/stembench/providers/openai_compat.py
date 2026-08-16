"""OpenAI-compatible chat-completions provider (OpenRouter, Opencode Zen).

Safety properties:
- hard free-model allowlist: refuses any model not listed as free (zero-spend guarantee);
- persistent per-day request counter enforcing a documented daily cap;
- client-side rate spacing (min interval between request starts);
- bounded retries with exponential backoff on 429/5xx/timeouts;
- never logs or stores credentials; auth header is set per-request, not serialized.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import date
from pathlib import Path
from typing import Any, Optional

import httpx

from stembench.providers.base import Completion, DailyBudgetExceeded, Provider, ProviderError

BUDGET_DIR = Path(os.environ.get("STEMBENCH_BUDGET_DIR", "data/cache"))

# Shared per-provider state: one budget tracker + one rate spacer per provider name,
# so parallel model workers against the same provider respect a single cap/rhythm.
_SHARED: dict[str, tuple["BudgetTracker", threading.Lock, list[float]]] = {}
_SHARED_LOCK = threading.Lock()


def _shared_state(name: str, daily_cap: Optional[int], min_interval: float):
    with _SHARED_LOCK:
        if name not in _SHARED:
            _SHARED[name] = (BudgetTracker(name, daily_cap), threading.Lock(), [0.0])
        tracker, lock, last = _SHARED[name]
        if daily_cap is not None:
            tracker.daily_cap = daily_cap
        return tracker, lock, last


class BudgetTracker:
    """Persistent per-provider daily request counter (thread-safe)."""

    def __init__(self, provider: str, daily_cap: Optional[int], path: Path | None = None):
        self.provider = provider
        self.daily_cap = daily_cap
        self.path = path or (BUDGET_DIR / f"budget_{provider}.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _read(self) -> dict[str, Any]:
        today = date.today().isoformat()
        try:
            data = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            data = {}
        if data.get("date") != today:
            data = {}
        return {
            "date": today,
            "count": int(data.get("count", 0)),
            "requests": data.get("requests", []),
        }

    def used_today(self) -> int:
        with self._lock:
            return int(self._read().get("count", 0))

    def reserve(self) -> None:
        """Count one request against today's budget or raise DailyBudgetExceeded."""
        with self._lock:
            data = self._read()
            if self.daily_cap is not None and data["count"] >= self.daily_cap:
                raise DailyBudgetExceeded(
                    f"{self.provider}: daily cap {self.daily_cap} reached "
                    f"({data['count']} used today)"
                )
            data["count"] += 1
            self.path.write_text(json.dumps(data, ensure_ascii=False, indent=1))

    def record_success(self, model: str, cost: Optional[float]) -> None:
        with self._lock:
            data = self._read()
            data.setdefault("requests", []).append(
                {"model": model, "cost": cost, "ts": time.time()}
            )
            self.path.write_text(json.dumps(data, ensure_ascii=False, indent=1))


class OpenAICompatProvider(Provider):
    def __init__(
        self,
        name: str,
        base_url: str,
        api_key_env: str,
        free_models: list[str],
        logprob_models: set[str] | None = None,
        requests_per_minute: int = 20,
        daily_cap: Optional[int] = None,
        timeout_s: float = 180.0,
        extra_headers: dict[str, str] | None = None,
    ):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.free_models = set(free_models)
        self.logprob_models = logprob_models or set()
        self._min_interval = 60.0 / requests_per_minute
        self.budget, self._spacing_lock, self._last_request_ts = _shared_state(
            name, daily_cap, self._min_interval
        )
        self.timeout_s = timeout_s
        self.extra_headers = extra_headers or {}
        self._client: httpx.Client | None = None

    # -- helpers ---------------------------------------------------------
    def _api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise ProviderError(f"missing env var {self.api_key_env} for {self.name}")
        return key

    def _assert_free(self, model: str) -> None:
        if model not in self.free_models:
            raise ProviderError(
                f"{self.name}: model '{model}' is not on the free allowlist; refusing "
                "to spend (zero-spend policy). Free models: "
                + ", ".join(sorted(self.free_models))
            )

    def _client_http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout_s)
        return self._client

    def _space_requests(self) -> None:
        with self._spacing_lock:
            wait = self._last_request_ts[0] + self._min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_request_ts[0] = time.monotonic()

    def supports_logprobs(self, model: str) -> bool:
        return model in self.logprob_models

    # -- main ------------------------------------------------------------
    def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
        request_logprobs: bool = True,
    ) -> Completion:
        self._assert_free(model)
        self.budget.reserve()
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if top_p is not None:
            payload["top_p"] = top_p
        if seed is not None:
            payload["seed"] = seed
        want_logprobs = request_logprobs and self.supports_logprobs(model)
        if want_logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = 5

        last_err: Exception | None = None
        for attempt in range(3):
            self._space_requests()
            try:
                resp = self._client_http().post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key()}",
                        "Content-Type": "application/json",
                        **self.extra_headers,
                    },
                    json=payload,
                )
                if resp.status_code == 429:
                    raise ProviderError(f"429 rate limited: {resp.text[:200]}")
                if resp.status_code >= 500:
                    raise ProviderError(f"{resp.status_code} server error")
                if resp.status_code >= 400:
                    # client errors other than auth: record and give up fast
                    raise _FatalClientError(f"{resp.status_code}: {resp.text[:300]}")
                data = resp.json()
                break
            except _FatalClientError as e:  # bad request, wrong model id, auth
                raise ProviderError(str(e)) from e
            except (httpx.TimeoutException, httpx.TransportError, ProviderError) as e:
                last_err = e
                time.sleep(min(2**attempt * 5, 30))
        else:
            raise ProviderError(f"{self.name}/{model}: retries exhausted: {last_err}")

        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = msg.get("content")
        if content is None and isinstance(msg.get("reasoning_content"), str):
            content = None  # reasoning-only response; keep as empty content
        lp = choice.get("logprobs") or None
        usage = data.get("usage") or {}
        cost = usage.get("cost")
        if cost is None and isinstance(data.get("cost"), dict):
            cost = data["cost"].get("total")
        self.budget.record_success(model, cost)
        return Completion(
            content=content or "",
            finish_reason=choice.get("finish_reason", ""),
            usage={
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            },
            logprobs=lp.get("content") if isinstance(lp, dict) else lp,
            model_reported=data.get("model", ""),
            cost=cost,
            raw=data,
        )

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None


class _FatalClientError(Exception):
    pass
