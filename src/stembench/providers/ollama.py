"""Ollama local provider (localhost only, no cost). Available when the user runs a
local Ollama server; not required for any test. Free-form model list: local models
are inherently zero-cost, so the allowlist check is skipped (localhost only)."""

from __future__ import annotations

import os

import httpx

from stembench.providers.base import Completion, Provider, ProviderError

BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


class OllamaProvider(Provider):
    name = "ollama"

    def __init__(self, timeout_s: float = 600.0):
        self.timeout_s = timeout_s
        self._client = httpx.Client(timeout=timeout_s)

    def supports_logprobs(self, model: str) -> bool:
        return True  # Ollama returns prompt+generated logprobs when requested

    def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float | None = None,
        seed: int | None = None,
        request_logprobs: bool = True,
        reasoning_effort: str | None = None,
    ) -> Completion:
        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if seed is not None:
            payload["options"]["seed"] = seed
        if request_logprobs:
            payload["logprobs"] = 5
        try:
            resp = self._client.post(f"{BASE}/api/chat", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise ProviderError(f"ollama: {e}") from e
        data = resp.json()
        return Completion(
            content=data.get("message", {}).get("content", ""),
            finish_reason="stop",
            usage={
                "prompt_tokens": data.get("prompt_eval_count"),
                "completion_tokens": data.get("eval_count"),
                "total_tokens": (data.get("prompt_eval_count") or 0)
                + (data.get("eval_count") or 0),
            },
            logprobs=data.get("prompt_logprobs"),
            model_reported=data.get("model", model),
            cost=0.0,
            raw={"synthetic_local": False, "local": True},
        )

    def close(self) -> None:
        self._client.close()
