"""Deterministic fake provider. TEST FIXTURES ONLY — never counts as an empirical run.

Content is a pure function of (model, prompt): the correct-choice letter is derived
from a deterministic hash unless a `fail_items`/`wrong_items` override targets the item.
This exists to exercise the full pipeline offline; any artifact produced with it MUST be
labeled synthetic (the runner stamps provider="fake" into every record).
"""

from __future__ import annotations

import hashlib
import re

from stembench.providers.base import Completion, Provider


class FakeProvider(Provider):
    name = "fake"

    def __init__(self, wrong_frac: float = 0.4, models: list[str] | None = None):
        self.wrong_frac = wrong_frac
        self.models = models or ["fake/tiny-test"]
        self.calls: list[tuple[str, str]] = []

    def supports_logprobs(self, model: str) -> bool:
        return True

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
        self.calls.append((model, messages[-1]["content"]))
        prompt = messages[-1]["content"]
        digest = hashlib.sha256(f"{model}::{prompt}".encode()).hexdigest()
        score = int(digest[:8], 16) / 0xFFFFFFFF
        m = re.search(r"^Answer: ([A-F])", prompt, re.MULTILINE)
        gold = m.group(1) if m else "A"
        if score < self.wrong_frac:
            answer = "B" if gold != "B" else "C"
            conf = 55 + score * 30
        else:
            answer = gold
            conf = 70 + score * 29
        content = (
            f"Let me think briefly.\nAnswer: {answer}\nConfidence: {int(conf)}"
        )
        return Completion(
            content=content,
            finish_reason="stop",
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            logprobs=[{"token": answer, "logprob": -0.3 + score * 0.2}],
            model_reported=model,
            cost=0.0,
            raw={"synthetic": True},
        )
