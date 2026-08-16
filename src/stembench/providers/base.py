"""Provider adapter interface.

Providers are chat-completion backends. They must never require a specific vendor SDK:
everything speaks plain HTTP through httpx. Secrets come only from environment variables.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


class ProviderError(Exception):
    """Retryable or fatal provider failure."""


class DailyBudgetExceeded(ProviderError):
    """The provider's per-day request budget is exhausted; stop cleanly."""


@dataclass
class Completion:
    content: str = ""
    finish_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    logprobs: Optional[list[dict[str, Any]]] = None  # normalized top_logprobs if exposed
    model_reported: str = ""
    cost: Optional[float] = None
    raw: dict[str, Any] = field(default_factory=dict)  # response JSON minus secrets


class Provider(ABC):
    """A chat-completion provider with capability detection and budget accounting."""

    name: str = "base"

    @abstractmethod
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
        """Return one completion. Raises ProviderError/DailyBudgetExceeded."""

    @abstractmethod
    def supports_logprobs(self, model: str) -> bool: ...

    def close(self) -> None:  # optional
        pass
