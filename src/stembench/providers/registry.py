"""Provider registry with verified free-model allowlists.

Allowlists were verified live on 2026-08-17 (see progress.md). Only models on these
lists can ever be called: OpenAICompatProvider._assert_free refuses everything else,
which enforces the zero-paid-spend policy (OpenRouter free tier; Zen free models).
"""

from __future__ import annotations

from stembench.providers.base import Provider
from stembench.providers.fake import FakeProvider
from stembench.providers.openai_compat import OpenAICompatProvider

OPENROUTER_FREE = [
    "cohere/north-mini-code:free",
    "dots-studio/dots-3-note-preview:free",
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-4-31b-it:free",
    "liquid/lfm-2.5-2.6b:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3.5-content-safety:free",
    "nvidia/nemotron-3.5-lightning:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "nvidia/nemotron-nano-9b-v2:free",
    "openai/gpt-oss-20b:free",
    "poolside/laguna-s-2.1:free",
    "poolside/laguna-xs-2.1:free",
    "stealth/ox-alpha",
    "z-ai/glm-5.2:free",
]

ZEN_FREE = [
    "big-pickle",
    "deepseek-v4-flash-free",
    "mimo-v2.5-free",
    "hy3-free",
    "nemotron-3-ultra-free",
    "nemotron-3.5-lightning-free",
    "laguna-s-2.1-free",
]

# Models confirmed to expose top_logprobs through OpenRouter (subset; capability
# detection at runtime keeps the record honest when a provider drops support).
OPENROUTER_LOGPROB_MODELS = {
    "openai/gpt-oss-20b:free",
    "google/gemma-4-31b-it:free",
    "z-ai/glm-5.2:free",
}


def build_provider(name: str) -> Provider:
    name = name.lower()
    if name == "fake":
        return FakeProvider()
    if name == "openrouter":
        return OpenAICompatProvider(
            name="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
            free_models=OPENROUTER_FREE,
            logprob_models=OPENROUTER_LOGPROB_MODELS,
            requests_per_minute=18,  # headroom under the documented 20 rpm
            daily_cap=50,  # documented free-tier daily cap (accounts with < $10 credits)
            uncapped_models={"stealth/ox-alpha"},  # user-relayed provider terms: unlimited; live-verified below
            extra_headers={"HTTP-Referer": "https://github.com/aayushkrm/llm-stembench",
                           "X-Title": "LLM-STEMBench"},
        )
    if name == "zen":
        return OpenAICompatProvider(
            name="zen",
            base_url="https://opencode.ai/zen/v1",
            api_key_env="OPENCODE_ZEN_API_KEY",
            free_models=ZEN_FREE,
            logprob_models=set(),  # capability not confirmed; detect opportunistically
            requests_per_minute=15,
            daily_cap=None,  # no documented account cap; upstream saturation handled by retries
        )
    if name == "ollama":
        from stembench.providers.ollama import OllamaProvider

        return OllamaProvider()
    raise KeyError(f"unknown provider '{name}' (openrouter|zen|ollama|fake)")
