# Provider references and zero-spend verification

Checked 2026-08-17. These sources govern provider integration and the hard free-model
allowlists in `src/stembench/providers/registry.py`.

| Provider source | Claim used by the project |
|---|---|
| [OpenRouter models API](https://openrouter.ai/docs/api/api-reference/models/get-models) | Machine-readable current model catalog used to verify exact IDs and zero pricing. |
| [OpenRouter free model variant](https://openrouter.ai/docs/guides/routing/model-variants/free) | The `:free` suffix selects a model's free variant; the project still uses an exact-ID allowlist rather than the unattributed `openrouter/free` router. |
| [OpenRouter FAQ](https://openrouter.ai/docs/faq) | OpenAI-compatible API behavior and free-model account limits. |
| [OpenRouter chat-completions API](https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request) | Request/response contract implemented by the shared OpenAI-compatible adapter. |
| [OpenRouter errors and debugging](https://openrouter.ai/docs/api/reference/errors-and-debugging) | Retryable status handling, including `Retry-After`; retries are bounded and each HTTP attempt consumes the local daily budget. |
| [OpenCode Zen documentation](https://opencode.ai/docs/zen/) | Zen endpoint, authentication, current free model IDs, and pricing table. Only IDs explicitly marked free are allowed. |

The allowlists are a second safety boundary, not a claim that provider catalogs are
immutable. Live evaluation records the exact requested and provider-reported model IDs
and stops if a provider reports nonzero cost. API keys remain only in the ignored
`.env`; this document records no credentials or account-specific values.
