"""Custom / Ollama (local) provider profile.

Covers any endpoint registered as provider="custom", including local
Ollama instances and OpenAI-compatible reasoning endpoints (GLM-5.2 on
Volcengine ARK, vLLM, llama.cpp, self-hosted Qwen3). Key quirks:
  - ollama_num_ctx → extra_body.options.num_ctx (local context window)
  - Qwen3 models (model name contains "qwen3") → extra_body.
    chat_template_kwargs.enable_thinking/.thinking_budget — see
    _model_is_qwen3's doc comment for why this needs its own branch
    instead of the reasoning_effort/think handling below.
  - reasoning_config disabled → top-level reasoning_effort="none"
    (Ollama /v1/chat/completions ignores think=False — ollama#14820)
    + extra_body.think = False for /api/chat and proxies
  - reasoning_config enabled + effort → top-level reasoning_effort
    (the native OpenAI-compatible format GLM/ARK expect; unset omits it
    so the endpoint's server default applies)

Deliberately NOT a separate registered provider for Qwen3: this
deployment pattern (model.api_key set directly in config.yaml, no
provider-specific env var) only works through hermes_cli's `custom`
special-casing — resolve_provider()/resolve_api_key_provider_credentials()
route any *other* registered provider name through PROVIDER_REGISTRY,
which demands an env-var-resolved API key and rejects a provider with
none configured ("No usable credentials found"). A standalone
`qwen-vllm` provider profile hit exactly that live (#confirmed in
production) before this branch replaced it.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

# Token budget per effort level for Qwen3's chat_template_kwargs.thinking_budget.
# Mirrors agent/anthropic_adapter.py's own THINKING_BUDGET table (same
# low/medium/high/xhigh values) so an effort level costs roughly the same
# order of magnitude of thinking tokens regardless of which backend/model
# it's applied to. max/ultra reuse xhigh's budget rather than growing
# further — vLLM's own generation max_tokens is the real ceiling worth
# tuning for a self-hosted deployment.
QWEN3_THINKING_BUDGET = {
    "minimal": 2000,
    "low": 4000,
    "medium": 8000,
    "high": 16000,
    "xhigh": 32000,
    "max": 32000,
    "ultra": 32000,
}


def _model_is_qwen3(model: str | None) -> bool:
    """Qwen3-generation models (qwen3, qwen3.5, qwen3.6, qwen3-coder, ...)
    control thinking via chat_template_kwargs.enable_thinking/.thinking_budget
    — a different wire shape than the Ollama ``think`` flag / GLM-ARK
    top-level ``reasoning_effort`` this profile speaks for every other
    backend. vLLM's OpenAI-compatible server serving Qwen3 silently
    ignores both of those, so without this branch every reasoning_effort
    level is a no-op for Qwen3 (confirmed live against Qwen/Qwen3.6-27B:
    the model burns its whole max_tokens budget on hidden reasoning
    before ever emitting visible content — content: null, finish_reason:
    "length" — the same quirk ClawMatrix's own internal/llmclient.go
    independently hit and worked around on a separate, non-Hermes call
    path). Earlier generations (qwen2, qwen2.5, ...) have no
    chat-template thinking toggle at all, so they're left untouched.
    """
    return "qwen3" in (model or "").strip().lower()


class CustomProfile(ProviderProfile):
    """Custom/Ollama local provider — think=false, num_ctx, and Qwen3 support."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        ollama_num_ctx: int | None = None,
        model: str | None = None,
        **ctx: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}

        # Ollama context window — applies regardless of model family.
        if ollama_num_ctx:
            options = extra_body.get("options", {})
            options["num_ctx"] = ollama_num_ctx
            extra_body["options"] = options

        if _model_is_qwen3(model):
            _effort = ""
            _enabled = True
            if reasoning_config and isinstance(reasoning_config, dict):
                _effort = (reasoning_config.get("effort") or "").strip().lower()
                _enabled = reasoning_config.get("enabled", True)
            # "none" is the disable alias used throughout this file (and by
            # Hermes's own parse_reasoning_effort) — not just enabled=False.
            if _effort == "none" or _enabled is False:
                extra_body["chat_template_kwargs"] = {"enable_thinking": False}
            else:
                budget = QWEN3_THINKING_BUDGET.get(_effort or "medium", QWEN3_THINKING_BUDGET["medium"])
                extra_body["chat_template_kwargs"] = {
                    "enable_thinking": True,
                    "thinking_budget": budget,
                }
            return extra_body, top_level

        # Reasoning / thinking control for every other custom OpenAI-compatible
        # endpoint (GLM-5.2 on Volcengine ARK, Ollama, llama.cpp, …).
        #
        #   - disabled  → extra_body.think = False (Ollama's thinking-off flag)
        #   - enabled + effort set → TOP-LEVEL reasoning_effort string, the
        #     format GLM-5.2/ARK and other OpenAI-compatible reasoning APIs
        #     expect (GLM documents "high" and "max"; "max" is its default).
        #   - enabled + no effort  → omit both, so the endpoint applies its own
        #     server-side default (do NOT force a level the user didn't pick).
        #
        # We deliberately do NOT emit ``think=True`` on enable: it is an
        # Ollama-only flag and thinking is already server-default-on for these
        # backends, so forcing it risks a 400 on GLM/vLLM endpoints that don't
        # recognize it. Mirrors the DeepSeek/Zai profile precedent.
        if reasoning_config and isinstance(reasoning_config, dict):
            _effort = (reasoning_config.get("effort") or "").strip().lower()
            _enabled = reasoning_config.get("enabled", True)
            if _effort == "none" or _enabled is False:
                # Ollama's /v1/chat/completions silently ignores
                # extra_body.think (only /api/chat honours it — ollama#14820)
                # but respects the top-level reasoning_effort field, so both
                # are needed to actually stop a thinking-capable model from
                # reasoning (#25758). Endpoints that recognize neither simply
                # ignore them.
                top_level["reasoning_effort"] = "none"
                extra_body["think"] = False
            elif _effort:
                top_level["reasoning_effort"] = _effort

        return extra_body, top_level

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Custom/Ollama: base_url is user-configured; fetch if set."""
        if not (base_url or self.base_url):
            return None
        return super().fetch_models(api_key=api_key, base_url=base_url, timeout=timeout)


custom = CustomProfile(
    name="custom",
    aliases=(
        "ollama",
        "local",
        "vllm",
        "llamacpp",
        "llama.cpp",
        "llama-cpp",
    ),
    env_vars=(),  # No fixed key — custom endpoint
    base_url="",  # User-configured
    # Without this, no max_tokens is sent and Ollama falls back to its internal
    # num_predict=128, truncating responses after a few tokens (#39281). This is
    # only a floor used when the user hasn't set model.max_tokens — they can
    # override per-model — so we set it generously rather than lowballing it.
    default_max_tokens=65536,
)

register_provider(custom)
