"""Self-hosted Qwen3 (vLLM / SGLang / any OpenAI-compatible endpoint) provider profile.

Qwen3's chat template controls "thinking" via ``chat_template_kwargs``
(``enable_thinking``, ``thinking_budget``) — not a top-level
``reasoning_effort`` field (Kimi/DeepSeek/GLM-ARK's convention) and not
Ollama's ``think`` flag, the two shapes the generic ``custom`` provider
profile already speaks (see ``plugins/model-providers/custom/__init__.py``).
A ``provider: custom`` deployment pointed at a self-hosted Qwen3 endpoint
therefore has no way to actually change the model's thinking behavior:
Hermes's own ``reasoning_effort`` picker (``hermes model`` / config.yaml's
``agent.reasoning_effort``) sends a field vLLM's OpenAI-compatible server
silently ignores, so every effort level is a no-op. Confirmed live against
Qwen/Qwen3.6-27B: with no ``enable_thinking`` override the model burns its
entire token budget on hidden reasoning before emitting any visible
content (see ClawMatrix's own ``internal/llmclient.go``, which hit and
worked around the identical quirk on a separate, non-Hermes call path).

This profile fixes that for self-hosted Qwen3 deployments by translating
``reasoning_config`` into ``extra_body.chat_template_kwargs.enable_thinking``
/ ``.thinking_budget`` instead.

Registered under a distinct name/alias set — deliberately NOT ``vllm``,
which ``custom`` already claims as an alias — so generic custom-endpoint
users running non-Qwen models through ``provider: custom`` are unaffected.
"""

from __future__ import annotations

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

# Token budget per effort level. Mirrors agent/anthropic_adapter.py's own
# THINKING_BUDGET table (same low/medium/high/xhigh values) so an effort
# level costs roughly the same order of magnitude of thinking tokens
# regardless of which backend/model it's applied to. max/ultra (Hermes's
# two strongest levels, mainly meant for frontier hosted models) reuse
# xhigh's budget rather than growing further — vLLM's own generation
# max_tokens is the real ceiling worth tuning for a self-hosted deployment.
THINKING_BUDGET = {
    "minimal": 2000,
    "low": 4000,
    "medium": 8000,
    "high": 16000,
    "xhigh": 32000,
    "max": 32000,
    "ultra": 32000,
}


def _model_supports_thinking(model: str | None) -> bool:
    """Qwen3-generation models expose enable_thinking/thinking_budget.

    Covers qwen3, qwen3.5, qwen3.6, qwen3-coder, qwen3-next, etc. Earlier
    generations (qwen2, qwen2.5, ...) have no chat-template thinking
    toggle at all, so requests for those are left untouched.
    """
    m = (model or "").strip().lower()
    return "qwen3" in m


class QwenVLLMProfile(ProviderProfile):
    """Self-hosted Qwen3 over an OpenAI-compatible endpoint (vLLM/SGLang/...)."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        model: str | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not _model_supports_thinking(model):
            return {}, {}

        if isinstance(reasoning_config, dict) and reasoning_config.get("enabled") is False:
            return {"chat_template_kwargs": {"enable_thinking": False}}, {}

        effort = "medium"
        if isinstance(reasoning_config, dict):
            effort = (reasoning_config.get("effort") or "medium").strip().lower()
        budget = THINKING_BUDGET.get(effort, THINKING_BUDGET["medium"])

        return {
            "chat_template_kwargs": {"enable_thinking": True, "thinking_budget": budget}
        }, {}

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Self-hosted: base_url is user-configured; fetch only if set (mirrors CustomProfile)."""
        if not (base_url or self.base_url):
            return None
        return super().fetch_models(api_key=api_key, base_url=base_url, timeout=timeout)


qwen_vllm = QwenVLLMProfile(
    name="qwen-vllm",
    aliases=("qwen3-vllm", "qwen-selfhosted"),
    display_name="Qwen3 (self-hosted)",
    description="Self-hosted Qwen3 over vLLM/SGLang or any OpenAI-compatible endpoint",
    env_vars=(),  # No fixed key -- user-configured endpoint, like `custom`.
    base_url="",  # User-configured.
    # Same rationale as CustomProfile: without an explicit floor, a
    # backend that defaults max_tokens low (e.g. Ollama-style num_predict)
    # would silently truncate responses. Users can still override via
    # model.max_tokens.
    default_max_tokens=65536,
)

register_provider(qwen_vllm)
