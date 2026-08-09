"""Unit tests for the qwen-vllm provider profile's thinking-mode wiring.

Self-hosted Qwen3 (vLLM/SGLang/etc.) controls "thinking" through
``chat_template_kwargs.enable_thinking`` / ``.thinking_budget`` rather
than the top-level ``reasoning_effort`` field the generic ``custom``
provider speaks. Confirmed live against Qwen/Qwen3.6-27B: omitting
``enable_thinking`` leaves the model burning its entire token budget on
hidden reasoning before any visible content is emitted.

These tests pin the profile's wire-shape contract so Qwen3 requests stay
correctly shaped without going live.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def qwen_vllm_profile():
    """Resolve the registered qwen-vllm profile.

    Going through ``providers.get_provider_profile`` keeps the test honest —
    if someone later replaces the registered class with a plain
    ``ProviderProfile``, every assertion below collapses.
    """
    # ``model_tools`` triggers plugin discovery on import, which is what
    # registers the qwen-vllm profile in the global provider registry.
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("qwen-vllm")
    assert profile is not None, "qwen-vllm provider profile must be registered"
    return profile


class TestQwenVLLMThinkingWireShape:
    """``build_api_kwargs_extras`` produces Qwen3's exact wire format."""

    def test_default_enables_thinking_with_medium_budget(self, qwen_vllm_profile):
        """No reasoning_config -> thinking enabled, medium budget."""
        extra_body, top_level = qwen_vllm_profile.build_api_kwargs_extras(
            reasoning_config=None, model="Qwen/Qwen3.6-27B"
        )
        assert extra_body == {
            "chat_template_kwargs": {"enable_thinking": True, "thinking_budget": 8000}
        }
        assert top_level == {}

    @pytest.mark.parametrize(
        "effort,budget",
        [
            ("minimal", 2000),
            ("low", 4000),
            ("medium", 8000),
            ("high", 16000),
            ("xhigh", 32000),
            ("max", 32000),
            ("ultra", 32000),
        ],
    )
    def test_effort_levels_map_to_thinking_budget(self, qwen_vllm_profile, effort, budget):
        extra_body, top_level = qwen_vllm_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort},
            model="Qwen/Qwen3.6-27B",
        )
        assert extra_body == {
            "chat_template_kwargs": {"enable_thinking": True, "thinking_budget": budget}
        }
        assert top_level == {}

    def test_effort_matching_is_case_and_whitespace_insensitive(self, qwen_vllm_profile):
        extra_body, _ = qwen_vllm_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "  HIGH  "},
            model="Qwen/Qwen3.6-27B",
        )
        assert extra_body["chat_template_kwargs"]["thinking_budget"] == 16000

    def test_unknown_effort_falls_back_to_medium_budget(self, qwen_vllm_profile):
        extra_body, _ = qwen_vllm_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "garbage"},
            model="Qwen/Qwen3.6-27B",
        )
        assert extra_body["chat_template_kwargs"]["thinking_budget"] == 8000

    def test_explicitly_disabled_sends_enable_thinking_false(self, qwen_vllm_profile):
        """``reasoning_config.enabled=False`` -> enable_thinking=False, no budget.

        The crucial bit is that the parameter is *sent* at all — without
        it, Qwen3.6-27B defaults to thinking-on and burns its whole
        max_tokens budget on hidden reasoning (see module docstring).
        """
        extra_body, top_level = qwen_vllm_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model="Qwen/Qwen3.6-27B"
        )
        assert extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
        assert top_level == {}

    def test_disabled_ignores_effort_field(self, qwen_vllm_profile):
        """Effort silently dropped when thinking is off."""
        extra_body, _ = qwen_vllm_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False, "effort": "high"},
            model="Qwen/Qwen3.6-27B",
        )
        assert extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


class TestQwenVLLMModelGating:
    """Only Qwen3-generation models get the chat_template_kwargs translation."""

    @pytest.mark.parametrize(
        "model",
        [
            "Qwen/Qwen3.6-27B",
            "qwen3-coder-480b",
            "qwen3.5-plus",
            "QWEN3-NEXT-80B",  # case-insensitive
        ],
    )
    def test_qwen3_family_emits_chat_template_kwargs(self, qwen_vllm_profile, model):
        extra_body, _ = qwen_vllm_profile.build_api_kwargs_extras(
            reasoning_config=None, model=model
        )
        assert "chat_template_kwargs" in extra_body

    @pytest.mark.parametrize(
        "model",
        [
            "qwen2.5-72b-instruct",
            "qwen-max",
            "",
            None,
            "some-other-model",
        ],
    )
    def test_non_qwen3_models_emit_nothing(self, qwen_vllm_profile, model):
        extra_body, top_level = qwen_vllm_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"}, model=model
        )
        assert extra_body == {}
        assert top_level == {}


class TestQwenVLLMFullKwargsIntegration:
    """End-to-end: the transport's full kwargs match the intended wire shape."""

    def test_full_kwargs_match_wire_shape(self, qwen_vllm_profile):
        from agent.transports.chat_completions import ChatCompletionsTransport

        kwargs = ChatCompletionsTransport().build_kwargs(
            model="Qwen/Qwen3.6-27B",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=qwen_vllm_profile,
            reasoning_config={"enabled": True, "effort": "high"},
            base_url="https://clawmatrix.example.com/llm/v1",
            provider_name="qwen-vllm",
        )
        assert kwargs["model"] == "Qwen/Qwen3.6-27B"
        assert "reasoning_effort" not in kwargs
        assert kwargs["extra_body"]["chat_template_kwargs"] == {
            "enable_thinking": True,
            "thinking_budget": 16000,
        }


class TestQwenVLLMProviderIdentity:
    """The new provider is distinct from `custom` and doesn't collide with it."""

    def test_does_not_claim_the_vllm_alias(self, qwen_vllm_profile):
        """`custom` already owns the `vllm` alias — qwen-vllm must not steal it,
        or every non-Qwen self-hosted vLLM user would be silently rerouted here.
        """
        assert "vllm" not in qwen_vllm_profile.aliases

    def test_custom_profile_is_unaffected(self):
        import model_tools  # noqa: F401
        import providers

        custom_profile = providers.get_provider_profile("custom")
        assert custom_profile is not None
        extra_body, top_level = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model="Qwen/Qwen3.6-27B",
        )
        # custom's own (ineffective-for-Qwen3, but unchanged) wire shape.
        assert top_level == {"reasoning_effort": "high"}
        assert "chat_template_kwargs" not in extra_body
