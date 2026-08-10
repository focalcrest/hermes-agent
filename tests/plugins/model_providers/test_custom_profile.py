"""Unit tests for the custom provider profile's reasoning wiring.

``provider=custom`` covers any OpenAI-compatible endpoint the user points
Hermes at — local Ollama, vLLM, llama.cpp, and hosted reasoning APIs like
GLM-5.2 on Volcengine ARK. Before #57601's salvage, ``CustomProfile`` emitted
nothing when reasoning was *enabled*, so a configured ``reasoning_effort``
was silently dropped for every custom endpoint.

These tests pin the wire-shape contract:
  - disabled            → extra_body.think = False
  - enabled + effort    → top-level reasoning_effort (native OpenAI-compat
                          format GLM/ARK expect), passed through verbatim
                          including ``max``/``xhigh``
  - enabled + no effort → nothing emitted (endpoint's server default applies)
  - ollama_num_ctx      → extra_body.options.num_ctx, orthogonal to reasoning
"""

from __future__ import annotations

import pytest


@pytest.fixture
def custom_profile():
    """Resolve the registered custom profile via the global registry.

    Importing ``model_tools`` triggers plugin discovery, which registers the
    ``custom`` profile. Going through ``get_provider_profile`` keeps the test
    honest — if the registered class is ever downgraded to a plain
    ``ProviderProfile``, the assertions below collapse.
    """
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("custom")
    assert profile is not None, "custom provider profile must be registered"
    return profile


class TestCustomReasoningWireShape:
    """``build_api_kwargs_extras`` produces the correct wire format."""

    def test_no_reasoning_config_emits_nothing(self, custom_profile):
        """Unset reasoning → omit everything so the endpoint's default applies."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config=None, model="glm-5.2"
        )
        assert eb == {}
        assert tl == {}

    def test_disabled_sends_think_false(self, custom_profile):
        """enabled=False → reasoning_effort='none' top-level + think=False.

        Both fields are required: Ollama's /v1/chat/completions silently
        ignores extra_body.think (only /api/chat honours it — ollama#14820)
        but respects top-level reasoning_effort (#25758). think=False stays
        for proxies and the native /api/chat path.
        """
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model="glm-5.2"
        )
        assert eb == {"think": False}
        assert tl == {"reasoning_effort": "none"}

    def test_effort_none_sends_think_false(self, custom_profile):
        """effort='none' is the disable alias → same dual emission."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "none"}, model="glm-5.2"
        )
        assert eb == {"think": False}
        assert tl == {"reasoning_effort": "none"}

    @pytest.mark.parametrize(
        "effort", ["minimal", "low", "medium", "high", "xhigh", "max"]
    )
    def test_enabled_effort_goes_top_level(self, custom_profile, effort):
        """enabled + effort → TOP-LEVEL reasoning_effort, passed through verbatim.

        GLM-5.2/ARK and OpenAI-compatible reasoning APIs read reasoning_effort
        as a top-level string, not nested in extra_body. ``max`` is GLM's
        native deep-reasoning level and must survive.
        """
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort}, model="glm-5.2"
        )
        assert tl == {"reasoning_effort": effort}
        assert "reasoning_effort" not in eb
        assert "think" not in eb


    def test_does_not_force_think_true_on_enable(self, custom_profile):
        """We must never send think=True on enable — it's Ollama-only and
        would 400 on GLM/vLLM endpoints that don't recognize it."""
        eb, _ = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"}, model="glm-5.2"
        )
        assert eb.get("think") is not True


class TestCustomReasoningWithNumCtx:
    """Ollama num_ctx and reasoning are independent and compose."""

    def test_num_ctx_alone(self, custom_profile):
        # Non-Qwen3 model deliberately — see TestCustomQwen3ThinkingWireShape
        # for why a "qwen3" model name now takes a different branch entirely.
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config=None, ollama_num_ctx=8192, model="llama3.1"
        )
        assert eb == {"options": {"num_ctx": 8192}}
        assert tl == {}

    def test_num_ctx_composes_with_qwen3_thinking(self, custom_profile):
        """num_ctx and Qwen3's chat_template_kwargs are independent and compose."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config=None, ollama_num_ctx=8192, model="Qwen/Qwen3.6-27B"
        )
        assert eb["options"] == {"num_ctx": 8192}
        assert eb["chat_template_kwargs"] == {
            "enable_thinking": True,
            "thinking_budget": 8000,
        }
        assert tl == {}


class TestCustomQwen3ThinkingWireShape:
    """Qwen3 models take chat_template_kwargs instead of think/reasoning_effort.

    vLLM's OpenAI-compatible server serving Qwen3 doesn't recognize the
    Ollama ``think`` flag or GLM/ARK's top-level ``reasoning_effort`` —
    both silently ignored, so every reasoning_effort level was a no-op for
    Qwen3 before this branch existed. Confirmed live against
    Qwen/Qwen3.6-27B: omitting ``chat_template_kwargs.enable_thinking``
    leaves the model burning its entire max_tokens budget on hidden
    reasoning before any visible content (content: null, finish_reason:
    "length") — which is exactly why, unlike the "omit when unset" default
    for every other custom endpoint below, Qwen3 always gets an explicit
    enable_thinking rather than silently deferring to the server default.
    """

    def test_no_reasoning_config_defaults_to_medium_thinking(self, custom_profile):
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config=None, model="Qwen/Qwen3.6-27B"
        )
        assert eb == {
            "chat_template_kwargs": {"enable_thinking": True, "thinking_budget": 8000}
        }
        assert tl == {}

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
    def test_effort_levels_map_to_thinking_budget(self, custom_profile, effort, budget):
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort},
            model="Qwen/Qwen3.6-27B",
        )
        assert eb == {
            "chat_template_kwargs": {"enable_thinking": True, "thinking_budget": budget}
        }
        assert tl == {}

    def test_unknown_effort_falls_back_to_medium_budget(self, custom_profile):
        eb, _ = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "garbage"},
            model="Qwen/Qwen3.6-27B",
        )
        assert eb["chat_template_kwargs"]["thinking_budget"] == 8000

    def test_disabled_sends_enable_thinking_false_no_budget(self, custom_profile):
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model="Qwen/Qwen3.6-27B"
        )
        assert eb == {"chat_template_kwargs": {"enable_thinking": False}}
        assert tl == {}

    def test_disabled_ignores_effort_field(self, custom_profile):
        eb, _ = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False, "effort": "high"},
            model="Qwen/Qwen3.6-27B",
        )
        assert eb == {"chat_template_kwargs": {"enable_thinking": False}}

    def test_effort_none_is_a_disable_alias(self, custom_profile):
        """``effort: "none"`` with no explicit ``enabled`` key must also
        disable thinking — the same alias the non-Qwen3 branch below (and
        Hermes's own parse_reasoning_effort) already honors. Regression
        test: an earlier version of this branch only checked
        ``enabled is False`` and so treated effort="none" as "enabled,
        unrecognized effort" -> silently turned thinking ON at the medium
        budget, the opposite of what was requested."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"effort": "none"}, model="Qwen/Qwen3.6-27B"
        )
        assert eb == {"chat_template_kwargs": {"enable_thinking": False}}
        assert tl == {}

    def test_never_emits_top_level_reasoning_effort_or_think(self, custom_profile):
        """Qwen3 must never get the Ollama/GLM-ARK wire shape — vLLM ignores
        both fields, so sending them would just be silent noise at best."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model="Qwen/Qwen3.6-27B",
        )
        assert "reasoning_effort" not in tl
        assert "think" not in eb

    @pytest.mark.parametrize(
        "model",
        [
            "Qwen/Qwen3.6-27B",
            "qwen3-coder-480b",
            "qwen3.5-plus",
            "QWEN3-NEXT-80B",  # case-insensitive
        ],
    )
    def test_qwen3_family_all_take_this_branch(self, custom_profile, model):
        eb, _ = custom_profile.build_api_kwargs_extras(reasoning_config=None, model=model)
        assert "chat_template_kwargs" in eb

    @pytest.mark.parametrize(
        "model",
        [
            "qwen2.5-72b-instruct",
            "qwen-max",
            "glm-5.2",
            "llama3.1",
            "",
            None,
        ],
    )
    def test_non_qwen3_models_are_unaffected(self, custom_profile, model):
        """Non-Qwen3 models (including earlier Qwen generations) must keep
        the existing Ollama/GLM-ARK wire shape untouched — this is the
        regression this whole branch must not introduce."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"}, model=model
        )
        assert "chat_template_kwargs" not in eb
        assert tl == {"reasoning_effort": "high"}

