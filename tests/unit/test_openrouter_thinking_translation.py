"""Thinking control for OpenRouter-hosted models.

`enable_thinking` is DashScope-native and OpenRouter does not translate it, so
before this translation existed there was no way to turn thinking off for an
OpenRouter model and the provider default always won. For Qwen3 that default is
thinking-on, which the provider refuses to combine with a forced tool_choice:

    "The tool_choice parameter does not support being set to required or object
     in thinking mode"

That is not cosmetic. The working-memory update forces the update_working_memory
tool precisely because it needs per-section decisions to merge; when the call
400s, the handler falls back to regenerating the document from scratch, so the
merge never runs and the failure is invisible apart from a warning.
"""

from openviking.models.vlm.backends.litellm_vlm import LiteLLMVLMProvider

OPENROUTER_MODEL = "openrouter/qwen/qwen3.6-flash"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "update_working_memory",
            "description": "Return per-section update decisions.",
            "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
        },
    }
]
FORCED_TOOL = {"type": "function", "function": {"name": "update_working_memory"}}


def _provider(**overrides):
    config = {
        "model": OPENROUTER_MODEL,
        "api_base": "https://openrouter.ai/api/v1",
        "api_key": "test-key",
        "temperature": 0.0,
    }
    config.update(overrides)
    return LiteLLMVLMProvider(config)


def _kwargs(provider, **call_kwargs):
    return provider._build_kwargs(
        OPENROUTER_MODEL, [{"role": "user", "content": "hi"}], **call_kwargs
    )


def test_thinking_off_is_translated_to_openrouters_reasoning_parameter():
    kwargs = _kwargs(_provider(), tools=TOOLS, tool_choice=FORCED_TOOL, thinking=False)

    assert kwargs["extra_body"]["reasoning"] == {"enabled": False}
    # The forced tool must survive: turning thinking off is what makes forcing
    # it legal, so a fix that dropped the force would defeat the purpose.
    assert kwargs["tool_choice"] == FORCED_TOOL


def test_thinking_on_is_translated_too():
    kwargs = _kwargs(_provider(), tools=TOOLS, tool_choice="auto", thinking=True)

    assert kwargs["extra_body"]["reasoning"] == {"enabled": True}


def test_no_stated_intent_leaves_the_provider_default_alone():
    # Callers that pass nothing must keep the behaviour they had before this
    # translation existed, so enabling it cannot silently change extraction.
    kwargs = _kwargs(_provider(), tools=TOOLS, tool_choice="auto", thinking=None)

    assert "reasoning" not in (kwargs.get("extra_body") or {})


def test_enable_thinking_is_not_sent_to_openrouter():
    # It is the DashScope field. Sending it here is what silently did nothing.
    kwargs = _kwargs(_provider(), tools=TOOLS, tool_choice=FORCED_TOOL, thinking=False)

    assert "enable_thinking" not in (kwargs.get("extra_body") or {})


def test_configured_extra_request_body_wins_over_the_default():
    provider = _provider(extra_request_body={"reasoning": {"enabled": True, "effort": "high"}})

    kwargs = _kwargs(provider, tools=TOOLS, tool_choice="auto", thinking=False)

    assert kwargs["extra_body"]["reasoning"] == {"enabled": True, "effort": "high"}


def test_non_openrouter_models_are_untouched():
    provider = _provider(model="ollama/llama3")

    kwargs = provider._build_kwargs(
        "ollama/llama3", [{"role": "user", "content": "hi"}], tools=TOOLS, thinking=False
    )

    assert "reasoning" not in (kwargs.get("extra_body") or {})


def test_the_working_memory_update_states_thinking_false():
    """The call site must state the intent, not inherit a default.

    _build_kwargs only overrides the provider default when `thinking` is not
    None, so this call site passing nothing is exactly what left the provider
    in thinking mode and produced the 400.
    """
    import inspect

    from openviking.session import session as session_module

    source = inspect.getsource(session_module.Session._generate_archive_summary_async)
    forced = source.index('"name": "update_working_memory"')
    # The thinking=False must belong to the forced-tool call, not some later one.
    assert "thinking=False" in source[forced : forced + 800]
