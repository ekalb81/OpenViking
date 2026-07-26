"""cache_control has to survive LiteLLM to reach OpenRouter.

We mark the system message as an ephemeral cache breakpoint for every
`openrouter/` model. LiteLLM then gates that marker on its own hardcoded
substring allowlist - claude, gemini, minimax, glm, z-ai - and strips it from
the content blocks of anything else *before the request leaves the process*.
Qwen is absent from that list, so the marker never reached the provider and
every call paid full input price.

Measured against the live provider with a 17.5k-token prefix: stock LiteLLM
reports cached_tokens=0 and cache_write_tokens=0 on repeated identical calls;
with the allowlist extended, the first call reports cache_write_tokens=17504 and
the next reports cached_tokens=17504, taking the prompt cost from $0.00329 to
$0.00033.
"""

import openviking.models.vlm.backends.litellm_vlm as litellm_vlm

pytest_plugins = ()


def _config():
    from litellm.llms.openrouter.chat.transformation import OpenrouterConfig

    return OpenrouterConfig()


def test_qwen_is_allowed_to_carry_cache_control():
    # Importing the backend module is what applies the extension.
    assert _config()._supports_cache_control_in_content("qwen/qwen3.6-flash") is True


def test_litellms_own_supported_models_still_pass():
    config = _config()

    assert config._supports_cache_control_in_content("anthropic/claude-sonnet-4.5") is True
    assert config._supports_cache_control_in_content("google/gemini-2.5-flash") is True


def test_unrelated_models_are_left_exactly_as_litellm_had_them():
    # Widening the allowlist past what a provider actually supports would send a
    # marker the provider may reject, so the extension must stay narrow.
    config = _config()

    assert config._supports_cache_control_in_content("openai/gpt-4o") is False
    assert config._supports_cache_control_in_content("meta-llama/llama-3-70b") is False


def test_applying_the_extension_twice_does_not_stack():
    from litellm.llms.openrouter.chat.transformation import OpenrouterConfig

    before = OpenrouterConfig._supports_cache_control_in_content

    litellm_vlm._extend_openrouter_cache_control()
    litellm_vlm._extend_openrouter_cache_control()

    assert OpenrouterConfig._supports_cache_control_in_content is before
    assert _config()._supports_cache_control_in_content("qwen/qwen3.6-flash") is True


def test_a_moved_litellm_internal_degrades_instead_of_raising(monkeypatch):
    """Patching a third-party internal must fail soft.

    If LiteLLM renames or removes the hook, the right outcome is stock
    behaviour - no caching - not an import-time crash that takes the VLM
    backend down with it.
    """
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if "openrouter" in name:
            raise ImportError("simulated litellm reshuffle")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)

    litellm_vlm._extend_openrouter_cache_control()  # must not raise


def test_the_system_message_is_still_the_thing_being_marked():
    """The extension is worthless if we stop emitting the marker."""
    provider = litellm_vlm.LiteLLMVLMProvider(
        {"model": "openrouter/qwen/qwen3.6-flash", "api_key": "k", "temperature": 0.0}
    )

    kwargs = provider._build_kwargs(
        "openrouter/qwen/qwen3.6-flash",
        [
            {"role": "system", "content": "a large static extraction schema"},
            {"role": "user", "content": "the transcript"},
        ],
    )

    system_content = kwargs["messages"][0]["content"]
    assert isinstance(system_content, list)
    assert system_content[0]["cache_control"] == {"type": "ephemeral"}
    # Only the stable prefix is marked; the variable turn must not be.
    assert not isinstance(kwargs["messages"][1]["content"], list)
