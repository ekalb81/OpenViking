"""Per-call-site model selection for the working-memory update.

Extraction and working memory are different jobs with different economics.
Extraction runs roughly eleven calls per commit and dominates spend; the
working-memory update is a single forced tool call whose output seeds every
later session in that conversation. Being able to point them at different
models is what lets you optimise each for the thing that actually matters to it.

The override merges rather than replaces: a config that only names a model must
not lose the parent's credentials, api_base, or timeouts.
"""

from openviking_cli.utils.config.vlm_config import VLMConfig


def _parent(**overrides):
    config = {
        "model": "openrouter/tencent/hy3-preview",
        "api_key": "parent-key",
        "api_base": "https://openrouter.ai/api/v1",
        "temperature": 0.0,
        "timeout": 600.0,
    }
    config.update(overrides)
    return VLMConfig(**config)


def test_without_an_override_the_same_config_is_used():
    # The default must remain one model for everything.
    parent = _parent()

    assert parent.for_working_memory() is parent


def test_the_override_changes_only_the_model():
    parent = _parent(working_memory={"model": "openrouter/z-ai/glm-5.2"})

    wm = parent.for_working_memory()

    assert wm.model == "openrouter/z-ai/glm-5.2"
    # Everything unstated is inherited - the whole point of merging.
    assert wm.api_key == "parent-key"
    assert wm.api_base == "https://openrouter.ai/api/v1"
    assert wm.timeout == 600.0


def test_extraction_still_sees_the_parent_model():
    # The override must not leak into the config extraction uses.
    parent = _parent(working_memory={"model": "openrouter/z-ai/glm-5.2"})

    parent.for_working_memory()

    assert parent.model == "openrouter/tencent/hy3-preview"


def test_unmentioned_fields_do_not_clobber_the_parent_with_defaults():
    """The subtle failure this guards.

    VLMConfig gives temperature and timeout defaults. Merging a whole config
    object would copy those *defaults* over the parent's real values, silently
    resetting a tuned timeout because someone changed a model name. A patch dict
    only carries what was written.
    """
    parent = _parent(temperature=0.7, timeout=900.0,
                     working_memory={"model": "openrouter/z-ai/glm-5.2"})

    wm = parent.for_working_memory()

    assert wm.temperature == 0.7
    assert wm.timeout == 900.0


def test_explicitly_set_fields_do_win():
    parent = _parent(temperature=0.7,
                     working_memory={"model": "m", "temperature": 0.0})

    wm = parent.for_working_memory()

    assert wm.temperature == 0.0


def test_the_resolved_config_carries_no_further_override():
    # Otherwise resolving twice could keep descending.
    parent = _parent(working_memory={"model": "openrouter/z-ai/glm-5.2"})

    wm = parent.for_working_memory()

    assert wm.working_memory is None
    assert wm.for_working_memory() is wm


def test_an_empty_override_block_is_a_no_op():
    parent = _parent(working_memory={})

    assert parent.for_working_memory() is parent


def test_the_working_memory_call_site_resolves_the_override():
    """A correct resolver nothing calls would change nothing."""
    import inspect

    from openviking.session import session as session_module

    source = inspect.getsource(session_module.Session._generate_archive_summary_async)
    assert "for_working_memory()" in source
    # It must resolve before the config is used to build the request.
    assert source.index("for_working_memory()") < source.index("get_completion_async")


def test_a_typo_in_the_override_is_rejected_not_ignored():
    # A silently-dropped key would leave the override looking applied while the
    # parent model kept serving every request.
    import pytest

    parent = _parent(working_memory={"modle": "openrouter/z-ai/glm-5.2"})

    with pytest.raises(ValueError, match="unknown field"):
        parent.for_working_memory()
