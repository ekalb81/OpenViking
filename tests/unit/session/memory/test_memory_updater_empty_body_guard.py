"""Refusal to persist a memory whose rendered body says nothing.

Templates render their headings unconditionally, so a field the model left blank
still produces a well-formed file containing no information. Events are the live
case: their template used to fall back to the raw chatlog when the summary was
empty, and removing that fallback left the prompt's "REQUIRED" instruction as
the only thing standing between a lazy model and an empty memory.

An empty memory is worse than a missing one — it occupies a slot, is returned by
recall, and says nothing when it gets there.
"""

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.session.memory.dataclass import ResolvedOperation
from openviking.session.memory.memory_type_registry import create_default_registry
from openviking.session.memory.memory_updater import (
    MemoryUpdater,
    _rendered_body_is_empty,
)
from openviking_cli.session.user_id import UserIdentifier


def _ctx():
    return RequestContext(user=UserIdentifier("account", "u"), role=Role.USER)


class _ExtractContext:
    """The minimum the events template touches.

    Passing None instead makes the render raise, and MemoryFileUtils then falls
    back to a plain-content file — which for events, whose body comes entirely
    from the template, is metadata and nothing else. That is a real latent
    failure mode, covered separately below.
    """

    def get_resource_event_content(self, *args, **kwargs):
        return ""  # a plain, non-resource event

    def get_event_content(self, *args, **kwargs):
        return ""

    def get_first_message_time_with_weekday_from_ranges(self, *args, **kwargs):
        return "Mon"


class _RecordingFs:
    def __init__(self):
        self.writes = []

    async def read_file(self, uri, ctx=None, **kwargs):
        return ""  # nothing stored yet

    async def write_file(self, uri, content, ctx=None, **kwargs):
        self.writes.append((uri, content))


def _updater(fs):
    updater = MemoryUpdater(registry=create_default_registry())
    updater._viking_fs = fs
    return updater


@pytest.mark.parametrize(
    "rendered",
    [
        "",
        "# Summary\n",
        "# Summary\n\n   \n",
        "---\na: 1\n---\n# Summary\n",
        "<!-- MEMORY_FIELDS {...} -->\n# Summary\n",
        "## Situation\n\n## Approach\n\n## Reflect\n",
    ],
)
def test_scaffolding_without_a_body_is_empty(rendered):
    assert _rendered_body_is_empty(rendered) is True


@pytest.mark.parametrize(
    "rendered",
    [
        "# Summary\nGina expanded the store.",
        "- Works as a software engineer.",
        "## Situation\n- A bounded change needs prior guidance.",
        "---\na: 1\n---\n# Summary\nReal content here.",
        "# Title\n\n1. First step\n2. Second step",
        "```\ncode block\n```",
        "| a | b |\n|---|---|\n| 1 | 2 |",
        "no headings at all, just prose",
    ],
)
def test_a_real_body_is_not_empty(rendered):
    # False positives here would silently drop genuine memories, which is a
    # worse failure than the one this guard exists to prevent.
    assert _rendered_body_is_empty(rendered) is False


@pytest.mark.asyncio
async def test_an_event_with_a_blank_summary_is_refused_rather_than_written():
    fs = _RecordingFs()
    op = ResolvedOperation(
        memory_type="events",
        memory_fields={"event_name": "nothing_happened", "goal": "", "summary": "   "},
        uris=["viking://user/u/memories/events/2026/07/25/nothing_happened.md"],
        page_id=1,
    )

    with pytest.raises(ValueError, match="empty body"):
        await _updater(fs)._apply_upsert(op, _ctx(), extract_context=_ExtractContext())

    assert fs.writes == [], "an empty memory must not reach storage"


@pytest.mark.asyncio
async def test_an_event_with_a_real_summary_is_written():
    fs = _RecordingFs()
    op = ResolvedOperation(
        memory_type="events",
        memory_fields={
            "event_name": "store_expanded",
            "goal": "record the expansion",
            "summary": "Gina expanded her clothing store on 2026-07-25.",
        },
        uris=["viking://user/u/memories/events/2026/07/25/store_expanded.md"],
        page_id=1,
    )

    await _updater(fs)._apply_upsert(op, _ctx(), extract_context=_ExtractContext())

    assert len(fs.writes) == 1
    assert "Gina expanded her clothing store" in fs.writes[0][1]


@pytest.mark.asyncio
async def test_the_refusal_is_reported_and_does_not_take_down_the_batch():
    # events are add_only, so a refusal must surface as an error against that
    # URI while the rest of the batch proceeds — not silently, and not fatally.
    fs = _RecordingFs()
    updater = _updater(fs)
    empty = ResolvedOperation(
        memory_type="events",
        memory_fields={"event_name": "blank", "goal": "", "summary": ""},
        uris=["viking://user/u/memories/events/2026/07/25/blank.md"],
        page_id=1,
    )

    with pytest.raises(ValueError):
        await updater._apply_upsert(empty, _ctx(), extract_context=_ExtractContext())

    # The apply loop turns this into result.add_error(uri, e) and continues for
    # non-experience types; what matters here is that it raises rather than
    # returning quietly, so the caller can record it.
    assert fs.writes == []


@pytest.mark.asyncio
async def test_a_failed_template_render_is_refused_rather_than_stored_as_metadata():
    """The latent failure the guard also closes.

    Events carry no `content` field — their whole body comes from the template.
    When rendering raises, MemoryFileUtils falls back to a plain-content file,
    which for events is the MEMORY_FIELDS comment and nothing else. Before this
    guard that wrote silently, producing a store full of well-formed events with
    no readable content.
    """
    fs = _RecordingFs()
    op = ResolvedOperation(
        memory_type="events",
        memory_fields={
            "event_name": "store_expanded",
            "goal": "record the expansion",
            "summary": "Gina expanded her clothing store on 2026-07-25.",
        },
        uris=["viking://user/u/memories/events/2026/07/25/store_expanded.md"],
        page_id=1,
    )

    # extract_context=None makes the events template raise mid-render.
    with pytest.raises(ValueError, match="empty body"):
        await _updater(fs)._apply_upsert(op, _ctx(), extract_context=None)

    assert fs.writes == []


def test_the_guard_is_wired_into_the_write_path():
    """A correct predicate that nothing calls would protect nothing."""
    import inspect

    from openviking.session.memory import memory_updater

    source = inspect.getsource(memory_updater.MemoryUpdater._apply_upsert)
    assert "_rendered_body_is_empty" in source
    guard = source.index("_rendered_body_is_empty")
    write = source.index("write_file(")
    assert guard < write, "the guard must run before the write, not after"
